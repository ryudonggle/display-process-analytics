"""16단계: 2023-09 이후 · 규칙(132건) · Dam Y(Stage2) 음수 편차 제외 데이터로 04(통계)·05(모델)·07(SHAP) 재실행
- 근거: 불량률이 2023-08 7.0% → 09 4.3%로 떨어진 시점에 바뀐 것은 Dam Y 음수 편차 비율(47% → 1%)뿐 (레시피 변경은 10월)
- 이미 설명되는 불량(규칙·Y 정렬)을 빼고 '아직 모르는 원인'만 본다
- 비교 기준: 전체 기간(규칙만 제외)으로 학습한 모델을 같은 평가 행에서 채점
결과: reports/16_sep_*.csv/json, figs/16_sep_*.png
"""
import pandas as pd, numpy as np, json, warnings
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import lightgbm as lgb, shap
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
warnings.filterwarnings('ignore'); SEED = 42

d = pd.read_parquet('data/features.parquet')
d['fe_long_wait'] = (d.fe_lead_total > 2).astype(int); d['fe_pxx_order'] = d.workorder.str.contains('PXX').astype(int)
d['rule'] = ((d.fe_long_wait == 1) | (d.equip_mismatch == 1) | (d.fe_Fill1_zero_dispense == 1)).astype(int)
d['yneg'] = (d['dev_Dam__Y(Stage2)'] < -0.05).astype(int)
for c in ['model', 'fe_wo_prefix', 'fe_wo_suffix']: d[c + '_code'] = d[c].astype('category').cat.codes
d['month'] = d.date_Dam.dt.to_period('M').astype(str)

tr = d[d.is_train == 1].reset_index(drop=True)
sep = (tr.date_Dam >= '2023-09-01') & (tr.rule == 0) & (tr.yneg == 0)
t = tr[sep].reset_index(drop=True); y = t.target.astype(int)
base_rows = tr[tr.rule == 0].reset_index(drop=True)   # 비교용: 전체 기간, 규칙만 제외
LOOKAHEAD = ['fe_wo_size', 'fe_wo_span_h', 'fe_wo_rank', 'fe_qty_ratio', 'wo_te']
FS = [f for f in json.load(open('reports/05_feature_sets.json'))['process_only']
      if f not in ('fe_long_wait', 'equip_mismatch', 'fe_Fill1_zero_dispense', 'fe_lead_total', 'fe_lead_Fill2_AC')]
FS = [f for f in FS if t[f].nunique() > 1]
P = dict(n_estimators=600, learning_rate=0.02, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.5, reg_lambda=5, verbose=-1, random_state=SEED)
met = lambda yt, p: dict(roc_auc=round(roc_auc_score(yt, p), 4), pr_auc=round(average_precision_score(yt, p), 4),
                         base_rate=round(float(np.mean(yt)), 4), lift=round(average_precision_score(yt, p) / np.mean(yt), 2))
out = dict(data=dict(n=len(t), defects=int(y.sum()), rate=round(y.mean(), 4)))

# ── 1) 통계 (s04 대응): Mann-Whitney + FDR → 월 고정효과 + 모델 + 라인 통제 로지스틱
EXCL = {'fe_days', 'fe_recipe_new', 'fe_hour', 'fe_weekday', 'line', 'equip_Dam', 'equip_Fill1', 'equip_Fill2', 'is_train', 'target',
        'rule', 'yneg', 'fe_long_wait'}
num = [c for c in t.columns if c.startswith(('Dam__', 'Fill1__', 'Fill2__', 'AutoClave__', 'fe_', 'dev_')) and c not in EXCL
       and pd.api.types.is_numeric_dtype(t[c]) and t[c].nunique() > 1]
rows = []
for c in num:
    a, b = t.loc[y == 1, c].dropna(), t.loc[y == 0, c].dropna()
    u, p = stats.mannwhitneyu(a, b, alternative='two-sided')
    rows.append(dict(feature=c, r=2 * u / (len(a) * len(b)) - 1, p=p, med_def=a.median(), med_ok=b.median()))
R = pd.DataFrame(rows); R['q'] = multipletests(R.p, method='fdr_bh')[1]; R['abs_r'] = R.r.abs()
R = R.sort_values('abs_r', ascending=False)
ctrl = pd.get_dummies(t[['month', 'model']], drop_first=True, dtype=float); ctrl['line2'] = (t.line == 2).astype(float)
adj = []
for c in R[R.q < .05].feature.head(25):
    X = ctrl.copy(); X['z'] = (t[c] - t[c].mean()) / t[c].std(); X = sm.add_constant(X.fillna(0))
    try:
        m = sm.Logit(y, X).fit(disp=0, maxiter=200); ci = m.conf_int().loc['z']
        adj.append(dict(feature=c, OR_per_SD=np.exp(m.params['z']), lo=np.exp(ci[0]), hi=np.exp(ci[1]), p_adj=m.pvalues['z']))
    except Exception:
        pass
A = pd.DataFrame(adj)
A['confirmed'] = A.p_adj < .05
R = R.merge(A, on='feature', how='left')
old = pd.read_csv('reports/04_univariate_tests.csv'); old_robust = set(old[old.robust].feature)
R['was_robust_04'] = R.feature.isin(old_robust)
R.to_csv('reports/16_sep_stats.csv', index=False)
conf = R[R.confirmed == True]
out['stats'] = dict(n_tested=len(R), n_sig=int((R.q < .05).sum()), n_confirmed_month_adj=int(len(conf)),
                    confirmed_also_in_04=int(conf.was_robust_04.sum()), confirmed_new=conf[~conf.was_robust_04].feature.tolist())

# ── 2) 모델 (s05·s08 대응): 같은 평가 행에서 '9월 이후 학습' vs '전체 기간 학습'
#   (a) 5-fold CV: 평가 행 = 9월+ 서브셋 OOF. 전체 기간 모델은 서브셋 밖 행을 학습에 추가로 사용
rt = [f for f in FS if f not in LOOKAHEAD and f != 'fe_days']
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
key = t['Set ID']; outside = base_rows[~base_rows['Set ID'].isin(key)]
oof_sep, oof_all = np.zeros(len(t)), np.zeros(len(t))
for a, b in skf.split(t, y):
    oof_sep[b] = lgb.LGBMClassifier(**P).fit(t.iloc[a][rt], y.iloc[a]).predict_proba(t.iloc[b][rt])[:, 1]
    A2 = pd.concat([t.iloc[a], outside])
    oof_all[b] = lgb.LGBMClassifier(**P).fit(A2[rt], A2.target.astype(int)).predict_proba(t.iloc[b][rt])[:, 1]
out['cv'] = dict(train_sep=met(y, oof_sep), train_all_period=met(y, oof_all))
#   (b) 시간 순: ~2024-02 학습 → 2024-03~04 서브셋 평가 (s08과 같은 기준일)
cut = pd.Timestamp('2024-03-01'); B = t[t.date_Dam >= cut]; yb = B.target.astype(int)
ps = lgb.LGBMClassifier(**P).fit(t[t.date_Dam < cut][rt], t[t.date_Dam < cut].target.astype(int)).predict_proba(B[rt])[:, 1]
Aall = base_rows[base_rows.date_Dam < cut]
pa = lgb.LGBMClassifier(**P).fit(Aall[rt], Aall.target.astype(int)).predict_proba(B[rt])[:, 1]
out['time_split'] = dict(n_val=len(B), train_sep=met(yb, ps), train_all_period=met(yb, pa))
for k in (1, 3, 5):
    c = np.quantile(ps, 1 - k / 100); s = ps >= c
    out['time_split'][f'top{k}pct_sep'] = dict(precision=round(yb[s].mean(), 3), recall=round(yb[s].sum() / yb.sum(), 3))

# ── 3) SHAP (s07 대응): 80% 학습 / 20% 설명
Xa, Xb, ya, yb2 = train_test_split(t[FS], y, test_size=0.2, stratify=y, random_state=SEED)
m = lgb.LGBMClassifier(**P).fit(Xa, ya); pb = m.predict_proba(Xb)[:, 1]
ex = shap.TreeExplainer(m); E = ex(Xb)
if E.values.ndim == 3: E = E[..., 1]
assert np.abs(np.asarray(E.base_values) + E.values.sum(1) - m.predict(Xb, raw_score=True)).max() < 1e-3
SV = pd.DataFrame(E.values, columns=FS); imp = SV.abs().mean().sort_values(ascending=False)
imp.to_csv('reports/16_sep_shap_importance.csv', header=['mean_abs_shap'])
def proc(f):
    for p, keys in [('Dam', ('Dam', 'fe_Dam')), ('Fill1', ('Fill1', 'fe_Fill1')), ('Fill2(UV)', ('Fill2', 'fe_Fill2')),
                    ('AutoClave(탈포)', ('AutoClave', 'fe_AC'))]:
        if f.startswith(keys) or f.startswith('dev_' + keys[0] + '__'): return p
    if f.startswith(('fe_lead', 'fe_hour', 'fe_weekday', 'fe_wo', 'fe_qty', 'fe_tact', 'fe_pxx')): return '운영(대기·작업지시)'
    if f.startswith('fe_cure'): return 'Fill2(UV)'
    return '기타(모델·라인)'
pg = imp.groupby(imp.index.map(proc)).sum(); pg = (pg / pg.sum() * 100).round(1).sort_values(ascending=False)
old_pg = json.load(open('reports/07_shap_summary.json'))['process_share']
out['shap'] = dict(holdout_pr_auc=round(average_precision_score(yb2, pb), 4), process_share=pg.to_dict(), process_share_07=old_pg,
                   top15=imp.head(15).round(4).to_dict())
top = imp.head(12).index.tolist(); thr = {}
fig, axes = plt.subplots(3, 4, figsize=(18, 11))
for ax, f in zip(axes.ravel(), top):
    x = Xb[f].values; s = SV[f].values
    ax.scatter(x, s, s=4, alpha=.25, color='#2c3e50'); ax.axhline(0, color='grey', lw=.8)
    v = pd.Series(s).groupby(x).agg(['mean', 'size'])
    if len(v) > 25:
        q = pd.qcut(x, 20, duplicates='drop'); v = pd.Series(s).groupby(q, observed=True).agg(['mean', 'size']); v.index = [i.mid for i in v.index]
    v = v[v['size'] >= 30]; ax.plot(v.index, v['mean'], 'o-', color='#c0392b', ms=3)
    thr[f] = dict(risk_values=[round(float(r), 3) for r in v[v['mean'] > 0.05].index][:15],
                  safe_values=[round(float(r), 3) for r in v[v['mean'] < -0.05].index][:15])
    ax.set_title(f[:42], fontsize=9); ax.set_ylabel('SHAP (log-odds)')
plt.tight_layout(); plt.savefig('figs/16_sep_shap_dependence.png', dpi=100); plt.close()
plt.figure(); shap.plots.beeswarm(E, max_display=20, show=False); plt.title('SHAP beeswarm — 2023-09+ (rule·Y- excluded)')
plt.tight_layout(); plt.savefig('figs/16_sep_shap_beeswarm.png', dpi=110, bbox_inches='tight'); plt.close()
out['shap']['thresholds'] = thr

json.dump(out, open('reports/16_sep_summary.json', 'w'), ensure_ascii=False, indent=1, default=str)
pd.set_option('display.width', 250)
print(json.dumps({k: v for k, v in out.items() if k != 'shap'}, ensure_ascii=False, indent=1, default=str))
print('SHAP', out['shap']['holdout_pr_auc'], pg.to_dict(), '\n07:', old_pg)
print(imp.head(15).round(4))
print(R[['feature', 'r', 'q', 'OR_per_SD', 'lo', 'hi', 'p_adj', 'confirmed', 'was_robust_04', 'med_def', 'med_ok']].head(30).round(4).to_string(index=False))
for f in top[:10]: print(f, thr[f])
