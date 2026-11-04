"""7단계 SHAP 해석
- 설명 대상: LightGBM(process_only 피처, 규칙 해당 행 제외)의 raw margin(log-odds of AbNormal)
- 학습 80% / 설명은 hold-out 20% (stratified) · TreeExplainer(tree_path_dependent) · 가법성 검증
"""
import pandas as pd, numpy as np, json, warnings
import lightgbm as lgb, shap
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings('ignore'); SEED = 42

d = pd.read_parquet('data/features.parquet')
d['fe_long_wait'] = (d.fe_lead_total > 2).astype(int); d['fe_pxx_order'] = d.workorder.str.contains('PXX').astype(int)
d['rule'] = ((d.fe_long_wait == 1) | (d.equip_mismatch == 1) | (d.fe_Fill1_zero_dispense == 1)).astype(int)
for c in ['model', 'fe_wo_prefix', 'fe_wo_suffix']: d[c + '_code'] = d[c].astype('category').cat.codes
FS = [f for f in json.load(open('reports/05_feature_sets.json'))['process_only']
      if f not in ('fe_long_wait', 'equip_mismatch', 'fe_Fill1_zero_dispense', 'fe_lead_total', 'fe_lead_Fill2_AC')]
t = d[(d.is_train == 1) & (d.rule == 0)].reset_index(drop=True); y = t.target.astype(int)
Xa, Xb, ya, yb = train_test_split(t[FS], y, test_size=0.2, stratify=y, random_state=SEED)
m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.02, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
                       colsample_bytree=0.5, reg_lambda=5, verbose=-1, random_state=SEED).fit(Xa, ya)
pb = m.predict_proba(Xb)[:, 1]
perf = dict(holdout_auc=roc_auc_score(yb, pb), holdout_pr_auc=average_precision_score(yb, pb), n_features=len(FS), n_explained=len(Xb))

ex = shap.TreeExplainer(m)  # 배경 없음 → tree_path_dependent, 출력 = raw margin
E = ex(Xb)
if E.values.ndim == 3: E = E[..., 1]
recon = np.asarray(E.base_values) + E.values.sum(1)
perf['additivity_max_err'] = float(np.abs(recon - m.predict(Xb, raw_score=True)).max())
assert perf['additivity_max_err'] < 1e-3

SV = pd.DataFrame(E.values, columns=FS)
imp = SV.abs().mean().sort_values(ascending=False)
imp.to_csv('reports/07_shap_importance.csv', header=['mean_abs_shap'])

# 공정 단위 기여도
def proc(f):
    for p, keys in [('Dam', ('Dam', 'fe_Dam')), ('Fill1', ('Fill1', 'fe_Fill1')), ('Fill2(UV)', ('Fill2', 'fe_Fill2')),
                    ('AutoClave(탈포)', ('AutoClave', 'fe_AC'))]:
        if f.startswith(keys) or f.startswith(('dev_' + keys[0] + '__',)): return p
    if f.startswith(('fe_lead', 'fe_hour', 'fe_weekday', 'fe_wo', 'fe_qty', 'fe_tact', 'fe_pxx')): return '운영(대기·작업지시)'
    if f.startswith('fe_cure'): return 'Fill2(UV)'
    return '기타(모델·라인)'
pg = imp.groupby(imp.index.map(proc)).sum().sort_values(ascending=False); pg = (pg / pg.sum() * 100).round(1)

# 상위 피처의 SHAP 의존 곡선 → 위험 구간(평균 SHAP>0) 도출
top = imp.head(12).index.tolist(); thr = {}
fig, axes = plt.subplots(3, 4, figsize=(18, 11))
for ax, f in zip(axes.ravel(), top):
    x = Xb[f].values; s = SV[f].values
    ax.scatter(x, s, s=4, alpha=.25, color='#2c3e50'); ax.axhline(0, color='grey', lw=.8)
    v = pd.Series(s).groupby(x).agg(['mean', 'size'])
    if len(v) > 25:  # 연속형은 20분위 bin
        q = pd.qcut(x, 20, duplicates='drop'); v = pd.Series(s).groupby(q, observed=True).agg(['mean', 'size']); v.index = [i.mid for i in v.index]
    v = v[v['size'] >= 30]
    ax.plot(v.index, v['mean'], 'o-', color='#c0392b', ms=3)
    risk = v[v['mean'] > 0.05].index.tolist(); safe = v[v['mean'] < -0.05].index.tolist()
    thr[f] = dict(risk_values=[round(float(r), 3) for r in risk][:15], safe_values=[round(float(r), 3) for r in safe][:15])
    ax.set_title(f[:42], fontsize=9); ax.set_ylabel('SHAP (log-odds)')
plt.tight_layout(); plt.savefig('figs/07_shap_dependence.png', dpi=100); plt.close()

plt.figure(); shap.plots.beeswarm(E, max_display=20, show=False); plt.title('SHAP beeswarm (AbNormal log-odds)')
plt.tight_layout(); plt.savefig('figs/07_shap_beeswarm.png', dpi=110, bbox_inches='tight'); plt.close()
fig, ax = plt.subplots(figsize=(6, 3.5)); pg.sort_values().plot.barh(ax=ax, color='#c0392b'); ax.set_xlabel('share of mean |SHAP| (%)')
ax.set_title('Contribution by process'); plt.tight_layout(); plt.savefig('figs/07_shap_by_process.png', dpi=110); plt.close()

# 상호작용 (1,000행)
sub = Xb.iloc[:1000]; IV = ex.shap_interaction_values(sub)
IV = IV[1] if isinstance(IV, list) else IV
M = np.abs(IV).mean(0); np.fill_diagonal(M, 0)
pairs = [(FS[i], FS[j], M[i, j] * 2) for i in range(len(FS)) for j in range(i + 1, len(FS))]
pairs = sorted(pairs, key=lambda r: -r[2])[:12]

json.dump(dict(perf=perf, process_share=pg.to_dict(), top20=imp.head(20).round(4).to_dict(), thresholds=thr,
               interactions=[dict(a=a, b=b, strength=round(float(s), 4)) for a, b, s in pairs]),
          open('reports/07_shap_summary.json', 'w'), ensure_ascii=False, indent=1)
print(perf); print(pg); print(imp.head(20).round(4))
for a, b, s in pairs: print(f'{s:.4f}  {a}  ×  {b}')
for f in top[:8]: print(f, thr[f])
