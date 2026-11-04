"""29단계: (1) 검사 결과가 몇 시간 뒤에 나오면 예측이 얼마나 좋아지나 (민감도) (2) 단순한 모델이 시간 변화에 더 강한가
(1) B++(lag): 같은 작업지시·같은 라인에서 '탈포 후 lag시간이 지난' train 제품의 검사 결과만 사용 (lag = 2 · 4 · 8 · 24시간)
    자기 자신은 자동 제외(자기 탈포 시각 + lag > 자기 탈포 시각). 기준 시각 = 대상 제품의 탈포 투입 시각
(2) 단순 모델: ① 견고한 조건 + 탐지 신호만 넣은 로지스틱 ② 규제 강화 LightGBM(얕은 트리, 강한 L2)
평가: 28단계와 같은 월별 롤링(2023-12 ~ 2024-04), train 전체 · 9월+ 알려진 원인 제외
결과: reports/29_feedback_and_simple.json, figs/29_feedback.png
"""
import pandas as pd, numpy as np, json, warnings
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import average_precision_score
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
warnings.filterwarnings('ignore'); SEED = 42
src = open('src/s28_stream_model.py', encoding='utf-8').read()
exec(src.split('res = {}')[0])     # d, SETS, POP 정의, fit_pred, MONTHS (28단계와 동일)

lab = d[d.is_train == 1][['workorder', 'line', 'date_AutoClave', 'target']].assign(y=lambda x: x.target.astype(int))


def lag_rate(key, lag_h, K=10, prior=0.058):
    """대상 제품의 탈포 시각 t 기준, 같은 key에서 (탈포 시각 + lag) <= t 인 train 제품의 불량률(평활)"""
    out_r = pd.Series(np.nan, index=d.index); out_n = pd.Series(0.0, index=d.index)
    for kv, g in d.groupby(key):
        L = lab[lab[key] == kv].sort_values('date_AutoClave')
        if L.empty: out_r[g.index] = prior; continue
        avail = (L.date_AutoClave + pd.Timedelta(hours=lag_h)).values; cy = np.r_[0, np.cumsum(L.y.values)]
        k = np.searchsorted(avail, g.date_AutoClave.values, 'right')
        out_n[g.index] = k; out_r[g.index] = (cy[k] + K * prior) / (k + K)
    return out_r, out_n


LAGS = [2, 4, 8, 24]
for h in LAGS:
    d[f'fL{h}_wo_rate'], d[f'fL{h}_wo_n'] = lag_rate('workorder', h)
    d[f'fL{h}_line_rate'], _ = lag_rate('line', h)
# 누수 점검: 어떤 제품의 라벨을 뒤집어도, 그 제품 자신과 '탈포 시각 < 그 제품 탈포+lag' 인 제품의 피처는 그대로
i0 = lab.index[len(lab) // 2]; t0 = lab.loc[i0, 'date_AutoClave']; wo0 = lab.loc[i0, 'workorder']
before = d.loc[(d.workorder == wo0), 'fL4_wo_rate'].copy()
lab.loc[i0, 'y'] = 1 - lab.loc[i0, 'y']; r2, _ = lag_rate('workorder', 4); lab.loc[i0, 'y'] = 1 - lab.loc[i0, 'y']
m_ = (d.workorder == wo0) & (d.date_AutoClave < t0 + pd.Timedelta(hours=4))
leak_ok = bool(np.allclose(before[m_[d.workorder == wo0].values].values, r2[m_].values))
print('lag leakage check:', leak_ok); assert leak_ok

tr = d[d.is_train == 1].reset_index(drop=True); tr['y'] = tr.target.astype(int); tr['mo'] = tr.date_Dam.dt.to_period('M').astype(str)
rule = (tr.fe_long_wait == 1) | (tr.equip_mismatch == 1) | (tr.fe_Fill1_zero_dispense == 1)
POP = {'all_train': tr, 'sep_known_removed': tr[(tr.date_Dam >= '2023-09-01') & ~rule & ~(tr['dev_Dam__Y(Stage2)'] < -0.05)].reset_index(drop=True)}
Bfs = SETS['B (실시간 로그)']
VAR = {'base (test 한 행)': ('lgb', SETS['base (test 한 행)']), 'B (실시간 로그)': ('lgb', Bfs)}
for h in LAGS: VAR[f'B + 검사 {h}h 후'] = ('lgb', Bfs + [f'fL{h}_wo_rate', f'fL{h}_wo_n', f'fL{h}_line_rate'])
SIMPLE = ['fe_long_wait', 'equip_mismatch', 'fe_Fill1_zero_dispense', 'fe_lead_Fill2_AC', 'fe_lead_total', 'fB_wait_stopped_no_break',
          'fB_small_batch', 'fB_ac_batch', 'fA_cold46', 'fA_uv45', 'fA_fast', 'fe_pxx_order', 'rt_wait_excess', 'fB_day_seq', 'line']
VAR['단순 로지스틱 (견고 조건 15개)'] = ('logit', SIMPLE)
VAR['작은 LightGBM (B)'] = ('lgb_small', Bfs)


def pred(kind, A, B, fs, seed):
    if kind == 'logit':
        XA, XB = A[fs].astype(float), B[fs].astype(float); med = XA.median()
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000)).fit(XA.fillna(med), A.y); return m.predict_proba(XB.fillna(med))[:, 1]
    if kind == 'lgb_small':
        XA, XB = pd.concat([A[fs], dev_fit(A, A)], axis=1), pd.concat([B[fs], dev_fit(A, B)], axis=1)
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=7, max_depth=3, min_child_samples=150, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.5, reg_lambda=30, verbose=-1, random_state=seed).fit(XA, A.y); return m.predict_proba(XB)[:, 1]
    return fit_pred(A, B, fs, seed)[0]


res = {}
for pn, t in POP.items():
    r = {}
    for tag, (kind, fs) in VAR.items():
        rows = []
        for mo in MONTHS:
            A, B = t[t.mo < mo], t[t.mo == mo]
            p = np.mean([pred(kind, A, B, fs, s) for s in ((SEED,) if kind == 'logit' else (SEED, 7))], 0)
            k = max(1, int(len(B) * .03)); top = np.argsort(-p)[:k]
            rows.append(dict(month=mo, pr_auc=round(average_precision_score(B.y, p), 4), lift=round(average_precision_score(B.y, p) / B.y.mean(), 2),
                             top3_precision=round(float(B.y.values[top].mean()), 3), top3_recall=round(float(B.y.values[top].sum() / B.y.sum()), 3)))
        R = pd.DataFrame(rows); r[tag] = dict(monthly=rows, mean_pr_auc=round(R.pr_auc.mean(), 4), mean_lift=round(R.lift.mean(), 2),
                                             mean_top3_precision=round(R.top3_precision.mean(), 3), mean_top3_recall=round(R.top3_recall.mean(), 3))
        print(f'{pn:18s} {tag:28s} PR-AUC {r[tag]["mean_pr_auc"]:.4f} lift {r[tag]["mean_lift"]} top3 P {r[tag]["mean_top3_precision"]} R {r[tag]["mean_top3_recall"]}', flush=True)
    res[pn] = r
json.dump(dict(leak_check=leak_ok, results=res), open('reports/29_feedback_and_simple.json', 'w'), ensure_ascii=False, indent=1)

fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
for a, pn in zip(ax, POP):
    xs = ['base', 'B'] + [f'+{h}h' for h in LAGS]; ys = [res[pn]['base (test 한 행)']['mean_lift'], res[pn]['B (실시간 로그)']['mean_lift']] + [res[pn][f'B + 검사 {h}h 후']['mean_lift'] for h in LAGS]
    a.plot(xs, ys, 'o-', color='#c0392b'); [a.text(i, y + .03, f'{y:.2f}x', ha='center', fontsize=8) for i, y in enumerate(ys)]
    a.set_ylabel('mean lift (PR-AUC / base rate)'); a.set_xlabel('information available (inspection results after N hours)'); a.set_title(f'Value of faster inspection feedback — {pn}', fontsize=10)
plt.tight_layout(); plt.savefig('figs/29_feedback.png', dpi=120, bbox_inches='tight')
