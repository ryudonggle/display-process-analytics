"""28단계: B 가정(실시간 생산 로그 사용 가능)에서 예측 성능 끌어올리기
가정
- B : 전체 생산 흐름의 시각·설정값(라벨 없음)을 그 시점까지 쓸 수 있다 → 탈포 배치, 라인 정지, 순번 등
- B+: 추가로 '전날까지' 검사가 끝난 제품의 결과(train 라벨)를 쓸 수 있다 → 작업지시·라인의 과거 불량률
       당일 제품 · 자기 자신의 라벨은 절대 사용하지 않음. test 라벨은 없으므로 train 라벨만 (실제 검사의 약 70% 표본에 해당)
평가: 월별 롤링 — 2023-12 ~ 2024-04 각 달을 미래로 두고 그 이전 전체로 학습 (시드 2개 평균). 집단: train 전체 / 9월+ 알려진 원인 제외
결과: reports/28_stream_model.json, figs/28_stream_model.png
"""
import pandas as pd, numpy as np, json, warnings
import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
warnings.filterwarnings('ignore'); SEED = 42
exec(open('src/s27_predict_from_findings.py', encoding='utf-8').read().split('LOOKAHEAD = ')[0])   # d + rt_* + fA_* + fB_* (라벨 미사용)

# ── B: 흐름 피처 보강 (라벨 미사용)
d['day'] = d.date_Dam.dt.normalize()
d['fB_wo_n_before'] = d.sort_values(['date_Dam', 'Dam__Production Qty']).groupby('workorder').cumcount().astype(float)   # 지금까지 투입 수
d['fB_wo_days'] = (d.day - d.groupby('workorder').day.transform('min')).dt.days.astype(float)                          # 작업지시 착수 후 경과일
d['fB_small_batch'] = (d.fB_ac_batch <= 3).astype(float)
B2 = B_F + ['fB_wo_n_before', 'fB_wo_days', 'fB_small_batch']

# ── B+: 전날까지의 검사 결과 (train 라벨만, 일 단위 누적에서 당일 제외)
lab = d[d.is_train == 1].assign(y=lambda x: x.target.astype(int))
prior = 0.058; K = 10


def past_rate(keys, name):
    g = lab.groupby(keys + ['day']).y.agg(['sum', 'count']).reset_index().sort_values(keys + ['day'])
    g[['cs', 'cn']] = g.groupby(keys)[['sum', 'count']].cumsum().values - g[['sum', 'count']].values   # 당일 제외 누적
    m = d[keys + ['day']].merge(g[keys + ['day', 'cs', 'cn']], on=keys + ['day'], how='left')
    # 그날 기록이 없는 (작업지시, 일) 조합: 그 이전 마지막 누적값 사용
    if m.cn.isna().any():
        last = g.assign(cs=g.cs + g['sum'], cn=g.cn + g['count'])[keys + ['day', 'cs', 'cn']]
        miss = m.cn.isna()
        q = d.loc[miss.values, keys + ['day']].reset_index()
        q = pd.merge_asof(q.sort_values('day'), last.sort_values('day'), on='day', by=keys, allow_exact_matches=False).set_index('index')
        m.loc[miss.values, ['cs', 'cn']] = q.loc[d.index[miss.values], ['cs', 'cn']].values
    cs, cn = m.cs.fillna(0).values, m.cn.fillna(0).values
    d[f'fC_{name}_rate'] = (cs + K * prior) / (cn + K * prior / prior)    # 평활: 사전확률 쪽으로 K건 가중
    d[f'fC_{name}_n'] = cn


past_rate(['workorder'], 'wo')          # 같은 작업지시(로트)의 전날까지 불량률
past_rate(['line'], 'line')             # 라인 누적 (장기)
lab2 = lab.copy()
# 라인의 직전 작업일 불량률 (단기 상태)
ld = lab.groupby(['line', 'day']).y.mean().rename('r').reset_index().sort_values(['line', 'day'])
ld['prev_r'] = ld.groupby('line').r.shift()
q = d[['line', 'day']].reset_index().sort_values('day')
q = pd.merge_asof(q, ld[['line', 'day', 'r']].rename(columns={'r': 'prev_day_r'}).sort_values('day'), on='day', by='line', allow_exact_matches=False).set_index('index')
d['fC_line_prevday_rate'] = q.loc[d.index, 'prev_day_r'].values
C_F = ['fC_wo_rate', 'fC_wo_n', 'fC_line_rate', 'fC_line_prevday_rate']

# 누수 점검: 당일 이후 라벨을 바꿔도 피처가 그대로인지 (임의의 날짜 이후 train 라벨을 뒤집어 재계산)
chk_day = pd.Timestamp('2024-01-15')
saved = d[C_F].copy(); lab_orig = lab.copy()
lab.loc[lab.day >= chk_day, 'y'] = 1 - lab.loc[lab.day >= chk_day, 'y']
past_rate(['workorder'], 'wo'); past_rate(['line'], 'line')
mask = d.day <= chk_day
leak_ok = bool(np.allclose(saved.loc[mask, ['fC_wo_rate', 'fC_line_rate']].values, d.loc[mask, ['fC_wo_rate', 'fC_line_rate']].values, equal_nan=True))
print('B+ leakage check (flip labels from 2024-01-15 → features up to that day unchanged):', leak_ok); assert leak_ok
lab = lab_orig; d[C_F] = saved

LOOKAHEAD = ['fe_wo_size', 'fe_wo_span_h', 'fe_wo_rank', 'fe_qty_ratio', 'wo_te']
FS = json.load(open('reports/05_feature_sets.json'))['process_only']
GROUPED = [f for f in FS if f.startswith('dev_') or f.startswith('fe_') and f.endswith('coord_dev_abs')]
base = [f for f in FS if f not in LOOKAHEAD and f != 'fe_days' and f not in GROUPED] + RTF
SETS = {'base (test 한 행)': base, 'B (실시간 로그)': base + A_F + B2, 'B+ (전날 검사 결과)': base + A_F + B2 + C_F}
P_ = dict(n_estimators=600, learning_rate=0.02, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
          colsample_bytree=0.5, reg_lambda=5, verbose=-1, random_state=SEED)
tr = d[d.is_train == 1].reset_index(drop=True); tr['y'] = tr.target.astype(int); tr['mo'] = tr.date_Dam.dt.to_period('M').astype(str)
rule = (tr.fe_long_wait == 1) | (tr.equip_mismatch == 1) | (tr.fe_Fill1_zero_dispense == 1)
POP = {'all_train': tr, 'sep_known_removed': tr[(tr.date_Dam >= '2023-09-01') & ~rule & ~(tr['dev_Dam__Y(Stage2)'] < -0.05)].reset_index(drop=True)}
MONTHS = ['2023-12', '2024-01', '2024-02', '2024-03', '2024-04']


def fit_pred(A, B, fs, seed):
    XA, XB = pd.concat([A[fs], dev_fit(A, A)], axis=1), pd.concat([B[fs], dev_fit(A, B)], axis=1)
    m = lgb.LGBMClassifier(**(P_ | dict(random_state=seed))).fit(XA, A.y); return m.predict_proba(XB)[:, 1], m, list(XA.columns)


res = {}
for pn, t in POP.items():
    r = {}
    for tag, fs in SETS.items():
        rows, allp, ally = [], [], []
        for mo in MONTHS:
            A, B = t[t.mo < mo], t[t.mo == mo]
            p = np.mean([fit_pred(A, B, fs, s)[0] for s in (SEED, 7)], 0)
            k = max(1, int(len(B) * .03)); top = np.argsort(-p)[:k]
            rows.append(dict(month=mo, n=len(B), base=round(float(B.y.mean()), 4), pr_auc=round(average_precision_score(B.y, p), 4),
                             lift=round(average_precision_score(B.y, p) / B.y.mean(), 2), top3_precision=round(float(B.y.values[top].mean()), 3),
                             top3_recall=round(float(B.y.values[top].sum() / B.y.sum()), 3)))
            allp.append(p); ally.append(B.y.values)
        R = pd.DataFrame(rows)
        r[tag] = dict(monthly=rows, mean_pr_auc=round(R.pr_auc.mean(), 4), mean_lift=round(R.lift.mean(), 2), mean_top3_precision=round(R.top3_precision.mean(), 3),
                      mean_top3_recall=round(R.top3_recall.mean(), 3), wins_vs_base=None)
    bm = pd.DataFrame(r['base (test 한 행)']['monthly']).pr_auc.values
    for tag in SETS: r[tag]['wins_vs_base'] = int((pd.DataFrame(r[tag]['monthly']).pr_auc.values > bm).sum())
    # 마지막 달 모델의 새 피처 중요도
    A, B = t[t.mo < MONTHS[-1]], t[t.mo == MONTHS[-1]]; _, m, cols = fit_pred(A, B, SETS['B+ (전날 검사 결과)'], SEED)
    imp = pd.Series(m.booster_.feature_importance('gain'), cols); r['importance_new'] = (imp[A_F + B2 + C_F] / imp.sum()).sort_values(ascending=False).round(4).head(10).to_dict()
    res[pn] = r
    print('=====', pn); [print(f'{tag:22s} PR-AUC {v["mean_pr_auc"]:.4f}  lift {v["mean_lift"]}  top3 P {v["mean_top3_precision"]} R {v["mean_top3_recall"]}  wins {v["wins_vs_base"]}/5') for tag, v in r.items() if tag in SETS]
    print('importance', r['importance_new'], flush=True)
json.dump(dict(leak_check=leak_ok, sets={k: len(v) for k, v in SETS.items()}, results=res), open('reports/28_stream_model.json', 'w'), ensure_ascii=False, indent=1)

fig, ax = plt.subplots(1, 2, figsize=(14, 4.4))
for a, pn in zip(ax, POP):
    for tag, col in zip(SETS, ['#9fb6d4', '#5b7db1', '#c0392b']):
        M = pd.DataFrame(res[pn][tag]['monthly']); a.plot(M.month, M.lift, 'o-', color=col, label=f'{tag.split(" ")[0]}  (mean {res[pn][tag]["mean_lift"]}x)')
    a.axhline(1, color='grey', ls='--', lw=.8); a.set_ylabel('PR-AUC / base rate (lift)'); a.set_title(f'Rolling monthly forecast — {pn}', fontsize=10); a.legend(fontsize=8)
plt.tight_layout(); plt.savefig('figs/28_stream_model.png', dpi=120, bbox_inches='tight')
