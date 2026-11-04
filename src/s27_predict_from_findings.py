"""27단계: 분석 결과를 피처로 넣으면 예측이 좋아지나
A. 행 단위(test 한 행 + 상수만): 20단계 rt_* + 챔버 저온·UV45·Dam 고속 플래그
B. 현장 흐름(전체 생산 로그, 라벨 없음, 그 시점까지 알 수 있는 정보): 탈포 배치 크기, 탈포기 휴지, 그날 첫 탈포,
   하루 순번, 대기 중 라인 정지, 휴식 직전 여부
   - 배치 크기: 같은 탈포 시각에 함께 들어간 제품 수 → 탈포 투입 시점에 확정되는 정보
   - 라벨(불량 여부)은 어떤 피처에도 쓰지 않음 (검사 결과는 늦게 나오므로)
   - test 파일 '한 행만'으로는 만들 수 없음. 실시간 생산 로그가 있어야 함
평가: 9월+ 알려진 원인 제외 / train 전체 · 5-fold CV + 시간 순(~2024-02 → 03~04, 시드 3개 평균) · 부트스트랩 CI
결과: reports/27_predict_from_findings.json
"""
import pandas as pd, numpy as np, json, warnings
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score
warnings.filterwarnings('ignore'); SEED = 42
exec(open('src/s20_ts_features.py', encoding='utf-8').read().split('# ② 평가')[0])   # d(+rt_*), RTF, dev_fit, MIN

# A. 행 단위 플래그
d['fA_cold46'] = (d['AutoClave__Chamber Temp.'] <= 46).astype(float)
d['fA_uv45'] = (d['Fill2__CURE SPEED'] == 45).astype(float)
d['fA_fast'] = (d['Dam__Stage2 Line1 Distance Speed'] >= 6500).astype(float)
A_F = ['fA_cold46', 'fA_uv45', 'fA_fast']
# B. 현장 흐름 피처 (라벨 미사용)
d['fB_ac_batch'] = d.groupby('date_AutoClave')['Set ID'].transform('size').astype(float)
u = np.sort(d.date_AutoClave.unique()); prev = pd.Series(np.r_[np.datetime64('NaT'), u[:-1]], index=u)
pv = d.date_AutoClave.map(prev); d['fB_ac_idle'] = (d.date_AutoClave - pv).dt.total_seconds() / 60
d['fB_ac_first_of_day'] = (pv.dt.normalize() != d.date_AutoClave.dt.normalize()).astype(float)
d.loc[d.fB_ac_first_of_day == 1, 'fB_ac_idle'] = np.nan
d = d.sort_values(['line', 'date_Dam', 'Dam__Production Qty'])
d['fB_day_seq'] = d.groupby(['line', d.date_Dam.dt.normalize()]).cumcount().astype(float) + 1
d = d.sort_index()
# 대기 중 같은 라인 Fill2 기록 공백(최대, 분) — 탈포 투입 시점에 아는 정보
mg = pd.Series(0.0, index=d.index)
for ln, g in d.groupby('line'):
    f2 = np.unique(g.date_Fill2.values); a, b = g.date_Fill2.values, g.date_AutoClave.values
    lo, hi = np.searchsorted(f2, a, 'right'), np.searchsorted(f2, b, 'left')
    mg.loc[g.index] = [(np.diff(np.r_[x, f2[l_:h_], y]) / np.timedelta64(1, 'm')).max() for x, y, l_, h_ in zip(a, b, lo, hi)]
d['fB_wait_max_gap'] = mg; d['fB_wait_stopped_no_break'] = ((mg >= 20) & (d.rt_wait_spans_break == 0)).astype(float)
B_F = ['fB_ac_batch', 'fB_ac_idle', 'fB_ac_first_of_day', 'fB_day_seq', 'fB_wait_max_gap', 'fB_wait_stopped_no_break']

LOOKAHEAD = ['fe_wo_size', 'fe_wo_span_h', 'fe_wo_rank', 'fe_qty_ratio', 'wo_te']
FS = json.load(open('reports/05_feature_sets.json'))['process_only']
GROUPED = [f for f in FS if f.startswith('dev_') or f.startswith('fe_') and f.endswith('coord_dev_abs')]
base = [f for f in FS if f not in LOOKAHEAD and f != 'fe_days' and f not in GROUPED] + RTF
P_ = dict(n_estimators=600, learning_rate=0.02, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
          colsample_bytree=0.5, reg_lambda=5, verbose=-1, random_state=SEED)
tr = d[d.is_train == 1].reset_index(drop=True); tr['y'] = tr.target.astype(int)
rule = (tr.fe_long_wait == 1) | (tr.equip_mismatch == 1) | (tr.fe_Fill1_zero_dispense == 1)
POP = {'sep_known_removed': tr[(tr.date_Dam >= '2023-09-01') & ~rule & ~(tr['dev_Dam__Y(Stage2)'] < -0.05)].reset_index(drop=True), 'all_train': tr}
rng = np.random.default_rng(SEED)


def fit_pred(A, B, fs, seed=SEED):
    XA, XB = pd.concat([A[fs], dev_fit(A, A)], axis=1), pd.concat([B[fs], dev_fit(A, B)], axis=1)
    m = lgb.LGBMClassifier(**(P_ | dict(random_state=seed))).fit(XA, A.y); return m.predict_proba(XB)[:, 1], m, list(XA.columns)


def boot(yv, p1, p2, n=1000):
    ds = [average_precision_score(yv[i], p2[i]) - average_precision_score(yv[i], p1[i])
          for i in (rng.integers(0, len(yv), len(yv)) for _ in range(n)) if yv[i].sum()]
    return [round(float(np.percentile(ds, 2.5)), 4), round(float(np.percentile(ds, 97.5)), 4)]


def topk(yv, p, k):
    c = np.quantile(p, 1 - k / 100); s = p >= c; return dict(precision=round(float(yv[s].mean()), 3), recall=round(float(yv[s].sum() / yv.sum()), 3))


res = {}
for name, t in POP.items():
    y = t.y.values; cut = pd.Timestamp('2024-03-01'); A, B = t[t.date_Dam < cut], t[t.date_Dam >= cut]; r = {}; keep = {}
    for tag, fs in [('base', base), ('A: +row flags', base + A_F), ('B: +stream', base + A_F + B_F)]:
        oof = np.zeros(len(t))
        for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(t, y): oof[b] = fit_pred(t.iloc[a], t.iloc[b], fs)[0]
        pb = np.mean([fit_pred(A, B, fs, s)[0] for s in (SEED, 1, 2)], 0)
        r[tag] = dict(cv_pr_auc=round(average_precision_score(y, oof), 4), time_pr_auc=round(average_precision_score(B.y, pb), 4),
                      time_roc_auc=round(roc_auc_score(B.y, pb), 4), time_top3=topk(B.y.values, pb, 3), time_top5=topk(B.y.values, pb, 5))
        keep[tag] = (oof, pb)
        if tag.startswith('B'):
            _, m, cols = fit_pred(A, B, fs); imp = pd.Series(m.booster_.feature_importance('gain'), cols)
            r['B_top_new'] = (imp[A_F + B_F] / imp.sum()).sort_values(ascending=False).round(4).to_dict()
    for tag in ('A: +row flags', 'B: +stream'):
        r[tag]['cv_diff_ci'] = boot(y, keep['base'][0], keep[tag][0]); r[tag]['time_diff_ci'] = boot(B.y.values, keep['base'][1], keep[tag][1])
    r['base_rate'] = round(float(y.mean()), 4); r['time_base_rate'] = round(float(B.y.mean()), 4)
    res[name] = r; print(name, json.dumps(r, ensure_ascii=False, indent=1), flush=True)
json.dump(dict(A=A_F, B=B_F, results=res), open('reports/27_predict_from_findings.json', 'w'), ensure_ascii=False, indent=1)
