"""20단계: 행 단위 시간 피처 — test(미래) 데이터 한 행만으로 만들 수 있는 것만
허용: 그 제품 행의 값 + 학습 구간에서 미리 정한 상수(휴식 시각표, 좌표 기준 중앙값)
불허: 다른 제품 행이 필요한 피처(같은 라인 흐름, 작업지시 그룹 통계), 라벨, 경과일
- 휴식 시각표는 Fill2 시각 분포에서 반복되는 공백으로 정한 상수. train만 / test만으로 각각 추정해도 같은지 확인
- 기존 dev_ 좌표 편차는 train+test 전체 중앙값을 쓰므로, 학습 구간 중앙값으로 다시 계산(dev_fit)
검증: ① test 행만 넣고 계산한 값 = 전체에서 계산한 값 (행 독립성) ② 5-fold CV + 시간 순(~2024-02 → 03~04) + 부트스트랩
결과: reports/20_ts_features.json, 20_run.log
"""
import pandas as pd, numpy as np, json, warnings
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings('ignore'); SEED = 42
MIN = pd.Timedelta('1min')


def break_schedule(df, thr=0.35):
    """Fill2 완료 시각(10분 칸)별 하루 평균 처리량이 주간 중앙값의 thr 미만인 연속 구간 = 정기 휴식 (08:30~17:00 사이)"""
    t = df.date_Fill2; tod = (t.dt.hour * 60 + t.dt.minute) // 10 * 10
    c = tod.value_counts().sort_index() / t.dt.normalize().nunique()
    c = c[(c.index >= 510) & (c.index < 1020)]; low = c[c < c.median() * thr].index.tolist()
    out, s = [], None
    for m in low:
        if s is None: s = p = m
        elif m == p + 10: p = m
        else: out.append((s, p + 10)); s = p = m
    if s is not None: out.append((s, p + 10))
    return out


def row_feats(R, BRK):
    """R: 제품 행들. 각 행의 값과 상수 BRK(분 단위 [시작, 끝) 목록)만 사용 — 행끼리 참조 없음"""
    F = pd.DataFrame(index=R.index)
    a, b = R.date_Fill2, R.date_AutoClave; day = a.dt.normalize()
    ov = pd.Series(0.0, index=R.index); nxt = pd.Series(np.inf, index=R.index)
    for s, e in BRK:
        bs, be = day + s * MIN, day + e * MIN
        ov += ((np.minimum(b, be) - np.maximum(a, bs)) / MIN).clip(lower=0)
        nxt = np.minimum(nxt, ((bs - a) / MIN).where(bs >= a, np.inf))
    wait = (b - a) / MIN
    F['rt_wait_break_overlap'] = ov                        # 대기 중 휴식과 겹친 시간(분)
    F['rt_wait_spans_break'] = (ov > 0).astype(float)
    F['rt_wait_excess'] = wait - ov                         # 휴식으로 설명되지 않는 대기(분)
    F['rt_long_wait_no_break'] = ((wait > 45) & (ov == 0)).astype(float)   # 비정기 정지 대리 지표
    F['rt_min_to_next_break'] = nxt.replace(np.inf, 600)    # Fill2 완료 후 다음 휴식까지(분)
    hrs = R.date_Fill2.dt.hour
    F['rt_evening'] = (hrs >= 17).astype(float)             # 잔업 시간대
    for c in ['date_Dam', 'date_Fill1', 'date_Fill2', 'date_AutoClave']:
        F[f'rt_tod_{c[5:]}'] = R[c].dt.hour * 60 + R[c].dt.minute
    st = pd.DataFrame({'d1': (R.date_Fill1 - R.date_Dam) / MIN, 'd2': (R.date_Fill2 - R.date_Fill1) / MIN, 'd3': wait})
    F['rt_max_stage_wait'] = st.max(1); F['rt_max_stage'] = st.values.argmax(1).astype(float)
    F['rt_cross_day'] = (R.date_AutoClave.dt.normalize() != R.date_Dam.dt.normalize()).astype(float)
    return F


def dev_fit(fit, apply_):
    """좌표 편차: 학습 행(fit)의 라인·모델별 중앙값을 상수로 저장해 적용"""
    cols = [c for c in fit.columns if 'HEAD NORMAL COORDINATE' in c]; key = lambda x: x.line.astype(str) + '_' + x.model
    med = fit.groupby(key(fit))[cols].median(); M = med.reindex(key(apply_)).set_index(apply_.index)
    D = apply_[cols] - M[cols]; D.columns = ['dvf_' + c for c in cols]
    D['dvf_abs_sum'] = D.abs().sum(1); return D


d = pd.read_parquet('data/features.parquet')
d['fe_long_wait'] = (d.fe_lead_total > 2).astype(int); d['fe_pxx_order'] = d.workorder.str.contains('PXX').astype(int)
for c in ['model', 'fe_wo_prefix', 'fe_wo_suffix']: d[c + '_code'] = d[c].astype('category').cat.codes
tr_all = d[d.is_train == 1]; te_all = d[d.is_train == 0]

# 휴식 시각표: train만 / test만 각각 추정 → 같아야 상수로 고정
BRK_tr, BRK_te = break_schedule(tr_all), break_schedule(te_all)
BRK = BRK_tr
print('break schedule train', BRK_tr, '| test only', BRK_te)

# ① 행 독립성: test 행만 넣어 계산 == 전체 계산
full = row_feats(d, BRK); only = row_feats(te_all, BRK)
same = all(np.allclose(full.loc[te_all.index, c].values, only[c].values, equal_nan=True) for c in only.columns)
print('row-independence check (test only == full):', same); assert same
RTF = list(full.columns); d = d.join(full)

# ② 평가
LOOKAHEAD = ['fe_wo_size', 'fe_wo_span_h', 'fe_wo_rank', 'fe_qty_ratio', 'wo_te']
FS = json.load(open('reports/05_feature_sets.json'))['process_only']
GROUPED = [f for f in FS if f.startswith('dev_') or f.startswith('fe_') and f.endswith('coord_dev_abs')]   # 전체 중앙값 사용 → 교체
rt = [f for f in FS if f not in LOOKAHEAD and f != 'fe_days' and f not in GROUPED]
P_ = dict(n_estimators=600, learning_rate=0.02, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
          colsample_bytree=0.5, reg_lambda=5, verbose=-1, random_state=SEED)
tr = d[d.is_train == 1].reset_index(drop=True); tr['y'] = tr.target.astype(int)
rule = (tr.fe_long_wait == 1) | (tr.equip_mismatch == 1) | (tr.fe_Fill1_zero_dispense == 1)
POP = {'all_train': tr, 'sep_known_removed': tr[(tr.date_Dam >= '2023-09-01') & ~rule & ~(tr['dev_Dam__Y(Stage2)'] < -0.05)].reset_index(drop=True)}
rng = np.random.default_rng(SEED)


def boot(yv, p1, p2, n=1000):
    ds = []
    for _ in range(n):
        i = rng.integers(0, len(yv), len(yv))
        if yv[i].sum(): ds.append(average_precision_score(yv[i], p2[i]) - average_precision_score(yv[i], p1[i]))
    return [round(float(np.percentile(ds, 2.5)), 4), round(float(np.percentile(ds, 97.5)), 4)]


def fit_pred(A, B, fs):
    DA, DB = dev_fit(A, A), dev_fit(A, B)                 # 기준 중앙값은 학습 행에서만
    XA, XB = pd.concat([A[fs], DA], axis=1), pd.concat([B[fs], DB], axis=1)
    m = lgb.LGBMClassifier(**P_).fit(XA, A.y); return m.predict_proba(XB)[:, 1], m, list(XA.columns)


res = {}
for name, t in POP.items():
    y = t.y.values; r = {}; cut = pd.Timestamp('2024-03-01'); A, B = t[t.date_Dam < cut], t[t.date_Dam >= cut]
    for tag, fs in [('base', rt), ('base+rowtime', rt + RTF)]:
        oof = np.zeros(len(t))
        for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(t, y):
            oof[b] = fit_pred(t.iloc[a], t.iloc[b], fs)[0]
        pb, m, cols = fit_pred(A, B, fs)
        r[tag] = dict(cv_pr_auc=round(average_precision_score(y, oof), 4), cv_roc_auc=round(roc_auc_score(y, oof), 4),
                      time_pr_auc=round(average_precision_score(B.y, pb), 4), time_roc_auc=round(roc_auc_score(B.y, pb), 4), _o=oof, _p=pb)
        if tag != 'base':
            imp = pd.Series(m.booster_.feature_importance('gain'), cols)
            r['rowtime_gain_share'] = round(float(imp[RTF].sum() / imp.sum()), 3)
            r['rowtime_top'] = (imp[RTF] / imp.sum()).sort_values(ascending=False).head(8).round(4).to_dict()
    r['cv_diff_ci'] = boot(y, r['base']['_o'], r['base+rowtime']['_o']); r['time_diff_ci'] = boot(B.y.values, r['base']['_p'], r['base+rowtime']['_p'])
    r['base_rate'] = round(float(y.mean()), 4); r['time_base_rate'] = round(float(B.y.mean()), 4)
    for tag in ('base', 'base+rowtime'): r[tag] = {k: v for k, v in r[tag].items() if not k.startswith('_')}
    res[name] = r; print(name, json.dumps(r, ensure_ascii=False), flush=True)

# 단변량 (9월+ 알려진 원인 제외 집단)
t = POP['sep_known_removed']; uni = {}
for f in ['rt_wait_spans_break', 'rt_long_wait_no_break', 'rt_evening', 'rt_cross_day', 'rt_max_stage']:
    uni[f] = t.groupby(f).y.agg(['size', 'mean']).round(4).to_dict('index')
for f in ['rt_wait_excess', 'rt_min_to_next_break', 'rt_tod_Fill2']:
    uni[f] = {str(k): [int(n), round(float(v) * 100, 2)] for k, (n, v) in t.groupby(pd.qcut(t[f], 5, duplicates='drop'), observed=True).y.agg(['size', 'mean']).iterrows()}
print(json.dumps(uni, ensure_ascii=False, default=str))
json.dump(dict(break_schedule_train=BRK_tr, break_schedule_test=BRK_te, row_independent=bool(same), features=RTF,
               replaced_grouped=GROUPED, results=res, univariate=uni), open('reports/20_ts_features.json', 'w'), ensure_ascii=False, indent=1, default=str)
