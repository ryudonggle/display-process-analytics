"""21단계: 행 단위 파생변수 추가 (test 한 행 + 미리 아는 상수만) — 달력·교대·팔레트·카운터·상호작용
20단계와 같은 원칙: 다른 제품 행·라벨·경과일 사용 안 함. 작업지시 접두어(연·월 코드)도 시간 추세라 제외
상수: 한국 공휴일 달력(2023-05 ~ 2024-04, 사전에 알 수 있음), 20단계 휴식 시각표
평가: 그룹별 추가(ablation) · 5-fold CV + 시간 순(~2024-02 → 03~04) · 부트스트랩 CI
결과: reports/21_more_rowfeats.json, 21_run.log
"""
import pandas as pd, numpy as np, json, warnings
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings('ignore'); SEED = 42
exec(open('src/s20_ts_features.py', encoding='utf-8').read().split('# ② 평가')[0])   # 20단계 행 단위 피처(rt_*) · dev_fit · 검증 재사용

HOLI = pd.to_datetime(['2023-05-05', '2023-05-29', '2023-06-06', '2023-08-15', '2023-09-28', '2023-09-29', '2023-10-02', '2023-10-03',
                       '2023-10-09', '2023-12-25', '2024-01-01', '2024-02-09', '2024-02-12', '2024-03-01', '2024-04-10'])
cal = pd.DataFrame(index=pd.date_range('2023-04-01', '2024-05-31'))
cal['off'] = (cal.index.weekday >= 5) | cal.index.isin(HOLI)
run, since, blen, cur = [], [], [], 0
last_off, prev_len = None, 0
for dt, off in cal.off.items():                                   # 달력 상수표: 직전 휴무 이후 경과일, 직전 휴무 길이
    if off: cur += 1; since.append(0); blen.append(np.nan)
    else:
        if cur: prev_len = cur; last_off = dt - pd.Timedelta(days=1)
        cur = 0; since.append((dt - last_off).days if last_off is not None else np.nan); blen.append(prev_len)
cal['since_off'], cal['prev_off_len'] = since, blen


def more_feats(R):
    F = pd.DataFrame(index=R.index); day = R.date_Dam.dt.normalize()
    C = cal.reindex(day.values); C.index = R.index
    F['cal_offday_work'] = C.off.astype(float)                    # 휴무일(주말·공휴일) 특근
    F['cal_days_since_off'] = C.since_off                         # 1 = 휴무 다음 첫 근무일
    F['cal_first_day_after_off'] = (C.since_off == 1).astype(float)
    F['cal_prev_off_len'] = C.prev_off_len                        # 직전 휴무 길이(연휴면 3일+)
    F['cal_dom'] = R.date_Dam.dt.day.astype(float)                # 월초·월말 (월 자체는 시간 추세라 제외)
    m = R.date_Dam.dt.hour * 60 + R.date_Dam.dt.minute
    F['sh_min_since_0800'] = m - 480; F['sh_hour_sin'] = np.sin(2 * np.pi * m / 1440); F['sh_hour_cos'] = np.cos(2 * np.pi * m / 1440)
    F['pl_pallet'] = R['Dam__PalletID']
    F['pl_mismatch'] = ((R['Dam__PalletID'] != R['Fill1__PalletID']) | (R['Dam__PalletID'] != R['Fill2__PalletID'])).astype(float)
    F['ct_qty'] = R['Dam__Production Qty']
    F['ct_qty_mismatch'] = ((R['Dam__Production Qty'] != R['Fill1__Production Qty']) | (R['Dam__Production Qty'] != R['Fill2__Production Qty'])).astype(float)
    F['ct_recipe'] = R['Dam__Receip No']
    F['ct_recipe_mismatch'] = ((R['Dam__Receip No'] != R['Fill1__Receip No']) | (R['Dam__Receip No'] != R['Fill2__Receip No'])).astype(float)
    short = (R.fe_AC_time_total < 400).astype(float); w = (R.date_AutoClave - R.date_Fill2) / MIN
    F['ix_wait_x_shortdegas'] = w * short
    F['ix_excess_x_shortdegas'] = R.rt_wait_excess * short
    F['ix_wait_x_uv45'] = w * (R['Fill2__CURE SPEED'] == 45)
    F['ix_speed_x_wait'] = R['Dam__Stage2 Line1 Distance Speed'] * w / 1000
    return F


full = more_feats(d); only = more_feats(d[d.is_train == 0])
same2 = all(np.allclose(full.loc[only.index, c].values.astype(float), only[c].values.astype(float), equal_nan=True) for c in only.columns)
print('row-independence check (21):', same2); assert same2
d = d.join(full); GRP = {'calendar': [c for c in full if c.startswith('cal_')], 'shift': [c for c in full if c.startswith('sh_')],
                         'pallet': [c for c in full if c.startswith('pl_')], 'counter': [c for c in full if c.startswith('ct_')],
                         'interaction': [c for c in full if c.startswith('ix_')]}
ALL_NEW = sum(GRP.values(), [])

LOOKAHEAD = ['fe_wo_size', 'fe_wo_span_h', 'fe_wo_rank', 'fe_qty_ratio', 'wo_te']
FS = json.load(open('reports/05_feature_sets.json'))['process_only']
GROUPED = [f for f in FS if f.startswith('dev_') or f.startswith('fe_') and f.endswith('coord_dev_abs')]
base = [f for f in FS if f not in LOOKAHEAD and f != 'fe_days' and f not in GROUPED] + RTF   # 20단계까지 포함한 기준
P_ = dict(n_estimators=600, learning_rate=0.02, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
          colsample_bytree=0.5, reg_lambda=5, verbose=-1, random_state=SEED)
tr = d[d.is_train == 1].reset_index(drop=True); tr['y'] = tr.target.astype(int)
rule = (tr.fe_long_wait == 1) | (tr.equip_mismatch == 1) | (tr.fe_Fill1_zero_dispense == 1)
POP = {'all_train': tr, 'sep_known_removed': tr[(tr.date_Dam >= '2023-09-01') & ~rule & ~(tr['dev_Dam__Y(Stage2)'] < -0.05)].reset_index(drop=True)}
rng = np.random.default_rng(SEED)


def fit_pred(A, B, fs, seed=SEED):
    XA, XB = pd.concat([A[fs], dev_fit(A, A)], axis=1), pd.concat([B[fs], dev_fit(A, B)], axis=1)
    m = lgb.LGBMClassifier(**(P_ | dict(random_state=seed))).fit(XA, A.y); return m.predict_proba(XB)[:, 1], m, list(XA.columns)


def boot(yv, p1, p2, n=1000):
    ds = [average_precision_score(yv[i], p2[i]) - average_precision_score(yv[i], p1[i])
          for i in (rng.integers(0, len(yv), len(yv)) for _ in range(n)) if yv[i].sum()]
    return [round(float(np.percentile(ds, 2.5)), 4), round(float(np.percentile(ds, 97.5)), 4)]


res = {}
for name, t in POP.items():
    y = t.y.values; cut = pd.Timestamp('2024-03-01'); A, B = t[t.date_Dam < cut], t[t.date_Dam >= cut]; r = {}
    sets = {'base(20)': base} | {f'+{g}': base + c for g, c in GRP.items()} | {'+all': base + ALL_NEW}
    keep = {}
    for tag, fs in sets.items():
        oof = np.zeros(len(t))
        for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(t, y): oof[b] = fit_pred(t.iloc[a], t.iloc[b], fs)[0]
        pbs = [fit_pred(A, B, fs, s)[0] for s in (SEED, 1, 2)]          # 시간 순은 시드 3개 평균
        pb = np.mean(pbs, 0)
        r[tag] = dict(cv_pr_auc=round(average_precision_score(y, oof), 4), time_pr_auc=round(average_precision_score(B.y, pb), 4),
                      time_roc_auc=round(roc_auc_score(B.y, pb), 4))
        keep[tag] = (oof, pb)
    for tag in sets:
        if tag != 'base(20)':
            r[tag]['cv_diff_ci'] = boot(y, keep['base(20)'][0], keep[tag][0]); r[tag]['time_diff_ci'] = boot(B.y.values, keep['base(20)'][1], keep[tag][1])
    _, m, cols = fit_pred(A, B, base + ALL_NEW); imp = pd.Series(m.booster_.feature_importance('gain'), cols)
    r['new_top'] = (imp[ALL_NEW] / imp.sum()).sort_values(ascending=False).head(10).round(4).to_dict()
    r['base_rate'] = round(float(y.mean()), 4); r['time_base_rate'] = round(float(B.y.mean()), 4)
    res[name] = r; print(name, json.dumps(r, ensure_ascii=False), flush=True)

t = POP['sep_known_removed']; uni = {}
for f in ['cal_offday_work', 'cal_first_day_after_off', 'pl_mismatch', 'ct_qty_mismatch', 'ct_recipe_mismatch', 'pl_pallet', 'cal_prev_off_len']:
    uni[f] = t.groupby(f).y.agg(['size', 'mean']).round(4).to_dict('index')
for f in ['cal_days_since_off', 'cal_dom', 'ct_qty', 'sh_min_since_0800']:
    uni[f] = {str(k): [int(n), round(float(v) * 100, 2)] for k, (n, v) in t.groupby(pd.qcut(t[f], 5, duplicates='drop'), observed=True).y.agg(['size', 'mean']).iterrows()}
print(json.dumps(uni, ensure_ascii=False, default=str))
json.dump(dict(row_independent=bool(same2), groups=GRP, results=res, univariate=uni), open('reports/21_more_rowfeats.json', 'w'),
          ensure_ascii=False, indent=1, default=str)
