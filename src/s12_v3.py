"""12단계 v3 피처: 분석에서 나온 '불량이 시간적으로 몰린다'는 관찰을 피처로 구현
- 이웃 라벨: 같은 Dam 설비에서 시간상 가까운 제품의 불량률(±30분·2시간·8시간, 과거만), 앞/뒤 최근접 라벨, 탈포 배치 동료 불량률
  → 폴드마다 학습 폴드 라벨만으로 계산(검증·test 라벨 미사용), 자기 자신은 제외
- 설정 변경: 같은 설비 직전 제품 대비 Dam 속도·토출·좌표 변경 여부, 변경 후 경과 개수
- 좌표 드리프트: Dam Y(Stage2) 편차의 직전 20개 이동평균·표준편차
같은 5-fold(seed 42)·LightGBM 기본 파라미터로 C(v1+v2) 대비 비교
"""
import pandas as pd, numpy as np, json, time, warnings
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
warnings.filterwarnings('ignore')

d = pd.read_parquet('data/features_v2.parquet')
d['fe_long_wait'] = (d.fe_lead_total > 2).astype(int); d['fe_pxx_order'] = d.v2_pxx_order
for c in ['model', 'fe_wo_prefix', 'fe_wo_suffix']: d[c + '_code'] = d[c].astype('category').cat.codes

# ── 라벨 없는 v3 피처 (전체 행 기준) ──
F = {}
def add(n, s, desc): d[n] = s; F[n] = desc
o = d.sort_values(['equip_Dam', 'date_Dam'])
g = o.groupby('equip_Dam')
for c, n in [('fe_Dam_speed_mean', 'speed'), ('Dam__DISCHARGED TIME OF RESIN(Stage2)', 's2_time'), ('dev_Dam__Y(Stage2)', 'y2')]:
    ch = (o[c] != g[c].shift(1)) & g[c].shift(1).notna()
    add(f'v3_chg_{n}', ch.astype(int).reindex(d.index), f'Dam {n} 직전 제품 대비 변경')
    blk = ch.groupby(o.equip_Dam).cumsum()
    add(f'v3_since_chg_{n}', o.groupby([o.equip_Dam, blk]).cumcount().reindex(d.index), f'Dam {n} 변경 후 경과 개수')
y2 = g['dev_Dam__Y(Stage2)']
add('v3_y2_roll20_mean', y2.transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean()).reindex(d.index), 'Dam Y2 편차 직전 20개 평균')
add('v3_y2_roll20_std', y2.transform(lambda s: s.shift(1).rolling(20, min_periods=5).std()).reindex(d.index), 'Dam Y2 편차 직전 20개 표준편차')
STATIC = list(F)

tr = d[d.is_train == 1].reset_index(drop=True); te = d[d.is_train == 0].reset_index(drop=True)
y = tr.target.astype(int).values
ALL = pd.concat([tr, te], ignore_index=True)  # 0..len(tr)-1 = train
ntr = len(tr)
ORD = ALL.sort_values(['equip_Dam', 'date_Dam']).index.values
EQ = ALL.equip_Dam.values[ORD]; T = (ALL.date_Dam.values[ORD] - np.datetime64('2023-01-01')) / np.timedelta64(1, 'm')
ACK = ALL.date_AutoClave.values

def nb_feats(lab_mask, k=10):
    """lab_mask: ALL 길이 bool — 라벨 사용 가능한 행(학습 폴드). 반환: ALL 길이 DataFrame"""
    lab = np.zeros(len(ALL)); lab[:ntr] = 1; lab = lab * lab_mask
    yy = np.zeros(len(ALL)); yy[:ntr] = y; yy = yy * lab
    prior = yy.sum() / lab.sum()
    out = {}
    L, Y = lab[ORD], yy[ORD]
    res = {f'nb_{w}': np.zeros(len(ALL)) for w in (30, 120, 480)}
    res.update({f'nbp_{w}': np.zeros(len(ALL)) for w in (120, 480)})
    prev_y = np.full(len(ALL), np.nan); next_y = np.full(len(ALL), np.nan); prev_dt = np.full(len(ALL), np.nan)
    for e in np.unique(EQ):
        s = np.where(EQ == e)[0]; t, l, yv = T[s], L[s], Y[s]
        cl, cy = np.r_[0, np.cumsum(l)], np.r_[0, np.cumsum(yv)]
        for w in (30, 120, 480):
            lo, hi = np.searchsorted(t, t - w, 'left'), np.searchsorted(t, t + w, 'right')
            n, sy = cl[hi] - cl[lo] - l, cy[hi] - cy[lo] - yv  # 자기 자신 제외
            res[f'nb_{w}'][ORD[s]] = (sy + k * prior) / (n + k)
        for w in (120, 480):  # 과거만 (실시간 가능)
            lo, hi = np.searchsorted(t, t - w, 'left'), np.searchsorted(t, t, 'left')
            n, sy = cl[hi] - cl[lo], cy[hi] - cy[lo]
            res[f'nbp_{w}'][ORD[s]] = (sy + k * prior) / (n + k)
        # 앞/뒤 최근접 라벨 행 (자기 제외)
        lv = pd.Series(np.where(l > 0, yv, np.nan)); tv = pd.Series(np.where(l > 0, t, np.nan))
        prev_y[ORD[s]] = lv.shift(1).ffill().values; next_y[ORD[s]] = lv.shift(-1).bfill().values
        prev_dt[ORD[s]] = t - tv.shift(1).ffill().values
    out.update(res); out['nb_prev_y'] = prev_y; out['nb_next_y'] = next_y; out['nb_prev_dt'] = prev_dt
    # 탈포 배치 동료 불량률 (자기 제외)
    b = pd.DataFrame({'k': ACK, 'l': lab, 'y': yy})
    gs = b.groupby('k')[['l', 'y']].transform('sum')
    out['nb_ac_batch'] = ((gs.y - b.y) + k * prior) / ((gs.l - b.l) + k)
    return pd.DataFrame(out)
NB = ['nb_30', 'nb_120', 'nb_480', 'nbp_120', 'nbp_480', 'nb_prev_y', 'nb_next_y', 'nb_prev_dt', 'nb_ac_batch']
NB_RT = ['nbp_120', 'nbp_480', 'nb_prev_y', 'nb_prev_dt']  # 과거 라벨만 → 검사 결과가 즉시 나오면 실시간 가능

def wo_te(trn, val, k=20):
    prior = trn.target.mean(); g = trn.groupby('workorder').target.agg(['sum', 'count'])
    return val.workorder.map((g['sum'] + k * prior) / (g['count'] + k)).fillna(prior).values

BASE_P = dict(n_estimators=600, learning_rate=0.02, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.5, reg_lambda=5, verbose=-1, n_jobs=8)
folds = []
for a, b in StratifiedKFold(5, shuffle=True, random_state=42).split(tr, y):
    A, B = tr.iloc[a], tr.iloc[b]; inner = np.zeros(len(A))
    for ia, ib in StratifiedKFold(5, shuffle=True, random_state=42).split(A, A.target): inner[ib] = wo_te(A.iloc[ia], A.iloc[ib])
    m = np.zeros(len(ALL), bool); m[a] = True
    folds.append((a, b, inner, wo_te(A, B), nb_feats(m)))
mfull = np.zeros(len(ALL), bool); mfull[:ntr] = True; NB_TEST = nb_feats(mfull).iloc[ntr:].reset_index(drop=True)

def best_f1(yt, p):
    ths = np.quantile(p, np.linspace(0.85, 0.995, 150)); f = np.array([f1_score(yt, p >= t) for t in ths]); i = f.argmax()
    return f[i], ths[i]

def X(df, fs, te_vals, nb):
    x = df[[f for f in fs if f != 'wo_te' and f not in NB]].copy()
    if 'wo_te' in fs: x['wo_te'] = te_vals
    for f in fs:
        if f in NB: x[f] = nb[f].values
    return x

def run(fs, test=False):
    oof = np.zeros(ntr); tp = np.zeros(len(te))
    for a, b, inner, teB, nb in folds:
        m = lgb.LGBMClassifier(**BASE_P, random_state=42).fit(X(tr.iloc[a], fs, inner, nb.iloc[a]), y[a])
        oof[b] = m.predict_proba(X(tr.iloc[b], fs, teB, nb.iloc[b]))[:, 1]
        if test: tp += m.predict_proba(X(te, fs, wo_te(tr.iloc[a], te), NB_TEST))[:, 1] / 5
    return oof, tp

FS1 = json.load(open('reports/05_feature_sets.json')); V2 = list(json.load(open('reports/10_feature_dict_v2.json')))
C = list(dict.fromkeys(FS1['full'] + V2))
SETS = {'C_v1+v2': C, 'H_C+static': C + STATIC, 'I_C+nb_realtime': C + NB_RT, 'J_C+nb_all': C + NB, 'K_C+static+nb_all': C + STATIC + NB}
res = []; OOF = {}
for k, fs in SETS.items():
    t0 = time.time(); OOF[k], tp = run(fs, test=(k == 'K_C+static+nb_all'))
    f, th = best_f1(y, OOF[k])
    r = dict(exp=k, f1=round(f, 4), pr_auc=round(average_precision_score(y, OOF[k]), 4), roc_auc=round(roc_auc_score(y, OOF[k]), 4),
             th=round(th, 4), n_feat=len(fs), sec=round(time.time() - t0))
    print(r, flush=True); res.append(r)
np.save('data/test_pred_v3.npy', tp)
R = pd.DataFrame(res); R.to_csv('reports/12_v3_results.csv', index=False)
json.dump({**F, **{n: '이웃 라벨(폴드 내 계산)' for n in NB}}, open('reports/12_feature_dict_v3.json', 'w'), ensure_ascii=False, indent=1)
print(R.to_string(index=False))
