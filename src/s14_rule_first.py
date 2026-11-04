"""14단계: 규칙 먼저 걸러내고 나머지만 모델링 (2단 구조) vs 현재 단일 모델
- 규칙: 2시간 초과 대기 · 라인 변경 · Fill1 미도포 → 불량 확정(점수 1.0)
- 나머지: 규칙 해당 행을 뺀 학습 폴드로 LightGBM 학습, 규칙 관련 변수 제외
같은 5-fold(seed 42) · C(v1+v2) 피처 · 기본 파라미터. 전체 성능과 '나머지 행' 성능을 함께 비교
"""
import pandas as pd, numpy as np, json, warnings
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
warnings.filterwarnings('ignore')

d = pd.read_parquet('data/features_v2.parquet')
d['fe_long_wait'] = (d.fe_lead_total > 2).astype(int); d['fe_pxx_order'] = d.v2_pxx_order
for c in ['model', 'fe_wo_prefix', 'fe_wo_suffix']: d[c + '_code'] = d[c].astype('category').cat.codes
d['rule'] = ((d.fe_long_wait == 1) | (d.equip_mismatch == 1) | (d.fe_Fill1_zero_dispense == 1)).astype(int)
tr = d[d.is_train == 1].reset_index(drop=True); te = d[d.is_train == 0].reset_index(drop=True)
y = tr.target.astype(int).values; rule = tr.rule.values.astype(bool)

FS1 = json.load(open('reports/05_feature_sets.json')); V2 = list(json.load(open('reports/10_feature_dict_v2.json')))
C = list(dict.fromkeys(FS1['full'] + V2))
RULE_VARS = {'fe_long_wait', 'equip_mismatch', 'fe_Fill1_zero_dispense', 'v2_wait_F2AC_ge40', 'v2_wait_F2AC_ge60'}
C_CLEAN = [f for f in C if f not in RULE_VARS]
P = dict(n_estimators=600, learning_rate=0.02, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.5, reg_lambda=5, verbose=-1, n_jobs=8, random_state=42)

def wo_te(trn, val, k=20):
    prior = trn.target.mean(); g = trn.groupby('workorder').target.agg(['sum', 'count'])
    return val.workorder.map((g['sum'] + k * prior) / (g['count'] + k)).fillna(prior).values

def X(df, fs, tev):
    x = df[[f for f in fs if f != 'wo_te']].copy()
    if 'wo_te' in fs: x['wo_te'] = tev
    return x

def fit_pred(A, B, fs):
    inner = np.zeros(len(A))
    for ia, ib in StratifiedKFold(5, shuffle=True, random_state=42).split(A, A.target): inner[ib] = wo_te(A.iloc[ia], A.iloc[ib])
    m = lgb.LGBMClassifier(**P).fit(X(A, fs, inner), A.target.astype(int))
    return m.predict_proba(X(B, fs, wo_te(A, B)))[:, 1]

oof_single = np.zeros(len(tr)); oof_two = np.zeros(len(tr))
for a, b in StratifiedKFold(5, shuffle=True, random_state=42).split(tr, y):
    A, B = tr.iloc[a], tr.iloc[b]
    oof_single[b] = fit_pred(A, B, C)
    oof_two[b] = fit_pred(A[A.rule == 0], B, C_CLEAN)
oof_two[rule] = 1.0
oof_single_r = np.where(rule, 1.0, oof_single)  # 단일 모델 + 규칙 덮어쓰기

def best_f1(yt, p):
    ths = np.quantile(p, np.linspace(0.85, 0.995, 150)); f = np.array([f1_score(yt, p >= t) for t in ths]); i = f.argmax()
    return f[i], ths[i]

def met(name, yt, p):
    f, th = best_f1(yt, p)
    return dict(exp=name, f1=round(f, 4), pr_auc=round(average_precision_score(yt, p), 4), roc_auc=round(roc_auc_score(yt, p), 4))

res = []
for nm, p in [('single', oof_single), ('single+rule', oof_single_r), ('two_stage', oof_two)]:
    res.append({**met(nm, y, p), 'eval': '전체 40,506'})
for nm, p in [('single', oof_single), ('two_stage', oof_two)]:
    res.append({**met(nm, y[~rule], p[~rule]), 'eval': f'나머지 {(~rule).sum():,}'})
R = pd.DataFrame(res); R.to_csv('reports/14_rule_first.csv', index=False)
print(R.to_string(index=False))
