"""8단계 검증: 핵심 수치 재계산, 미래정보(look-ahead) 피처 제거 시 성능, 시간 순 분할 성능, 제출 파일 생성"""
import pandas as pd, numpy as np, json, warnings
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
warnings.filterwarnings('ignore'); SEED = 42
V = {}

# 1) 원본 CSV에서 핵심 수치 재계산 (전처리 파이프라인과 독립)
raw = pd.read_csv('data/train.csv', low_memory=False, usecols=['target', 'Equipment_Dam', 'Equipment_Fill1', 'Equipment_Fill2',
                                                              'Collect Date_Dam', 'Collect Date_AutoClave', 'Model.Suffix_Dam'])
yb = raw.target.eq('AbNormal')
V['defect'] = [int(yb.sum()), len(raw), round(yb.mean() * 100, 2)]
mis = raw.Equipment_Dam.str[-1].ne(raw.Equipment_Fill1.str[-1]) | raw.Equipment_Fill1.str[-1].ne(raw.Equipment_Fill2.str[-1])
lead = (pd.to_datetime(raw['Collect Date_AutoClave']) - pd.to_datetime(raw['Collect Date_Dam'])).dt.total_seconds() / 3600
rule = mis | (lead > 2)
V['rule_recheck'] = dict(mismatch=[int(mis.sum()), round(yb[mis].mean() * 100, 1)], long_wait=[int((lead > 2).sum()), round(yb[lead > 2].mean() * 100, 1)],
                         union=[int(rule.sum()), int(yb[rule].sum())])
V['model_C'] = round(yb[raw['Model.Suffix_Dam'].eq(sorted(raw['Model.Suffix_Dam'].dropna().unique())[2])].mean() * 100, 2)

# 2) OOF 상위 k% 재검사 시 포착률
O = pd.read_parquet('data/oof.parquet')
for k in (1, 3, 5, 10):
    cut = O.lightgbm.quantile(1 - k / 100); sel = O.lightgbm >= cut
    V[f'top{k}pct'] = dict(precision=round(O.target[sel].mean(), 3), recall=round(O.target[sel].sum() / O.target.sum(), 3))

# 3) 미래정보 점검: 작업지시 전체를 봐야 알 수 있는 피처 제거 → 실시간 운영 가능 성능
d = pd.read_parquet('data/features.parquet')
d['fe_long_wait'] = (d.fe_lead_total > 2).astype(int); d['fe_pxx_order'] = d.workorder.str.contains('PXX').astype(int)
for c in ['model', 'fe_wo_prefix', 'fe_wo_suffix']: d[c + '_code'] = d[c].astype('category').cat.codes
FS = json.load(open('reports/05_feature_sets.json'))
LOOKAHEAD = ['fe_wo_size', 'fe_wo_span_h', 'fe_wo_rank', 'fe_qty_ratio', 'wo_te']
tr = d[d.is_train == 1].reset_index(drop=True); y = tr.target.astype(int).values
P = dict(n_estimators=600, learning_rate=0.02, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.5, reg_lambda=5, verbose=-1, random_state=SEED)
def cv(fs):
    oof = np.zeros(len(tr))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(tr, y):
        oof[b] = lgb.LGBMClassifier(**P).fit(tr.iloc[a][fs], y[a]).predict_proba(tr.iloc[b][fs])[:, 1]
    return oof
def met(yt, p):
    ths = np.quantile(p, np.linspace(0.8, 0.995, 120)); return dict(roc_auc=round(roc_auc_score(yt, p), 4), pr_auc=round(average_precision_score(yt, p), 4),
                                                                      best_f1=round(max(f1_score(yt, p >= t) for t in ths), 4))
rt = [f for f in FS['full'] if f not in LOOKAHEAD]
V['realtime_no_lookahead_cv'] = met(y, cv(rt))
V['realtime_no_lookahead_n_features'] = len(rt)

# 4) 시간 순 분할: ~2024-02 학습 → 2024-03~04 검증 (미래 예측 성능)
cut = pd.Timestamp('2024-03-01'); A = tr[tr.date_Dam < cut]; B = tr[tr.date_Dam >= cut]
for name, fs in [('full_minus_lookahead', rt), ('process_only', FS['process_only'])]:
    fs = [f for f in fs if f not in LOOKAHEAD and f != 'fe_days']
    p = lgb.LGBMClassifier(**P).fit(A[fs], A.target.astype(int)).predict_proba(B[fs])[:, 1]
    V[f'time_split_{name}'] = met(B.target.astype(int).values, p) | dict(n_val=len(B), val_defect=round(B.target.mean(), 4))

# 5) 제출 파일: GBDT 3종 test 예측 평균 → OOF와 같은 양성 비율 기준 threshold, 규칙 해당 행은 불량 확정
te = d[d.is_train == 0].reset_index(drop=True)
tp = np.mean([np.load(f'data/test_pred_{m}.npy') for m in ['lightgbm', 'xgboost', 'catboost']], axis=0)
oofm = O[['lightgbm', 'xgboost', 'catboost']].mean(1).values
ths = np.quantile(oofm, np.linspace(0.8, 0.995, 120)); f = [f1_score(O.target, oofm >= t) for t in ths]; th = ths[int(np.argmax(f))]
pred = tp >= th
te_rule = (te.equip_mismatch == 1) | (te.fe_lead_total > 2) | (te.fe_Fill1_zero_dispense == 1)
pred = pred | te_rule.values
sub = pd.DataFrame({'Set ID': te['Set ID'], 'target': np.where(pred, 'AbNormal', 'Normal'), 'prob': tp.round(5)})
sub.to_csv('reports/submission.csv', index=False)
V['submission'] = dict(threshold=round(float(th), 4), oof_f1_at_th=round(max(f), 4), n=len(sub), n_abnormal=int(pred.sum()),
                       rate=round(pred.mean(), 4), rule_rows=int(te_rule.sum()))
json.dump(V, open('reports/08_validation.json', 'w'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(V, ensure_ascii=False, indent=1, default=str))
