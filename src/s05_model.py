"""5단계 ML 모델링: 5-fold Stratified CV, OOF 예측으로 모델 비교, F1 최적 threshold, 규칙 후처리"""
import pandas as pd, numpy as np, json, time, warnings, sys
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
warnings.filterwarnings('ignore')
SEED = 42

d = pd.read_parquet('data/features.parquet')
d['fe_long_wait'] = (d.fe_lead_total > 2).astype(int)
d['fe_pxx_order'] = d.workorder.str.contains('PXX').astype(int)
d['rule'] = ((d.fe_long_wait == 1) | (d.equip_mismatch == 1) | (d.fe_Fill1_zero_dispense == 1)).astype(int)
for c in ['model', 'fe_wo_prefix', 'fe_wo_suffix']:
    d[c + '_code'] = d[c].astype('category').cat.codes

TIME_ID = {'fe_days', 'fe_wo_prefix_code', 'fe_recipe_new', 'Dam__Production Qty', 'Fill1__Production Qty', 'Fill2__Production Qty',
           'Dam__PalletID', 'Fill1__PalletID', 'Fill2__PalletID', 'Dam__Receip No', 'Fill1__Receip No', 'Fill2__Receip No', 'wo_te'}
base = [c for c in d.columns if c.startswith(('Dam__', 'Fill1__', 'Fill2__', 'AutoClave__', 'fe_', 'dev_'))
        and pd.api.types.is_numeric_dtype(d[c])]
FEATS = {'full': list(dict.fromkeys(base + ['line', 'equip_mismatch', 'model_code', 'fe_wo_prefix_code', 'fe_wo_suffix_code', 'wo_te'])),
         'process_only': [c for c in base + ['line', 'equip_mismatch', 'model_code'] if c not in TIME_ID]}

tr = d[d.is_train == 1].reset_index(drop=True); te = d[d.is_train == 0].reset_index(drop=True)
y = tr.target.astype(int).values

def wo_te(trn, val, k=20):
    """작업지시 타깃 인코딩 (폴드 내부 학습분으로만 계산, 사전확률로 평활)"""
    prior = trn.target.mean(); g = trn.groupby('workorder').target.agg(['sum', 'count'])
    m = (g['sum'] + k * prior) / (g['count'] + k)
    return val.workorder.map(m).fillna(prior).values

def models(spw):
    return {
        'logistic': lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight='balanced', max_iter=2000)),
        'random_forest': lambda: RandomForestClassifier(500, min_samples_leaf=5, class_weight='balanced_subsample', n_jobs=-1, random_state=SEED),
        'lightgbm': lambda: lgb.LGBMClassifier(n_estimators=600, learning_rate=0.02, num_leaves=31, min_child_samples=40, subsample=0.8,
                                              subsample_freq=1, colsample_bytree=0.5, reg_lambda=5, verbose=-1, random_state=SEED),
        'xgboost': lambda: xgb.XGBClassifier(n_estimators=600, learning_rate=0.02, max_depth=5, subsample=0.8, colsample_bytree=0.5,
                                            min_child_weight=5, reg_lambda=5, tree_method='hist', n_jobs=8, random_state=SEED, eval_metric='aucpr'),
        'catboost': lambda: CatBoostClassifier(iterations=800, learning_rate=0.03, depth=6, l2_leaf_reg=5, verbose=0, random_seed=SEED, thread_count=8),
    }

def best_f1(yt, p):
    ths = np.linspace(0.02, 0.9, 177); f = [f1_score(yt, p >= t) for t in ths]
    i = int(np.argmax(f)); return ths[i], f[i]

def metrics(yt, p, name):
    th, f = best_f1(yt, p); pr = p >= th
    return dict(model=name, roc_auc=roc_auc_score(yt, p), pr_auc=average_precision_score(yt, p), best_f1=f, threshold=th,
                precision=precision_score(yt, pr), recall=recall_score(yt, pr))

def run_cv(name, fs, splitter, groups=None, test_pred=False):
    oof = np.zeros(len(tr)); tp = np.zeros(len(te)); mk = models((1 - y.mean()) / y.mean())[name]
    for k, (a, b) in enumerate(splitter.split(tr, y, groups)):
        A, B = tr.iloc[a].copy(), tr.iloc[b].copy()
        A['wo_te'] = wo_te(A, A); B['wo_te'] = wo_te(A, B)
        # 학습분의 TE는 자기 자신 포함 → 내부 5-fold로 재계산해 과적합 방지
        inner = np.zeros(len(A))
        for ia, ib in StratifiedKFold(5, shuffle=True, random_state=SEED).split(A, A.target):
            inner[ib] = wo_te(A.iloc[ia], A.iloc[ib])
        A['wo_te'] = inner
        m = mk(); X = A[fs].fillna(-999) if name in ('logistic', 'random_forest') else A[fs]
        m.fit(X, A.target.astype(int))
        XB = B[fs].fillna(-999) if name in ('logistic', 'random_forest') else B[fs]
        oof[b] = m.predict_proba(XB)[:, 1]
        if test_pred:
            T = te.copy(); T['wo_te'] = wo_te(A, T)
            tp += m.predict_proba(T[fs].fillna(-999) if name in ('logistic', 'random_forest') else T[fs])[:, 1] / splitter.get_n_splits()
    return oof, tp

skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
res, OOF = [], {}
for name in ['logistic', 'random_forest', 'lightgbm', 'xgboost', 'catboost']:
    t0 = time.time()
    oof, tp = run_cv(name, FEATS['full'], skf, test_pred=True)
    OOF[name] = oof; np.save(f'data/test_pred_{name}.npy', tp)
    r = metrics(y, oof, name); r['sec'] = round(time.time() - t0); res.append(r); print(r, flush=True)

# 앙상블 (GBDT 3종 순위 평균)
from scipy.stats import rankdata
ens = np.mean([rankdata(OOF[m]) / len(y) for m in ['lightgbm', 'xgboost', 'catboost']], axis=0)
OOF['ensemble_gbdt'] = ens; res.append(metrics(y, ens, 'ensemble_gbdt'))
# 규칙 후처리: 결정적 규칙 해당 행은 불량으로 확정
for m in ['lightgbm', 'ensemble_gbdt']:
    p = OOF[m].copy(); p[tr.rule.values == 1] = 1.0; res.append(metrics(y, p, m + '+rule'))

# 피처 세트 비교 / 그룹 분할(처음 보는 작업지시) 강건성 — LightGBM
oof_po, _ = run_cv('lightgbm', FEATS['process_only'], skf); res.append(metrics(y, oof_po, 'lightgbm[process_only]')); OOF['lgb_process_only'] = oof_po
sgk = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
oof_g, _ = run_cv('lightgbm', FEATS['full'], sgk, groups=tr.workorder.values); res.append(metrics(y, oof_g, 'lightgbm[group_by_workorder]'))

R = pd.DataFrame(res).round(4); R.to_csv('reports/05_model_results.csv', index=False)
pd.DataFrame(OOF).assign(target=y, rule=tr.rule.values).to_parquet('data/oof.parquet', index=False)
json.dump(FEATS, open('reports/05_feature_sets.json', 'w'), ensure_ascii=False, indent=0)
print(R.to_string(index=False))
