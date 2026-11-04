"""11단계 F1 개선 실험: 같은 5-fold(seed 42)로 피처 세트·선택·튜닝을 순차 비교 (LightGBM)
A raw → B v1 → C v1+v2 → D 실시간 가능 → E 폴드 내부 피처선택 → F Optuna 튜닝 → G 재검증(seed 7) + 시간 순 분할
"""
import pandas as pd, numpy as np, json, time, warnings, sys
import lightgbm as lgb, optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
warnings.filterwarnings('ignore'); optuna.logging.set_verbosity(optuna.logging.WARNING)

d = pd.read_parquet('data/features_v2.parquet')
d['fe_long_wait'] = (d.fe_lead_total > 2).astype(int); d['fe_pxx_order'] = d.v2_pxx_order
for c in ['model', 'fe_wo_prefix', 'fe_wo_suffix']: d[c + '_code'] = d[c].astype('category').cat.codes
tr = d[d.is_train == 1].reset_index(drop=True); te = d[d.is_train == 0].reset_index(drop=True)
y = tr.target.astype(int).values
FS1 = json.load(open('reports/05_feature_sets.json'))
V2 = list(json.load(open('reports/10_feature_dict_v2.json')))
RAW = [c for c in tr.columns if c.startswith(('Dam__', 'Fill1__', 'Fill2__', 'AutoClave__'))] + ['model_code', 'line']
LOOKAHEAD = ['fe_wo_size', 'fe_wo_span_h', 'fe_wo_rank', 'fe_qty_ratio', 'v2_Dam_gap_next', 'v2_Fill1_gap_next', 'v2_Fill2_gap_next']
SETS = {'A_raw': RAW, 'B_v1': FS1['full'], 'C_v1+v2': list(dict.fromkeys(FS1['full'] + V2))}
SETS['D_realtime'] = [f for f in SETS['C_v1+v2'] if f not in LOOKAHEAD]
BASE_P = dict(n_estimators=600, learning_rate=0.02, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.5, reg_lambda=5, verbose=-1, n_jobs=8)

def wo_te(trn, val, k=20):
    prior = trn.target.mean(); g = trn.groupby('workorder').target.agg(['sum', 'count'])
    return val.workorder.map((g['sum'] + k * prior) / (g['count'] + k)).fillna(prior).values

def make_folds(seed):
    """폴드별 작업지시 타깃 인코딩을 한 번만 계산해 재사용"""
    out = []
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(tr, y):
        A, B = tr.iloc[a], tr.iloc[b]; inner = np.zeros(len(A))
        for ia, ib in StratifiedKFold(5, shuffle=True, random_state=seed).split(A, A.target): inner[ib] = wo_te(A.iloc[ia], A.iloc[ib])
        out.append((a, b, inner, wo_te(A, B), wo_te(A, te)))
    return out

def Xof(idx_or_df, fs, te_vals):
    X = (tr.iloc[idx_or_df] if not isinstance(idx_or_df, pd.DataFrame) else idx_or_df)[[f for f in fs if f != 'wo_te']].copy()
    if 'wo_te' in fs: X['wo_te'] = te_vals
    return X

def best_f1(yt, p):
    ths = np.quantile(p, np.linspace(0.85, 0.995, 150)); f = np.array([f1_score(yt, p >= t) for t in ths]); i = f.argmax()
    return f[i], ths[i]

def run(fs, folds, params=BASE_P, topk=None, seed=42, test=False):
    oof = np.zeros(len(tr)); tp = np.zeros(len(te))
    for a, b, inner, teB, teT in folds:
        XA, XB = Xof(a, fs, inner), Xof(b, fs, teB)
        m = lgb.LGBMClassifier(**params, random_state=seed).fit(XA, y[a])
        if topk:  # 폴드 학습분 중요도로만 선택 → 누수 없음
            keep = pd.Series(m.booster_.feature_importance('gain'), XA.columns).nlargest(topk).index.tolist()
            XA, XB = XA[keep], XB[keep]; m = lgb.LGBMClassifier(**params, random_state=seed).fit(XA, y[a])
        oof[b] = m.predict_proba(XB)[:, 1]
        if test: tp += m.predict_proba(Xof(te, fs, teT)[XA.columns])[:, 1] / len(folds)
    return oof, tp

def met(name, p, extra=None):
    f, th = best_f1(y, p)
    r = dict(exp=name, f1=round(f, 4), pr_auc=round(average_precision_score(y, p), 4), roc_auc=round(roc_auc_score(y, p), 4), th=round(th, 4))
    if extra: r.update(extra)
    print(r, flush=True); return r

folds = make_folds(42); res = []; OOF = {}
for k, fs in SETS.items():
    t0 = time.time(); OOF[k], _ = run(fs, folds); res.append(met(k, OOF[k], dict(n_feat=len(fs), sec=round(time.time() - t0))))

best_set = 'C_v1+v2' if res[2]['f1'] >= res[3]['f1'] else 'D_realtime'
for k in (60, 100):
    OOF[f'E_top{k}'], _ = run(SETS[best_set], folds, topk=k); res.append(met(f'E_{best_set}_top{k}', OOF[f'E_top{k}'], dict(n_feat=k)))

# F) Optuna: 3-fold 서브 CV로 탐색(시간 절약) → 5-fold로 평가
sel = max([r for r in res if r['exp'].startswith(('C_', 'D_', 'E_'))], key=lambda r: r['f1'])
topk = sel.get('n_feat') if sel['exp'].startswith('E_') else None
fs_best = SETS[best_set]
folds3 = make_folds(123)[:3]
def objective(trial):
    p = dict(n_estimators=trial.suggest_int('n_estimators', 300, 1500, step=100), learning_rate=trial.suggest_float('learning_rate', 0.008, 0.05, log=True),
             num_leaves=trial.suggest_int('num_leaves', 8, 96, log=True), min_child_samples=trial.suggest_int('min_child_samples', 10, 200, log=True),
             subsample=trial.suggest_float('subsample', 0.5, 1.0), subsample_freq=1, colsample_bytree=trial.suggest_float('colsample_bytree', 0.2, 0.9),
             reg_lambda=trial.suggest_float('reg_lambda', 1e-2, 50, log=True), reg_alpha=trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
             scale_pos_weight=trial.suggest_float('scale_pos_weight', 1, 6), verbose=-1, n_jobs=8)
    o, _ = run(fs_best, folds3, p, topk)
    idx = np.concatenate([b for _, b, *_ in folds3]); return average_precision_score(y[idx], o[idx])  # PR-AUC로 탐색(F1보다 안정)
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=0))
study.optimize(objective, n_trials=int(sys.argv[1]) if len(sys.argv) > 1 else 30)
TP = {**study.best_params, 'subsample_freq': 1, 'verbose': -1, 'n_jobs': 8}
OOF['F_tuned'], tp_f = run(fs_best, folds, TP, topk, test=True); res.append(met('F_tuned', OOF['F_tuned'], dict(n_feat=topk or len(fs_best))))

# G) 다른 seed 폴드로 재검증 (튜닝 과적합 점검) + 시드 앙상블
folds7 = make_folds(7)
o7, tp7 = run(fs_best, folds7, TP, topk, seed=7, test=True); res.append(met('G_tuned_seed7_folds', o7))
o_b7, _ = run(SETS['B_v1'], folds7, BASE_P, seed=7); res.append(met('G_B_v1_seed7_folds', o_b7))

# 시간 순 분할 (~2024-02 학습 → 2024-03~04): v1 vs 최종(실시간 피처, wo_te 제외)
cut = tr.date_Dam < pd.Timestamp('2024-03-01')
def ts(fs, p):
    fs = [f for f in fs if f not in ('wo_te', 'fe_days') and f not in LOOKAHEAD]
    pr = lgb.LGBMClassifier(**p, random_state=42).fit(tr.loc[cut, fs], y[cut]).predict_proba(tr.loc[~cut, fs])[:, 1]
    f, _ = best_f1(y[~cut], pr) if False else (max(f1_score(y[~cut], pr >= t) for t in np.quantile(pr, np.linspace(.85, .995, 150))), 0)
    return dict(f1=round(f, 4), pr_auc=round(average_precision_score(y[~cut], pr), 4))
TS = {'B_v1': ts(SETS['B_v1'], BASE_P), 'final': ts(fs_best, TP)}
print('time split', TS)

R = pd.DataFrame(res); R.to_csv('reports/11_experiments.csv', index=False)
json.dump(dict(best_set=best_set, topk=topk, best_params=study.best_params, best_cv_prauc_3fold=study.best_value, time_split=TS),
          open('reports/11_best.json', 'w'), ensure_ascii=False, indent=1, default=str)
pd.DataFrame(OOF).assign(target=y).to_parquet('data/oof_v2.parquet', index=False)
np.save('data/test_pred_final.npy', (tp_f + tp7) / 2)
print(R.to_string(index=False))
