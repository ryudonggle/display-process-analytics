"""6단계 DL: 표 데이터 MLP (Lightning, CPU) — 5-fold OOF로 GBDT와 동일 조건 비교 + 앙상블 기여 확인"""
import pandas as pd, numpy as np, json, warnings, logging
import torch, torch.nn as nn, lightning as L
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from scipy.stats import rankdata
warnings.filterwarnings('ignore'); logging.getLogger('lightning').setLevel(logging.ERROR)
torch.set_num_threads(8); SEED = 42

d = pd.read_parquet('data/features.parquet')
d['fe_long_wait'] = (d.fe_lead_total > 2).astype(int); d['fe_pxx_order'] = d.workorder.str.contains('PXX').astype(int)
d['rule'] = ((d.fe_long_wait == 1) | (d.equip_mismatch == 1) | (d.fe_Fill1_zero_dispense == 1)).astype(int)
for c in ['model', 'fe_wo_prefix', 'fe_wo_suffix']: d[c + '_code'] = d[c].astype('category').cat.codes
FEATS = json.load(open('reports/05_feature_sets.json'))['full']
tr = d[d.is_train == 1].reset_index(drop=True); y = tr.target.astype(int).values

def wo_te(trn, val, k=20):
    prior = trn.target.mean(); g = trn.groupby('workorder').target.agg(['sum', 'count'])
    return val.workorder.map((g['sum'] + k * prior) / (g['count'] + k)).fillna(prior).values

class MLP(L.LightningModule):
    def __init__(self, n_in, n_cat, pos_weight, hidden=256, drop=0.3, lr=1e-3):
        super().__init__(); self.save_hyperparameters()
        self.emb = nn.Embedding(n_cat, 8)  # 모델(Model.Suffix) 임베딩
        self.net = nn.Sequential(nn.Linear(n_in + 8, hidden), nn.BatchNorm1d(hidden), nn.SiLU(), nn.Dropout(drop),
                                 nn.Linear(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.SiLU(), nn.Dropout(drop),
                                 nn.Linear(hidden // 2, 1))
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    def forward(self, x, c): return self.net(torch.cat([x, self.emb(c)], 1)).squeeze(1)
    def training_step(self, b, _):
        x, c, t = b; l = self.loss(self(x, c), t); self.log('train/loss', l, prog_bar=False); return l
    def validation_step(self, b, _):
        x, c, t = b; self.log('val/loss', self.loss(self(x, c), t), prog_bar=False)
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        return {'optimizer': opt, 'lr_scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)}

num = [f for f in FEATS if f != 'model_code']
oof = np.zeros(len(tr))
for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(tr, y)):
    L.seed_everything(SEED + k, workers=True)
    A, B = tr.iloc[a].copy(), tr.iloc[b].copy()
    inner = np.zeros(len(A))
    for ia, ib in StratifiedKFold(5, shuffle=True, random_state=SEED).split(A, A.target): inner[ib] = wo_te(A.iloc[ia], A.iloc[ib])
    B['wo_te'] = wo_te(A, B); A['wo_te'] = inner
    qt = QuantileTransformer(n_quantiles=200, output_distribution='normal', random_state=SEED)  # 학습 폴드로만 fit
    XA = qt.fit_transform(A[num].fillna(A[num].median())); XB = qt.transform(B[num].fillna(A[num].median()))
    # 내부 검증(조기종료용) 10% 분리
    rng = np.random.RandomState(SEED); iv = rng.rand(len(A)) < 0.1
    mk = lambda X, C, T: TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(C, dtype=torch.long), torch.tensor(T, dtype=torch.float32))
    yA = A.target.values.astype(np.float32); cA = A.model_code.values
    dl_tr = DataLoader(mk(XA[~iv], cA[~iv], yA[~iv]), batch_size=512, shuffle=True)
    dl_va = DataLoader(mk(XA[iv], cA[iv], yA[iv]), batch_size=2048)
    model = MLP(len(num), int(d.model_code.max()) + 1, pos_weight=float((1 - yA.mean()) / yA.mean()) ** 0.5)
    tr_ = L.Trainer(max_epochs=30, accelerator='cpu', logger=False, enable_checkpointing=False, enable_progress_bar=False,
                    enable_model_summary=False, deterministic=True,
                    callbacks=[L.pytorch.callbacks.EarlyStopping('val/loss', patience=5)])
    tr_.fit(model, dl_tr, dl_va)
    model.eval()
    with torch.no_grad():
        oof[b] = torch.sigmoid(model(torch.tensor(XB, dtype=torch.float32), torch.tensor(B.model_code.values, dtype=torch.long))).numpy()
    print(f'fold {k} epochs={tr_.current_epoch} auc={roc_auc_score(y[b], oof[b]):.4f}', flush=True)

def best_f1(yt, p):
    ths = np.quantile(p, np.linspace(0.80, 0.995, 120)); f = [f1_score(yt, p >= t) for t in ths]; i = int(np.argmax(f)); return f[i]
O = pd.read_parquet('data/oof.parquet'); O['mlp'] = oof; O.to_parquet('data/oof.parquet', index=False)
r = lambda p: rankdata(p) / len(p)
rows = []
for name, p in [('mlp', oof), ('lightgbm', O.lightgbm), ('ensemble_gbdt', O.ensemble_gbdt),
                ('ensemble_gbdt+mlp', (3 * O.ensemble_gbdt + r(oof)) / 4)]:
    rows.append(dict(model=name, roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p), best_f1=best_f1(y, p)))
R = pd.DataFrame(rows).round(4); R.to_csv('reports/06_dl_results.csv', index=False); print(R.to_string(index=False))
print('corr(mlp, lgb rank)', np.corrcoef(r(oof), r(O.lightgbm))[0, 1].round(3))
