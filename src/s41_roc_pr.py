"""41단계: ROC 곡선 · PR 곡선 — 원본 변수 모델 vs 최종 모델 (같은 기간 5-fold OOF, 38단계 data/ladder_oof.parquet)
결과: figs/proj_roc_pr.png, reports/41_roc_pr.json
"""
import json, numpy as np, pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for w in ('Regular', 'Bold'): fm.fontManager.addfont(f'fonts/NotoSansKR-{w}.ttf')
plt.rcParams.update({'font.family': 'Noto Sans KR Static', 'axes.unicode_minus': False, 'axes.spines.top': False, 'axes.spines.right': False})
RED, BLUE, GREY, DARK = '#c0392b', '#8fa9cc', '#aaaaaa', '#1a2b4c'
O = pd.read_parquet('data/ladder_oof.parquet'); y = O.y.values; base = y.mean()
M = {'원본 변수 모델': (O['0. 원본 그대로'].values, BLUE), '최종 모델': (O['3. + 분석 기반 변수'].values, RED)}
out = {nm: dict(roc_auc=round(roc_auc_score(y, p), 4), pr_auc=round(average_precision_score(y, p), 4)) for nm, (p, _) in M.items()}
json.dump(out, open('reports/41_roc_pr.json', 'w'), ensure_ascii=False, indent=1); print(out)

fig, ax = plt.subplots(1, 2, figsize=(13, 5.6))
a = ax[0]
for nm, (p, c) in M.items():
    f, t, _ = roc_curve(y, p); a.plot(f, t, color=c, lw=3 if c == RED else 2, label=f"{nm}  AUC {out[nm]['roc_auc']:.3f}")
    if c == RED: a.fill_between(f, t, color=RED, alpha=.08)
a.plot([0, 1], [0, 1], color=GREY, ls='--', lw=1, label='무작위  AUC 0.500')
a.set_xlabel('정상인데 불량 의심으로 고른 비율 (헛검사율)', fontsize=11); a.set_ylabel('불량 중 잡아낸 비율 (재현율)', fontsize=11)
a.set_xlim(0, 1); a.set_ylim(0, 1.02); a.legend(fontsize=10.5, frameon=False, loc='lower right')
a.set_title('ROC 곡선 — 불량과 정상을 얼마나 잘 구분하나\n(왼쪽 위로 붙을수록 좋음)', fontsize=12.5, weight='bold', loc='left')
b = ax[1]
for nm, (p, c) in M.items():
    pr, rc, _ = precision_recall_curve(y, p); b.plot(rc, pr, color=c, lw=3 if c == RED else 2, label=f"{nm}  PR-AUC {out[nm]['pr_auc']:.3f}")
    if c == RED: b.fill_between(rc, pr, color=RED, alpha=.08)
b.axhline(base, color=GREY, ls='--', lw=1, label=f'무작위  PR-AUC {base:.3f}')
b.set_xlabel('불량 중 잡아낸 비율 (재현율)', fontsize=11); b.set_ylabel('고른 것 중 진짜 불량 비율 (정밀도)', fontsize=11)
b.set_xlim(0, 1); b.set_ylim(0, 1.02); b.legend(fontsize=10.5, frameon=False, loc='upper right')
b.annotate('상위 1% 재검사\n정밀도 84%', xy=(.144, .835), xytext=(.32, .7), fontsize=10.5, color=RED, weight='bold', arrowprops=dict(arrowstyle='->', color=RED))
b.set_title('PR 곡선 — 골라낸 것이 얼마나 진짜 불량인가\n(오른쪽 위로 붙을수록 좋음 · 불량이 드문 데이터에 적합)', fontsize=12.5, weight='bold', loc='left')
fig.text(.01, -.03, '같은 기간 5-fold 예측 · 제품 40,506개(불량 2,350개). 곡선 아래 넓이(AUC)가 클수록 좋다. 같은 LightGBM · 같은 설정, 변수만 다름', fontsize=10, color='#444')
plt.tight_layout(); plt.savefig('figs/proj_roc_pr.png', dpi=150, bbox_inches='tight'); plt.close()
