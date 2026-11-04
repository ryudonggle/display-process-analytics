"""42단계: 재학습 없이 저장된 OOF · json으로 혼동행렬 · F1 곡선을 최종 색(포스터 기준)으로 다시 그림
결과: figs/proj_ladder_confusion.png, figs/proj_f1curve.png
"""
import json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from sklearn.metrics import precision_recall_curve

for w in ('Regular', 'Bold'): fm.fontManager.addfont(f'fonts/NotoSansKR-{w}.ttf')
plt.rcParams.update({'font.family': 'Noto Sans KR Static', 'axes.unicode_minus': False})
INK, ACC, RED, GRAY = '#222222', '#1D6A72', '#C0392B', '#7E868D'

# ── 단계별 혼동행렬: s38의 그림 코드를 그대로 실행
out = json.load(open('reports/38_ladder_confusion.json', encoding='utf-8')); DARK = INK
src = open('src/s38_ladder_confusion.py', encoding='utf-8').read()
exec(src[src.index('LAB = ['):].split("print(")[0])

# ── F1 곡선
O = pd.read_parquet('data/ladder_oof.parquet'); y = O.y.values
C = json.load(open('reports/35_confusion.json', encoding='utf-8'))
fig, ax = plt.subplots(1, 2, figsize=(14, 4.4))
for a, nm, col in zip(ax, ['원본 변수', '최종 변수'], ['0. 원본 그대로', '3. + 분석 기반 변수']):
    pr, rc, th = precision_recall_curve(y, O[col].values); pr, rc = pr[:-1], rc[:-1]
    f = 2 * pr * rc / np.maximum(pr + rc, 1e-9); m = th <= .8
    a.plot(th[m], pr[m], color=GRAY, lw=2, label='정밀도 (고른 것 중 진짜 불량)')
    a.plot(th[m], rc[m], color=RED, lw=2, label='재현율 (불량 중 잡은 비율)')
    a.plot(th[m], f[m], color=ACC, lw=3, label='F1 (둘의 조화 평균)')
    bt, bf = C[nm]['best_threshold'], C[nm]['f1_best']['f1']
    a.axvline(bt, color=INK, ls='--', lw=1); a.scatter([bt], [bf], color=ACC, zorder=3, s=40)
    a.text(bt + .01, bf + .04, f'F1 최대 {bf:.3f}\n(기준선 {bt:.2f})', fontsize=9.5, color=ACC, weight='bold')
    a.set_xlim(0, .8); a.set_ylim(0, 1.02); a.set_xlabel('기준선: 위험 점수가 이 값 이상이면 불량 의심')
    a.set_title(f'{nm} 모델', fontsize=11.5, weight='bold', loc='left')
    for s in ('top', 'right'): a.spines[s].set_visible(False)
    if nm == '원본 변수': a.legend(fontsize=8.5, frameon=False, loc='upper right')
fig.suptitle('기준선을 올리면 헛검사는 줄고(정밀도↑) 놓치는 불량은 늘어난다(재현율↓) — F1은 그 균형점', fontsize=12, weight='bold', x=.01, ha='left', color=INK)
plt.tight_layout(); plt.savefig('figs/proj_f1curve.png', dpi=150, bbox_inches='tight'); plt.close(); print('ok')
