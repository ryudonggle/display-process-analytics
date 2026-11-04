"""35단계: 혼동행렬(TP · FP · FN · TN)과 F1 시각화 — 원본 변수 모델 vs 최종 모델(32단계 0 · 3단계)
① 같은 기간 5-fold OOF: 기준선별 정밀도 · 재현율 · F1 곡선, F1 최대 기준선 · 상위 1% · 상위 3%의 혼동행렬
② 다음 달 예측(2023-12 ~ 2024-04 롤링): 매달 상위 3%를 불량 의심으로 볼 때 5개월 합산 혼동행렬
결과: reports/35_confusion.json, figs/proj_confusion.png, figs/proj_f1curve.png
"""
import json, numpy as np, pandas as pd, warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
warnings.filterwarnings('ignore')
exec(open('src/s32_feature_ladder.py', encoding='utf-8').read().split('MONTHS = ')[0])
for w in ('Regular', 'Bold'): fm.fontManager.addfont(f'fonts/NotoSansKR-{w}.ttf')
plt.rcParams.update({'font.family': 'Noto Sans KR Static', 'axes.unicode_minus': False, 'axes.spines.top': False, 'axes.spines.right': False})
RED, BLUE, GREY, DARK = '#c0392b', '#8fa9cc', '#999999', '#1a2b4c'

SETS = {'원본 변수': (COMMON + RAW, False), '최종 변수': (COMMON + MEAS + PHYS + ANAL, True)}
y = tr.y.values; MONTHS = ['2023-12', '2024-01', '2024-02', '2024-03', '2024-04']
oof, roll = {}, {}
for nm, (fs, dev) in SETS.items():
    o = np.zeros(len(tr))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(tr, y): o[b] = fit_pred2(tr.iloc[a], tr.iloc[b], fs, dev, SEED)
    oof[nm] = o
    yy, pp = [], []
    for mo in MONTHS:
        A, B = tr[tr.mo < mo], tr[tr.mo == mo]; p = fit_pred2(A, B, fs, dev, SEED)
        flag = np.zeros(len(B), bool); flag[np.argsort(-p)[:max(1, int(len(B) * .03))]] = True
        yy.append(B.y.values); pp.append(flag)
    roll[nm] = (np.concatenate(yy), np.concatenate(pp))


def cm(yt, flag):
    tp = int((flag & (yt == 1)).sum()); fp = int((flag & (yt == 0)).sum()); fn = int((~flag & (yt == 1)).sum()); tn = int((~flag & (yt == 0)).sum())
    pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1); f1 = 2 * pr * rc / max(pr + rc, 1e-9)
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, precision=round(pr, 3), recall=round(rc, 3), f1=round(f1, 3))


out = {}
for nm in SETS:
    pr, rc, th = precision_recall_curve(y, oof[nm]); f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-9); i = int(np.nanargmax(f1[:-1]))
    best = th[i]; o = oof[nm]
    out[nm] = dict(best_threshold=round(float(best), 3), f1_best=cm(y, o >= best),
                   top1=cm(y, o >= np.quantile(o, .99)), top3=cm(y, o >= np.quantile(o, .97)), next_month_top3=cm(*roll[nm]))
    out[nm]['_curve'] = (th, pr[:-1], rc[:-1], f1[:-1])
    print(nm, {k: v for k, v in out[nm].items() if k != '_curve'})

# ── 그림 1: 혼동행렬 4칸 (최종 모델) + 원본 비교 줄
def draw(ax, c, title):
    M = np.array([[c['TP'], c['FN']], [c['FP'], c['TN']]])
    col = np.array([['#c0392b', '#f2c9c3'], ['#f6e3b4', '#dfe7f1']])
    nm_ = np.array([['TP\n불량을 불량으로\n(잡았다)', 'FN\n불량을 정상으로\n(놓쳤다)'], ['FP\n정상을 불량으로\n(헛검사)', 'TN\n정상을 정상으로\n(통과)']])
    for r in range(2):
        for k in range(2):
            ax.add_patch(plt.Rectangle((k, 1 - r), 1, 1, fc=col[r, k], ec='white', lw=3))
            ax.text(k + .5, 1 - r + .66, f'{M[r, k]:,}', ha='center', va='center', fontsize=17, weight='bold', color='white' if (r, k) == (0, 0) else DARK)
            ax.text(k + .5, 1 - r + .3, nm_[r, k], ha='center', va='center', fontsize=8.5, color='white' if (r, k) == (0, 0) else '#333')
    ax.set_xlim(0, 2); ax.set_ylim(0, 2); ax.set_xticks([.5, 1.5]); ax.set_xticklabels(['모델: 불량 의심', '모델: 정상'], fontsize=9.5)
    ax.set_yticks([1.5, .5]); ax.set_yticklabels(['실제 불량', '실제 정상'], fontsize=9.5); ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    ax.xaxis.set_ticks_position('top')
    ax.set_title(title + f"\n정밀도 {c['precision']*100:.0f}% · 재현율 {c['recall']*100:.0f}% · F1 {c['f1']:.3f}", fontsize=10.5, weight='bold', pad=26)

F, R0 = out['최종 변수'], out['원본 변수']
fig, ax = plt.subplots(1, 4, figsize=(19, 5.2))
draw(ax[0], F['f1_best'], f"① F1이 가장 높은 기준선 (점수 ≥ {F['best_threshold']:.2f})")
draw(ax[1], F['top1'], '② 위험 상위 1%만 재검사')
draw(ax[2], F['top3'], '③ 위험 상위 3%만 재검사')
draw(ax[3], F['next_month_top3'], '④ 다음 달 예측 · 매달 상위 3% (5개월 합)')
fig.text(.01, -.04, f"같은 기간 기준 제품 40,506개(불량 2,350개). ④는 2023.12 ~ 2024.04 제품 16,029개. "
         f"원본 변수 모델과 비교 — F1 최대: {R0['f1_best']['f1']:.3f} → {F['f1_best']['f1']:.3f}, 상위 1% 정밀도: {R0['top1']['precision']*100:.0f}% → {F['top1']['precision']*100:.0f}%, "
         f"다음 달 상위 3% 정밀도: {R0['next_month_top3']['precision']*100:.0f}% → {F['next_month_top3']['precision']*100:.0f}%", fontsize=9.5, color='#444')
plt.tight_layout(); plt.savefig('figs/proj_confusion.png', dpi=150, bbox_inches='tight'); plt.close()

# ── 그림 2: 기준선별 정밀도 · 재현율 · F1
fig, ax = plt.subplots(1, 2, figsize=(14, 4.4))
for a, nm in zip(ax, ['원본 변수', '최종 변수']):
    th, p, r, f = out[nm]['_curve']; m = th <= .8
    a.plot(th[m], p[m], color='#7E868D', lw=2, label='정밀도 (고른 것 중 진짜 불량)'); a.plot(th[m], r[m], color='#C0392B', lw=2, label='재현율 (불량 중 잡은 비율)')
    a.plot(th[m], f[m], color='#1D6A72', lw=3, label='F1 (둘의 조화 평균)')
    bt = out[nm]['best_threshold']; bf = out[nm]['f1_best']['f1']
    a.axvline(bt, color=GREY, ls='--', lw=1); a.scatter([bt], [bf], color=RED, zorder=3, s=40); a.text(bt + .01, bf + .04, f'F1 최대 {bf:.3f}\n(기준선 {bt:.2f})', fontsize=9.5, color=RED, weight='bold')
    a.set_xlim(0, .8); a.set_ylim(0, 1.02); a.set_xlabel('기준선: 위험 점수가 이 값 이상이면 불량 의심'); a.set_title(f'{nm} 모델', fontsize=11.5, weight='bold', loc='left')
    if nm == '원본 변수': a.legend(fontsize=8.5, frameon=False, loc='upper right')
fig.suptitle('기준선을 올리면 헛검사는 줄고(정밀도↑) 놓치는 불량은 늘어난다(재현율↓) — F1은 그 균형점', fontsize=12, weight='bold', x=.01, ha='left', y=1.03)
plt.tight_layout(); plt.savefig('figs/proj_f1curve.png', dpi=150, bbox_inches='tight'); plt.close()
for nm in out: out[nm].pop('_curve')
json.dump(out, open('reports/35_confusion.json', 'w'), ensure_ascii=False, indent=1)
