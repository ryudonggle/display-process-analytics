"""37단계: 보고서(A4 세로)용으로 혼동행렬 · ROI 그림을 다시 배치 — 35 · 36단계 json에서 읽어 그림만 다시 그림(재학습 없음)
결과: figs/proj_confusion_a4.png, figs/proj_roi_a4.png
"""
import json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for w in ('Regular', 'Bold'): fm.fontManager.addfont(f'fonts/NotoSansKR-{w}.ttf')
plt.rcParams.update({'font.family': 'Noto Sans KR Static', 'axes.unicode_minus': False, 'axes.spines.top': False, 'axes.spines.right': False})
RED, BLUE, GREY, DARK = '#1D6A72', '#B9C0C6', '#DDE0E3', '#222222'
C = json.load(open('reports/35_confusion.json', encoding='utf-8')); R = json.load(open('reports/36_roi.json', encoding='utf-8'))


def draw(ax, c, title):
    M = np.array([[c['TP'], c['FN']], [c['FP'], c['TN']]])
    col = np.array([['#1D6A72', '#F6DEDA'], ['#DDE0E3', '#D7E9EA']])
    nm = np.array([['TP · 잡았다', 'FN · 놓쳤다'], ['FP · 헛검사', 'TN · 통과']])
    for r in range(2):
        for k in range(2):
            w = (r, k) == (0, 0)
            ax.add_patch(plt.Rectangle((k, 1 - r), 1, 1, fc=col[r, k], ec='white', lw=3))
            ax.text(k + .5, 1 - r + .58, f'{M[r, k]:,}', ha='center', va='center', fontsize=22, weight='bold', color='white' if w else DARK)
            ax.text(k + .5, 1 - r + .25, nm[r, k], ha='center', va='center', fontsize=11.5, color='white' if w else '#333')
    ax.set_xlim(0, 2); ax.set_ylim(0, 2); ax.set_xticks([.5, 1.5]); ax.set_xticklabels(['모델: 불량 의심', '모델: 정상'], fontsize=11)
    ax.set_yticks([1.5, .5]); ax.set_yticklabels(['실제\n불량', '실제\n정상'], fontsize=11); ax.tick_params(length=0); ax.xaxis.set_ticks_position('top')
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title(f"{title}\n정밀도 {c['precision']*100:.0f}% · 재현율 {c['recall']*100:.0f}% · F1 {c['f1']:.3f}", fontsize=13, weight='bold', pad=30, loc='left')


F = C['최종 변수']; O = C['원본 변수']
fig, ax = plt.subplots(2, 2, figsize=(12, 11.5))
draw(ax[0, 0], F['f1_best'], f"① F1이 가장 높은 기준선 (점수 ≥ {F['best_threshold']:.2f})")
draw(ax[0, 1], F['top1'], '② 위험 상위 1%만 재검사')
draw(ax[1, 0], F['top3'], '③ 위험 상위 3%만 재검사')
draw(ax[1, 1], O['top1'], '④ 비교: 원본 변수 모델 · 상위 1% 재검사')
fig.text(.01, -.02, f"같은 기간 5-fold 예측 · 제품 40,506개(불량 2,350개)\n"
         f"원본 변수 모델 대비: F1 최대 {O['f1_best']['f1']:.3f} → {F['f1_best']['f1']:.3f} · 상위 1% 재검사 시 잡은 불량 {O['top1']['TP']} → {F['top1']['TP']}개, 헛검사 {O['top1']['FP']} → {F['top1']['FP']}개", fontsize=11, color='#222222')
plt.tight_layout(h_pad=3); plt.savefig('figs/proj_confusion_a4.png', dpi=150, bbox_inches='tight'); plt.close()

# ROI
T = R['table']; names = ['무작위', '원본 변수 모델', '최종 모델']; COL = {'무작위': GREY, '원본 변수 모델': BLUE, '최종 모델': RED}
KS = sorted(float(k) for k in T['최종 모델']); per = 10000 / R['n_products']; V = 10
fig = plt.figure(figsize=(12, 10)); gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1], hspace=.45, wspace=.35)
a = fig.add_subplot(gs[0, :])
for nm in names:
    net = [(V * T[nm][str(k)]['TP'] - T[nm][str(k)]['inspected']) * per for k in KS]
    a.plot(KS, net, color=COL[nm], lw=3 if nm == '최종 모델' else 2, label=nm)
    if nm == '최종 모델':
        i = int(np.argmax(net)); a.scatter([KS[i]], [net[i]], color=RED, s=60, zorder=3)
        a.annotate(f'최대 +{net[i]:.0f} (상위 {KS[i]:g}% 재검사)', xy=(KS[i], net[i]), xytext=(KS[i] + 2.2, net[i] + 40), fontsize=12, color=RED, weight='bold',
                   arrowprops=dict(arrowstyle='->', color=RED)); top = net[i]
a.axhline(0, color='#222222', lw=1); a.set_ylim(top=top * 1.5); a.set_xlabel('매달 재검사하는 비율 (위험 점수 상위 k%)', fontsize=11.5); a.set_ylabel('제품 1만 개당 순이익', fontsize=11.5)
a.legend(fontsize=11, frameon=False, loc='lower left'); a.tick_params(labelsize=10.5)
a.set_title(f'② 재검사를 얼마나 할까 — 불량 1개 잡는 가치 = 재검사 비용의 {V}배 가정', fontsize=13, weight='bold', loc='left')
A3 = R['at_3pct']; b = fig.add_subplot(gs[1, 0]); v = [A3[n]['precision'] * 100 for n in names]
b.bar(['무작위', '원본 변수\n모델', '최종\n모델'], v, color=[COL[n] for n in names], width=.6)
for i, x in enumerate(v): b.text(i, x + .5, f'{x:.1f}개', ha='center', fontsize=13, weight='bold', color=RED if i == 2 else DARK)
b.set_ylim(0, max(v) * 1.3); b.set_ylabel('재검사 100회당 잡는 불량', fontsize=11); b.tick_params(labelsize=10.5)
b.set_title('① 같은 검사 횟수로 잡는 불량\n(매달 생산량의 3% 재검사)', fontsize=12.5, weight='bold', loc='left')
c = fig.add_subplot(gs[1, 1]); be = [A3[n]['breakeven_V'] for n in names]
c.bar(['무작위', '원본 변수\n모델', '최종\n모델'], be, color=[COL[n] for n in names], width=.6)
for i, x in enumerate(be): c.text(i, x + .5, f'{x:.1f}배', ha='center', fontsize=13, weight='bold', color=RED if i == 2 else DARK)
c.set_ylim(0, max(be) * 1.3); c.set_ylabel('필요한 "불량 1개 가치 ÷ 재검사 비용"', fontsize=11); c.tick_params(labelsize=10.5)
c.set_title('③ 손익분기점 — 낮을수록 쉽게 이득\n(상위 3% 재검사 기준)', fontsize=12.5, weight='bold', loc='left')
fig.text(.01, .01, f"다음 달 예측 5개월 제품 {R['n_products']:,}개 · 불량 {R['n_defects']:,}개 기준. 실제 비용은 데이터에 없어 비율로 가정함 (재검사 1회 = 1)", fontsize=10.5, color='#222222')
plt.savefig('figs/proj_roi_a4.png', dpi=150, bbox_inches='tight'); plt.close(); print('ok')
