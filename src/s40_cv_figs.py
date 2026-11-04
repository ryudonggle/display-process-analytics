"""40단계: 같은 기간 예측(대회 방식, 5-fold OOF)만으로 성능 사다리 · ROI 그림 — README · 폼보드용
입력: reports/32_feature_ladder.json, data/ladder_oof.parquet (38단계)
ROI 가정은 36단계와 같음: 재검사 1회 = 1, 불량 1개 잡는 가치 = V배 (V = 10 기본)
결과: figs/proj_ladder_cv.png, figs/proj_roi_cv.png, reports/40_cv_roi.json
"""
import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for w in ('Regular', 'Bold'): fm.fontManager.addfont(f'fonts/NotoSansKR-{w}.ttf')
plt.rcParams.update({'font.family': 'Noto Sans KR Static', 'axes.unicode_minus': False, 'axes.spines.top': False, 'axes.spines.right': False})
RED, BLUE, GREY, DARK = '#1D6A72', '#B9C0C6', '#DDE0E3', '#222222'   # 포스터 기준: 예측·최종=청록, 비교=회색
L = json.load(open('reports/32_feature_ladder.json', encoding='utf-8')); S = L['stages'][:4]; base = L['base_rate']
O = pd.read_parquet('data/ladder_oof.parquet'); y = O.y.values; N = len(y)
lab = ['원본 그대로', '데이터 복원', '+ 공정 물리 변수', '+ 공정 사이\n대기 시간 등']

# ── 성능 사다리 (같은 기간)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw=dict(width_ratios=[1.4, .9]))
v = [s['cv_pr_auc'] for s in S]
ax[0].bar(lab, v, color=[BLUE] * 3 + [RED], width=.62)
for i, x in enumerate(v): ax[0].text(i, x + .006, f'{x:.3f}', ha='center', fontsize=12, weight='bold', color=RED if i == 3 else DARK)
ax[0].axhline(base, color='#222222', ls='--', lw=1); ax[0].text(-.4, base + .005, '무작위', fontsize=9, color='#222222')
ax[0].set_ylim(0, max(v) * 1.2); ax[0].set_ylabel('PR-AUC'); ax[0].tick_params(axis='x', labelsize=9.5)
ax[0].set_title('예측 성능 (PR-AUC)', fontsize=12, weight='bold', loc='left')
v0, v3 = S[0]['cv_top1_precision'] * 100, S[3]['cv_top1_precision'] * 100
ax[1].bar(['원본 그대로', '최종'], [v0, v3], color=[BLUE, RED], width=.55)
for i, x in enumerate([v0, v3]): ax[1].text(i, x + 2, f'{x:.0f}%', ha='center', fontsize=15, weight='bold', color=RED if i else DARK)
ax[1].set_ylim(0, 100); ax[1].set_ylabel('%'); ax[1].set_title('위험 상위 1% 재검사 시\n진짜 불량 비율', fontsize=12, weight='bold', loc='left')
fig.suptitle(f"원본 {S[0]['cv_pr_auc']:.3f} → 최종 {S[3]['cv_pr_auc']:.3f} (1.7배) · 상위 1% 재검사 정밀도 {v0:.0f}% → {v3:.0f}%",
             fontsize=13.5, weight='bold', color=DARK, x=.01, ha='left', y=1.03)
plt.tight_layout(); plt.savefig('figs/proj_ladder_cv.png', dpi=150, bbox_inches='tight'); plt.close()

# ── ROI (같은 기간)
KS = np.r_[0.5, np.arange(1, 15.5, 0.5)]; V = 10; per = 10000 / N
P = {'원본 변수 모델': O['0. 원본 그대로'].values, '최종 모델': O['3. + 분석 기반 변수'].values}
tab = {'무작위': {k: (round(N * k / 100) * base, round(N * k / 100)) for k in KS}}
for nm, p in P.items():
    o = np.argsort(-p); tab[nm] = {k: (int(y[o[:round(N * k / 100)]].sum()), round(N * k / 100)) for k in KS}
names = ['무작위', '원본 변수 모델', '최종 모델']; COL = {'무작위': GREY, '원본 변수 모델': BLUE, '최종 모델': RED}
prec = {n: {k: tab[n][k][0] / tab[n][k][1] for k in KS} for n in names}
out = {n: dict(top1_precision=round(prec[n][1.0], 3), top3_precision=round(prec[n][3.0], 3), breakeven_top3=round(1 / prec[n][3.0], 1),
              roi_top3={str(v_): round(v_ * prec[n][3.0] - 1, 3) for v_ in (5, 10, 20)}) for n in names}
json.dump(out, open('reports/40_cv_roi.json', 'w'), ensure_ascii=False, indent=1); print(json.dumps(out, ensure_ascii=False))
fig = plt.figure(figsize=(12, 10)); gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1], hspace=.45, wspace=.35)
a = fig.add_subplot(gs[0, :])
for nm in names:
    net = [(V * tab[nm][k][0] - tab[nm][k][1]) * per for k in KS]
    a.plot(KS, net, color=COL[nm], lw=3 if nm == '최종 모델' else 2, label=nm)
    if nm == '최종 모델':
        i = int(np.argmax(net)); a.scatter([KS[i]], [net[i]], color=RED, s=60, zorder=3); top = net[i]
        a.annotate(f'최대 +{net[i]:,.0f} (상위 {KS[i]:g}% 재검사)', xy=(KS[i], net[i]), xytext=(KS[i] + 2.2, net[i] * 1.12), fontsize=12, color=RED, weight='bold', arrowprops=dict(arrowstyle='->', color=RED))
a.axhline(0, color='#222222', lw=1); a.set_ylim(top=top * 1.35); a.set_xlabel('재검사하는 비율 (위험 점수 상위 k%)', fontsize=11.5); a.set_ylabel('제품 1만 개당 순이익', fontsize=11.5)
a.legend(fontsize=11, frameon=False, loc='lower left'); a.set_title(f'② 재검사를 얼마나 할까 — 불량 1개 잡는 가치 = 재검사 비용의 {V}배 가정', fontsize=13, weight='bold', loc='left')
b = fig.add_subplot(gs[1, 0]); v = [prec[n][3.0] * 100 for n in names]
b.bar(['무작위', '원본 변수\n모델', '최종\n모델'], v, color=[COL[n] for n in names], width=.6)
for i, x in enumerate(v): b.text(i, x + .8, f'{x:.1f}개', ha='center', fontsize=13, weight='bold', color=RED if i == 2 else DARK)
b.set_ylim(0, max(v) * 1.3); b.set_ylabel('재검사 100회당 잡는 불량'); b.set_title('① 같은 검사 횟수로 잡는 불량\n(생산량의 3% 재검사)', fontsize=12.5, weight='bold', loc='left')
c = fig.add_subplot(gs[1, 1]); be = [1 / prec[n][3.0] for n in names]
c.bar(['무작위', '원본 변수\n모델', '최종\n모델'], be, color=[COL[n] for n in names], width=.6)
for i, x in enumerate(be): c.text(i, x + .4, f'{x:.1f}배', ha='center', fontsize=13, weight='bold', color=RED if i == 2 else DARK)
c.set_ylim(0, max(be) * 1.3); c.set_ylabel('필요한 "불량 1개 가치 ÷ 재검사 비용"'); c.set_title('③ 손익분기점 — 낮을수록 쉽게 이득\n(상위 3% 재검사 기준)', fontsize=12.5, weight='bold', loc='left')
fig.text(.01, .01, f"같은 기간 5-fold 예측 · 제품 {N:,}개 · 불량 {int(y.sum()):,}개 기준. 실제 비용은 데이터에 없어 비율로 가정함 (재검사 1회 = 1)", fontsize=10.5, color='#222222')
plt.savefig('figs/proj_roi_cv.png', dpi=150, bbox_inches='tight'); plt.close(); print('ok')
