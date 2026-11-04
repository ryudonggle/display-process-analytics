"""36단계: 재검사 ROI 시각화 — 다음 달 예측(2023-12 ~ 2024-04 롤링) 기준, 무작위 · 원본 변수 모델 · 최종 모델
가정 (실제 비용은 데이터에 없음 → 비율로 표현)
- 재검사 1회 비용 = 1
- 불량 1개를 재검사로 미리 잡을 때 아끼는 가치 = V (재검사 비용의 V배). 시나리오 V = 5 · 10 · 20
- 순이익 = V × TP − (재검사 수), ROI = 순이익 ÷ 재검사 비용
- 손익분기 V = 1 / 정밀도 (불량 하나 잡는 가치가 이 배수를 넘으면 재검사가 이득)
결과: reports/36_roi.json, figs/proj_roi.png
"""
import json, numpy as np, pandas as pd, warnings
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
warnings.filterwarnings('ignore')
exec(open('src/s32_feature_ladder.py', encoding='utf-8').read().split('MONTHS = ')[0])
for w in ('Regular', 'Bold'): fm.fontManager.addfont(f'fonts/NotoSansKR-{w}.ttf')
plt.rcParams.update({'font.family': 'Noto Sans KR Static', 'axes.unicode_minus': False, 'axes.spines.top': False, 'axes.spines.right': False})
RED, BLUE, GREY, DARK = '#c0392b', '#8fa9cc', '#aaaaaa', '#1a2b4c'
MONTHS = ['2023-12', '2024-01', '2024-02', '2024-03', '2024-04']
SETS = {'원본 변수 모델': (COMMON + RAW, False), '최종 모델': (COMMON + MEAS + PHYS + ANAL, True)}
KS = np.r_[0.5, np.arange(1, 15.5, 0.5)]
VS = [5, 10, 20]
rng = np.random.default_rng(SEED)

# 달마다 상위 k%를 재검사 → 5개월 합산 TP · 검사 수
res = {nm: {k: [0, 0] for k in KS} for nm in list(SETS) + ['무작위']}
tot_n = tot_def = 0
for mo in MONTHS:
    A, B = tr[tr.mo < mo], tr[tr.mo == mo]; yb = B.y.values; tot_n += len(B); tot_def += yb.sum()
    P = {nm: fit_pred2(A, B, fs, dev, SEED) for nm, (fs, dev) in SETS.items()}
    P['무작위'] = rng.random(len(B))
    for nm, p in P.items():
        order = np.argsort(-p)
        for k in KS:
            n = max(1, int(round(len(B) * k / 100))); res[nm][k][0] += int(yb[order[:n]].sum()); res[nm][k][1] += n
base = tot_def / tot_n
# 무작위는 기댓값으로 대체(표본 흔들림 제거)
for k in KS: res['무작위'][k][0] = res['무작위'][k][1] * base

per = 10000 / tot_n   # 제품 1만 개당으로 환산
out = dict(base_rate=round(base, 4), months=MONTHS, n_products=int(tot_n), n_defects=int(tot_def), table={})
for nm in res:
    out['table'][nm] = {str(k): dict(TP=round(res[nm][k][0], 1), inspected=res[nm][k][1], precision=round(res[nm][k][0] / res[nm][k][1], 4)) for k in KS}
k3 = 3.0
prec3 = {nm: res[nm][k3][0] / res[nm][k3][1] for nm in res}
out['at_3pct'] = {nm: dict(precision=round(prec3[nm], 4), breakeven_V=round(1 / prec3[nm], 1),
                           roi={str(V): round(V * prec3[nm] - 1, 3) for V in VS}) for nm in res}
print(json.dumps(out['at_3pct'], ensure_ascii=False, indent=1))

COL = {'무작위': GREY, '원본 변수 모델': BLUE, '최종 모델': RED}
fig, ax = plt.subplots(1, 3, figsize=(18, 4.9), gridspec_kw=dict(width_ratios=[.9, 1.3, .9]))
# ① 재검사 100회당 잡는 불량
names = ['무작위', '원본 변수 모델', '최종 모델']; v = [prec3[n] * 100 for n in names]
ax[0].bar(['무작위로\n고를 때', '원본 변수\n모델', '최종\n모델'], v, color=[COL[n] for n in names], width=.6)
for i, x in enumerate(v): ax[0].text(i, x + .5, f'{x:.1f}개', ha='center', fontsize=12.5, weight='bold', color=RED if i == 2 else DARK)
ax[0].set_ylim(0, max(v) * 1.25); ax[0].set_ylabel('재검사 100회당 잡는 불량 (개)')
ax[0].set_title('① 같은 검사 횟수로 잡는 불량\n(매달 생산량의 3% 재검사)', fontsize=11, weight='bold', loc='left')
# ② 검사 비율별 순이익 (V = 10)
V = 10
for nm in names:
    net = [(V * res[nm][k][0] - res[nm][k][1]) * per for k in KS]
    ax[1].plot(KS, net, color=COL[nm], lw=2.4 if nm == '최종 모델' else 1.8, label=nm)
    if nm == '최종 모델':
        i = int(np.argmax(net)); ax[1].scatter([KS[i]], [net[i]], color=RED, zorder=3)
        ax[1].text(KS[i] + .6, net[i] - 10, f'최대 +{net[i]:.0f}\n(상위 {KS[i]:g}% 재검사)', fontsize=9.5, color=RED, weight='bold')
        top_net = net[i]
ax[1].axhline(0, color='#666', lw=1); ax[1].set_ylim(top=top_net * 1.45); ax[1].set_xlabel('매달 재검사하는 비율 (위험 점수 상위 k%)'); ax[1].set_ylabel('제품 1만 개당 순이익 (재검사 비용 = 1)')
ax[1].legend(fontsize=9, frameon=False, loc='lower left')
ax[1].set_title(f'② 재검사를 얼마나 할까? (불량 1개 잡는 가치 = 재검사 비용의 {V}배 가정)', fontsize=11, weight='bold', loc='left')
# ③ 손익분기 배수
be = [1 / prec3[n] for n in names]
ax[2].barh(['무작위로 고를 때', '원본 변수 모델', '최종 모델'][::-1], be[::-1], color=[COL[n] for n in names][::-1], height=.55)
for i, x in enumerate(be[::-1]): ax[2].text(x + .4, i, f'{x:.1f}배', va='center', fontsize=12, weight='bold', color=RED if i == 0 else DARK)
ax[2].set_xlim(0, max(be) * 1.25); ax[2].set_xlabel('불량 1개 잡는 가치가 재검사 비용의 몇 배를 넘어야 이득인가')
ax[2].set_title('③ 손익분기점: 낮을수록 쉽게 이득\n(상위 3% 재검사 기준)', fontsize=11, weight='bold', loc='left')
roi = {n: out['at_3pct'][n]['roi'][str(V)] for n in names}
fig.text(.01, -.06, f"ROI (상위 3% 재검사, 가치 {V}배 가정): 무작위 {roi['무작위']*100:+.0f}% · 원본 변수 모델 {roi['원본 변수 모델']*100:+.0f}% · 최종 모델 {roi['최종 모델']*100:+.0f}%"
         f"   |   다음 달 예측 5개월({MONTHS[0]} ~ {MONTHS[-1]}) 제품 {tot_n:,}개 · 불량 {int(tot_def):,}개 기준. 실제 비용은 데이터에 없어 비율로 가정함",
         fontsize=9.5, color='#444')
plt.tight_layout(); plt.savefig('figs/proj_roi.png', dpi=150, bbox_inches='tight'); plt.close()
json.dump(out, open('reports/36_roi.json', 'w'), ensure_ascii=False, indent=1)
