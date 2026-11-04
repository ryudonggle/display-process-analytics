"""39단계: 원인 해결로 얻는 이득 — 늘어나는 양품 수 · 수율 · 연간 이득(디스플레이 1개 가치 시나리오)
초과 불량 = 해당 조건의 불량 − 정상 조건의 불량률로 기대되는 불량
- Seg 노즐 위치: 보정 전 좌표로 만든 제품의 초과 불량 412개(train, 16_actionable) — 9월 보정으로 이미 얻은 이득
- 탈포 작은 배치(≤ 3개, 4 ~ 9개) · 챔버 ≤ 46℃: 2023-09 이후 · 알려진 원인 제외 집단, 기준 = 배치 10개 이상 / 챔버 47℃ 이상
환산: train은 전체 제품의 70% → × (전체 ÷ train), 8개월 → 12개월
디스플레이 1개 가치(확정 단가 없음): 30만 · 60만 · 120만 원 — 12인치 계기판 모듈 약 410달러 추정치(출처 신뢰도 낮음) 주변으로 설정
결과: reports/39_gain.json, figs/proj_gain.png
"""
import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for w in ('Regular', 'Bold'): fm.fontManager.addfont(f'fonts/NotoSansKR-{w}.ttf')
plt.rcParams.update({'font.family': 'Noto Sans KR Static', 'axes.unicode_minus': False, 'axes.spines.top': False, 'axes.spines.right': False})
RED, BLUE, TEAL, DARK, GREY = '#c0392b', '#8fa9cc', '#1D6A72', '#1a2b4c', '#aaaaaa'

d = pd.read_parquet('data/features.parquet'); d['ac_batch'] = d.groupby('date_AutoClave')['Set ID'].transform('size')
tr = d[d.is_train == 1].copy(); tr['y'] = tr.target.astype(int); N = len(d); k_rows = N / len(tr)
rule = (tr.fe_lead_total > 2) | (tr.equip_mismatch == 1) | (tr.fe_Fill1_zero_dispense == 1)
t = tr[(tr.date_Dam >= '2023-09-01') & ~rule & ~(tr['dev_Dam__Y(Stage2)'] < -0.05)]
k_year = 12 / t.date_Dam.dt.to_period('M').nunique()
ref = t[t.ac_batch >= 10].y.mean()
small = t.ac_batch <= 3; mid = (t.ac_batch >= 4) & (t.ac_batch <= 9); cold = t['AutoClave__Chamber Temp.'] <= 46
ex = lambda m, r: t.y[m].sum() - m.sum() * r
CAUSES = [('Seg 구간 노즐 위치 보정', 412 * k_rows, '이미 얻은 이득 (2023년 9월 보정) · 재발 감시로 유지'),
          ('탈포기 1 ~ 3개만 넣고 돌리기 방지', ex(small, ref) * k_rows * k_year, '최소 투입 수량 기준'),
          ('탈포기 4 ~ 9개만 넣고 돌리기 방지', ex(mid, ref) * k_rows * k_year, '배치 채워서 돌리기'),
          ('탈포 챔버 46℃ 이하 방지', ex(cold, t.y[~cold].mean()) * k_rows * k_year, '온도 하한 47℃')]
PRICES = [30, 60, 120]   # 만 원
defect_year = tr.y.mean() * N
out = dict(products_per_year=N, defect_rate=round(tr.y.mean(), 4), defects_per_year=round(defect_year), prices_10k_won=PRICES, causes=[])
for nm, n, how in CAUSES:
    out['causes'].append(dict(cause=nm, extra_good_units=round(n), yield_gain_pp=round(n / N * 100, 2), how=how,
                              gain_100m_won={str(p): round(n * p / 1e4, 2) for p in PRICES}))
tot = sum(c['extra_good_units'] for c in out['causes']); fut = sum(c['extra_good_units'] for c in out['causes'][1:])
out['total'] = dict(extra_good_units=tot, yield_gain_pp=round(tot / N * 100, 2), gain_100m_won={str(p): round(tot * p / 1e4, 2) for p in PRICES})
out['future_only'] = dict(extra_good_units=fut, yield_gain_pp=round(fut / N * 100, 2), gain_100m_won={str(p): round(fut * p / 1e4, 2) for p in PRICES})
json.dump(out, open('reports/39_gain.json', 'w'), ensure_ascii=False, indent=1); print(json.dumps(out, ensure_ascii=False, indent=1))

def won(x):
    """억 단위 실수 → 'N억 M천만 원'"""
    eok = int(x); chun = int(round((x - eok) * 10))
    if chun == 10: eok, chun = eok + 1, 0
    return (f'{eok}억 ' if eok else '') + (f'{chun}천만 ' if chun else '') + '원'


MID = '60'; C = out['causes']; tot_mid = out['total']['gain_100m_won'][MID]; fut_mid = out['future_only']['gain_100m_won'][MID]
fig = plt.figure(figsize=(16, 5.2)); gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.2, .95], wspace=.55)
# ① 큰 숫자
a = fig.add_subplot(gs[0]); a.axis('off')
a.add_patch(plt.Rectangle((0, 0), 1, 1, transform=a.transAxes, fc='#fbecea', ec='none'))
a.text(.06, .88, '개선안을 적용하면 1년에', fontsize=14, color=DARK, transform=a.transAxes)
a.text(.06, .64, f"양품 +{out['future_only']['extra_good_units']:,}개", fontsize=32, weight='bold', color=RED, transform=a.transAxes)
a.text(.06, .50, f"{won(fut_mid)} · 수율 +{out['future_only']['yield_gain_pp']:.1f}%p", fontsize=14, weight='bold', color=RED, transform=a.transAxes)
a.text(.06, .34, f"노즐 보정으로 이미 얻은 이득 포함 시", fontsize=11, color=DARK, transform=a.transAxes)
a.text(.06, .24, f"양품 +{out['total']['extra_good_units']:,}개 · {won(tot_mid)}", fontsize=12, color=DARK, transform=a.transAxes)
a.text(.06, .08, '디스플레이 1개 60만 원 가정', fontsize=10, color='#666', transform=a.transAxes)
# ② 원인별 양품
b = fig.add_subplot(gs[1]); SHORT = ['Seg 노즐 위치 보정', '탈포기 1~3개 방지', '탈포기 4~9개 방지', '챔버 46℃ 이하 방지']; names = SHORT; n = [c['extra_good_units'] for c in C]; cols = [TEAL, RED, '#e07b6e', '#e3a857']
b.barh(names[::-1], n[::-1], color=cols[::-1], height=.55)
for i, c in enumerate(C[::-1]): b.text(c['extra_good_units'] + 8, i, f"+{c['extra_good_units']:,}개 · {won(c['gain_100m_won'][MID])}", va='center', fontsize=10.5, weight='bold', color=DARK)
b.set_xlim(0, max(n) * 1.75); b.set_xlabel('1년에 늘어나는 양품 (개)'); b.tick_params(axis='y', labelsize=10.5)
b.set_title('원인별로 늘어나는 양품과 이득', fontsize=12.5, weight='bold', loc='left')
# ③ 단가 시나리오
c3 = fig.add_subplot(gs[2]); x = np.arange(3)
done = [C[0]['gain_100m_won'][str(p)] for p in PRICES]; more = [out['future_only']['gain_100m_won'][str(p)] for p in PRICES]
c3.bar(x, done, .6, color=TEAL, label='이미 얻은 이득 (Seg 노즐 보정)'); c3.bar(x, more, .6, bottom=done, color=RED, label='앞으로 얻을 이득')
for i in range(3): c3.text(i, done[i] + more[i] + .25, won(done[i] + more[i]), ha='center', fontsize=11, weight='bold', color=RED if i == 1 else DARK)
c3.set_xticks(x); c3.set_xticklabels([f'{p}만 원' for p in PRICES]); c3.set_xlabel('디스플레이 1개 가치 (가정)'); c3.set_ylabel('연간 이득 (억 원)')
c3.set_ylim(0, (done[-1] + more[-1]) * 1.25); c3.legend(fontsize=9, frameon=False, loc='upper left')
c3.set_title('단가에 따라 달라지는 이득', fontsize=12.5, weight='bold', loc='left')
fig.text(.01, -.04, f"데이터의 연간 생산 {N:,}개(불량 약 {round(defect_year):,}개) · 이 라인 · 합착 공정만 계산. 확정 단가가 없어 1개 가치를 30만 · 60만 · 120만 원으로 가정"
         "(12인치 계기판 모듈 약 410달러 추정치 참고). 원인별 효과는 관찰된 관계에서 추정한 값으로 시험 확인이 필요함", fontsize=9.5, color='#444')
plt.savefig('figs/proj_gain.png', dpi=150, bbox_inches='tight'); plt.close(); print('ok')
