"""30단계: 졸업작품 보고서(A 원인 진단형 · B 예측 모델형)용 한국어 그림
수치 출처: 데이터에서 직접 계산 + reports/08 · 19 · 22 · 24 · 28 · 29 json (정지 길이별 표는 19_wait_degas.md §4-4)
결과: figs/grad_*.png
"""
import pandas as pd, numpy as np, json
from statsmodels.stats.proportion import proportion_confint
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib import font_manager as fm
for w in ('Regular', 'Bold'): fm.fontManager.addfont(f'fonts/NotoSansKR-{w}.ttf')   # 가변 폰트에서 만든 고정 굵기 폰트
plt.rcParams.update({'font.family': 'Noto Sans KR Static', 'axes.unicode_minus': False, 'axes.spines.top': False, 'axes.spines.right': False})
RED, BLUE, GREY, DARK = '#C0392B', '#B9C0C6', '#222222', '#222222'   # 포스터 기준: 원인=빨강, 비교=회색
J = lambda f: json.load(open(f'reports/{f}', encoding='utf-8'))
save = lambda n: (plt.savefig(f'figs/{n}.png', dpi=150, bbox_inches='tight'), plt.close())

d = pd.read_parquet('data/features.parquet'); d['ac_batch'] = d.groupby('date_AutoClave')['Set ID'].transform('size')
tr = d[d.is_train == 1].copy(); tr['y'] = tr.target.astype(int); tr['m'] = tr.date_Dam.dt.to_period('M').astype(str)
rule = (tr.fe_lead_total > 2) | (tr.equip_mismatch == 1) | (tr.fe_Fill1_zero_dispense == 1)
t = tr[(tr.date_Dam >= '2023-09-01') & ~rule & ~(tr['dev_Dam__Y(Stage2)'] < -0.05)].copy()


def bars(ax, labels, vals, colors, fmt='{:.1f}%', ns=None, ylim=None, yerr=None):
    ax.bar(labels, vals, color=colors, width=.62)
    if yerr is not None: ax.errorbar(range(len(vals)), vals, yerr=yerr, fmt='none', ecolor='#222222', capsize=3, lw=1)
    top = (np.array(vals) + (yerr[1] if yerr is not None else 0))
    for i, v in enumerate(vals):
        ax.text(i, top[i] + (ylim or max(top)) * .02, fmt.format(v) + (f'\n({ns[i]:,}건)' if ns is not None else ''), ha='center', fontsize=9)
    if ylim: ax.set_ylim(0, ylim)


# ── 공통: 공정 흐름도
fig, ax = plt.subplots(figsize=(12, 2.3)); ax.axis('off')
steps = [('Dam 도포', '바깥 가벽을 그린다'), ('Fill1 도포', '안쪽에 레진을 채운다'), ('Fill2 · UV', '레진을 빛으로 반쯤 굳힌다'),
         ('합착', '두 장을 붙인다'), ('탈포', '압력으로 기포를 뺀다'), ('검사', '정상 / 불량 판정')]
for i, (a, b) in enumerate(steps):
    x = i * 2.05
    ax.add_patch(FancyBboxPatch((x, .25), 1.7, 1.1, boxstyle='round,pad=0.05', fc='#eef2f8' if i not in (4,) else '#fbe9e7', ec=DARK, lw=1.2))
    ax.text(x + .85, .95, a, ha='center', va='center', fontsize=12, weight='bold', color=DARK); ax.text(x + .85, .55, b, ha='center', va='center', fontsize=9, color='#222222')
    if i < len(steps) - 1: ax.annotate('', xy=(x + 2.0, .8), xytext=(x + 1.78, .8), arrowprops=dict(arrowstyle='->', color=DARK, lw=1.5))
ax.set_xlim(-.1, 12.2); ax.set_ylim(0, 1.6)
ax.text(0, 1.5, '차량용 디스플레이 합착 공정 (도포 구간: Stage1 · 2 · 3) — 불량 유형: 기포 · 정렬 틀어짐 · 크랙', fontsize=10, color='#222222')
save('grad_process')

# ── A1: 9월 하락과 Y 정렬
mm = tr[~rule].groupby('m').agg(rate=('y', 'mean'), yneg=('dev_Dam__Y(Stage2)', lambda s: (s < -0.05).mean()))
fig, ax = plt.subplots(figsize=(10, 4.2)); x = range(len(mm))
ax.bar(x, mm.yneg * 100, color=BLUE, label='Stage2 구간 헤드 Y 좌표가 기준보다 어긋난 제품 비율 (왼쪽 축)')
a2 = ax.twinx(); a2.plot(x, mm.rate * 100, 'o-', color=RED, lw=2, label='불량률 (오른쪽 축)'); a2.set_ylim(0, 10); a2.set_ylabel('불량률 (%)')
a2.spines['top'].set_visible(False)
ax.axvline(3.5, color=DARK, ls='--', lw=1); ax.text(3.6, 92, '9월: Stage2 구간 헤드 정렬 보정\n→ 불량률 7.0% → 4.3%', fontsize=9, color=DARK)
ax.axvline(4.5, color=GREY, ls=':', lw=1); ax.text(4.6, 70, '10월: 레시피 변경\n→ 더 줄지 않음', fontsize=9, color='#222222')
ax.set_xticks(list(x)); ax.set_xticklabels([m[2:].replace('-', '.') for m in mm.index], fontsize=9); ax.set_ylabel('어긋난 제품 비율 (%)')
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, fontsize=8.5, loc='lower left', frameon=False)
ax.set_title('불량을 줄인 것은 레시피가 아니라 "헤드 정렬"이었다', fontsize=12, weight='bold', loc='left'); save('grad_A1_alignment')

# ── A2: 역인과 — 대기 중 라인 상태 / 정지 길이
o19 = J('19_wait_degas.json'); sk = pd.DataFrame(o19['stop_kind']).set_index(['long', 'stop_kind'])
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
v = [sk.loc[(1, k), 'mean'] * 100 for k in ['none', 'regular', 'irregular']]; n = [int(sk.loc[(1, k), 'size']) for k in ['none', 'regular', 'irregular']]
bars(ax[0], ['라인 계속 가동', '정기 휴식에\n걸림', '예정에 없던\n정지에 걸림'], v, [BLUE, BLUE, RED], ns=n, ylim=65)
ax[0].set_ylabel('불량률 (%)'); ax[0].set_title('45분 넘게 기다린 제품: 그동안 라인에 무슨 일이?', fontsize=10.5, loc='left')
# 19_wait_degas.md §4-4 ② 비정기 정지에 걸린 장기 대기 제품, 정지 길이별
bars(ax[1], ['30분 이하', '30~60분', '60분 초과'], [77.8, 36.1, 38.5], [RED, BLUE, BLUE], ns=[36, 36, 13], ylim=95)
ax[1].set_title('예정에 없던 정지: 짧게 멈출수록 불량이 많다', fontsize=10.5, loc='left'); ax[1].set_ylabel('불량률 (%)')
fig.text(.5, -.05, '→ 기다려서 불량이 된 것이 아니라, 문제가 생긴 제품 때문에 라인이 멈추고 그 제품이 기다린 것 (역인과)', ha='center', fontsize=10, color=DARK)
save('grad_A2_reverse')

# ── A3: 탈포 배치 크기
g = t.groupby(pd.cut(t.ac_batch, [0, 1, 3, 6, 10, 14, 100], labels=['1개', '2~3개', '4~6개', '7~10개', '11~14개', '15개+']), observed=True).y.agg(['size', 'sum'])
r = g['sum'] / g['size'] * 100; lo, hi = zip(*[proportion_confint(k, n, method='wilson') for k, n in zip(g['sum'], g['size'])])
fig, ax = plt.subplots(figsize=(9, 4.2))
bars(ax, list(g.index), r.values, [RED, RED, BLUE, BLUE, '#5b7db1', BLUE], ns=g['size'].values, ylim=32, yerr=[r.values - np.array(lo) * 100, np.array(hi) * 100 - r.values])
ax.axhline(t.y.mean() * 100, color=GREY, ls='--', lw=1); ax.text(2.5, t.y.mean() * 100 + 2.2, f'전체 평균 {t.y.mean()*100:.1f}% (점선)', fontsize=8.5, color='#222222', ha='center')
ax.set_xlabel('탈포기에 함께 넣은 제품 수'); ax.set_ylabel('불량률 (%)')
ax.set_title('탈포기를 덜 채우고 돌리면 불량이 2.7배', fontsize=12, weight='bold', loc='left'); save('grad_A3_batch')

# ── A4: 위험 조건 중첩
F = [t['Dam__Stage2 Line1 Distance Speed'] >= 6500, t['Fill2__CURE SPEED'] == 45, t.ac_batch <= 9]
k = np.minimum(sum(m.astype(int) for m in F), 2); g = t.groupby(k).y.agg(['size', 'mean'])
fig, ax = plt.subplots(figsize=(7, 4)); bars(ax, ['0개', '1개', '2개 이상'], g['mean'].values * 100, [BLUE, BLUE, RED], ns=g['size'].values, ylim=9)
ax.set_xlabel('겹친 위험 조건 수 (Dam 토출 스피드 빠름 · UV 과조사 · 탈포 9개 이하)'); ax.set_ylabel('불량률 (%)')
ax.set_title('위험 조건이 겹칠수록 불량률이 오른다', fontsize=12, weight='bold', loc='left'); save('grad_A4_stack')

# ── A5: 작업지시(로트) — 두 라인 불량률
w = t.groupby(['workorder', 'line']).y.agg(['size', 'mean']).query('size>=15').reset_index().pivot(index='workorder', columns='line', values='mean').dropna() * 100
fig, ax = plt.subplots(figsize=(5.6, 5)); ax.scatter(w[1], w[2], s=14, alpha=.5, color=DARK)
mx = max(w.max()) * 1.05; ax.plot([0, mx], [0, mx], color=GREY, ls='--', lw=1); ax.set_xlim(0, mx); ax.set_ylim(0, mx)
ax.set_xlabel('라인 #1에서의 불량률 (%)'); ax.set_ylabel('라인 #2에서의 불량률 (%)')
ax.set_title(f'같은 작업지시는 두 라인에서 비슷하게 좋거나 나쁘다\n(작업지시 {len(w)}개, 상관 {w[1].corr(w[2]):.2f})', fontsize=10.5, loc='left'); save('grad_A5_lot')

# ── B1: 랜덤 분할 vs 시간 순
fig, ax = plt.subplots(figsize=(8, 4))
bars(ax, ['같은 기간\n무작위 분할\n(대회 방식)', '미래 기간\n예측\n(현장 방식)', '알려진 원인 제외\n미래 예측', '무작위로\n찍었을 때'], [0.284, 0.148, 0.093, 0.058],
     [BLUE, RED, RED, '#dddddd'], fmt='{:.3f}', ylim=.34)
ax.set_ylabel('PR-AUC (높을수록 좋음)'); ax.set_title('대회 점수와 현장 성능은 다르다', fontsize=12, weight='bold', loc='left'); save('grad_B1_split')

# ── B2: 상위 k% 재검사
o8 = J('08_validation.json'); ks = ['top1pct', 'top3pct', 'top5pct', 'top10pct']; lab = ['상위 1%', '상위 3%', '상위 5%', '상위 10%']
fig, ax = plt.subplots(figsize=(9, 4)); x = np.arange(4)
p = [o8[k]['precision'] * 100 for k in ks]; rc = [o8[k]['recall'] * 100 for k in ks]
ax.bar(x - .2, p, .38, color=RED, label='정밀도: 골라낸 것 중 진짜 불량 비율'); ax.bar(x + .2, rc, .38, color=BLUE, label='재현율: 전체 불량 중 잡아낸 비율')
for i in range(4): ax.text(i - .2, p[i] + 1.5, f'{p[i]:.0f}%', ha='center', fontsize=9); ax.text(i + .2, rc[i] + 1.5, f'{rc[i]:.0f}%', ha='center', fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_ylabel('%'); ax.set_ylim(0, 100); ax.legend(fontsize=8.5, frameon=False)
ax.set_title('위험 점수 상위 몇 %를 다시 검사하면? (같은 기간 기준)', fontsize=11.5, weight='bold', loc='left'); save('grad_B2_topk')

# ── B3: 성능 개선 시도 사다리
o29 = J('29_feedback_and_simple.json')['results']['all_train']
items = [('기준: 제품 한 행 정보', 'base (test 한 행)'), ('+ 실시간 생산 로그', 'B (실시간 로그)'), ('+ 검사 결과 2시간 뒤', 'B + 검사 2h 후'),
         ('+ 검사 결과 8시간 뒤', 'B + 검사 8h 후'), ('+ 검사 결과 24시간 뒤', 'B + 검사 24h 후'), ('단순 로지스틱 (조건 15개)', '단순 로지스틱 (견고 조건 15개)'),
         ('작은 LightGBM (강한 규제)', '작은 LightGBM (B)')]
fig, ax = plt.subplots(figsize=(9, 4.2)); vals = [o29[k]['mean_lift'] for _, k in items]
ax.barh(range(len(items))[::-1], vals, color=[RED] + [BLUE] * (len(items) - 1), height=.6)
for i, v in enumerate(vals): ax.text(v + .03, len(items) - 1 - i, f'{v:.2f}배', va='center', fontsize=9)
ax.set_yticks(range(len(items))[::-1]); ax.set_yticklabels([a for a, _ in items], fontsize=9.5); ax.set_xlim(0, 3.8); ax.axvline(1, color=GREY, ls='--', lw=1)
ax.set_xlabel('무작위 대비 몇 배 잘 맞히나 (5개월 롤링 평균)'); ax.set_title('정보를 더 줘도 약 3배에서 멈춘다 — 천장은 "정보"', fontsize=12, weight='bold', loc='left'); save('grad_B3_ceiling')

# ── B4: 월별 롤링
o28 = J('28_stream_model.json')['results']['all_train']
fig, ax = plt.subplots(figsize=(9, 3.8))
for tag, col, nm in [('base (test 한 행)', RED, '제품 한 행 정보'), ('B (실시간 로그)', BLUE, '+ 실시간 생산 로그')]:
    M = pd.DataFrame(o28[tag]['monthly']); ax.plot([m[2:].replace('-', '.') for m in M.month], M.lift, 'o-', color=col, lw=2, label=nm)
ax.axhline(1, color=GREY, ls='--', lw=1); ax.set_ylim(0, 4.5); ax.set_ylabel('무작위 대비 (배)'); ax.legend(fontsize=9, frameon=False)
ax.set_title('달마다 "다음 달"을 예측해 보면 (2023.12 ~ 2024.04)', fontsize=11.5, weight='bold', loc='left'); save('grad_B4_rolling')

# ── B5: 공정별 기여도 (SHAP)
ps = J('07_shap_summary.json')['process_share']; ren = {'Dam': 'Dam 도포', '운영(대기·작업지시)': '운영 (대기·작업지시)', 'Fill1': 'Fill1 도포', 'AutoClave(탈포)': '탈포', 'Fill2(UV)': 'Fill2 · UV', '기타(모델·라인)': '기타'}
s = pd.Series(ps).rename(ren).sort_values()
fig, ax = plt.subplots(figsize=(7.5, 3.6)); ax.barh(s.index, s.values, color=[RED if v == s.max() else BLUE for v in s.values], height=.6)
for i, v in enumerate(s.values): ax.text(v + .5, i, f'{v:.1f}%', va='center', fontsize=9)
ax.set_xlabel('모델 판단에 기여한 몫 (%)'); ax.set_xlim(0, 42); ax.set_title('모델은 어느 공정을 보고 불량을 판단했나 (SHAP)', fontsize=11.5, weight='bold', loc='left'); save('grad_B5_shap')
# ── A6: 구간별 — 토출량이 가장 적은 Stage2 구간이 가장 약하다 (전체 기간, 규칙 제외)
u = tr[~rule]; REG = [(1, 'Stage1'), (2, 'Stage2'), (3, 'Stage3')]
vol = [u[f'Dam__Dispense Volume(Stage{s})'].median() for s, _ in REG]
r_bad = [u.y[u[f'dev_Dam__Y(Stage{s})'] < -0.05].mean() * 100 for s, _ in REG]; r_ok = [u.y[u[f'dev_Dam__Y(Stage{s})'] >= -0.05].mean() * 100 for s, _ in REG]
fig, ax = plt.subplots(1, 2, figsize=(11, 4)); lab = [l for _, l in REG]
bars(ax[0], lab, vol, [BLUE, RED, BLUE], fmt='{:.2f}', ylim=1.3); ax[0].set_ylabel('Dam 레진 토출량 (중앙값)')
ax[0].set_title('Stage2 구간은 가벽 레진이 절반뿐', fontsize=10.5, loc='left')
x = np.arange(3); ax[1].bar(x - .2, r_ok, .38, color='#cfd8e6', label='좌표 정상'); ax[1].bar(x + .2, r_bad, .38, color=[BLUE, RED, BLUE], label='좌표 Y가 기준보다 어긋남')
for i in range(3): ax[1].text(i - .2, r_ok[i] + .2, f'{r_ok[i]:.1f}%', ha='center', fontsize=9); ax[1].text(i + .2, r_bad[i] + .2, f'{r_bad[i]:.1f}%', ha='center', fontsize=9)
ax[1].set_xticks(x); ax[1].set_xticklabels(lab); ax[1].set_ylabel('불량률 (%)'); ax[1].set_ylim(0, 11); ax[1].legend(fontsize=8.5, frameon=False)
ax[1].set_title('같은 좌표 오차도 Stage2에서만 불량으로 이어진다', fontsize=10.5, loc='left')
fig.text(.5, -.04, '→ 가장 얇은 가벽이 가장 약하다', ha='center', fontsize=10.5, color=DARK, weight='bold'); save('grad_A6_seg')
print('figs done')
