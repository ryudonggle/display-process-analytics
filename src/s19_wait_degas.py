"""19단계: Fill2 → 탈포 대기 × 탈포 레시피 상호작용 심층 분석
- 대상: 2023-09 이후, 라인 변경·Fill1 미도포·Dam Y 음수 편차 제외 (2시간 초과 대기는 구간표에서만 포함)
- 탈포 레시피: 총시간 ≥ 400 = 긴 탈포(2단 저압 90~121초 유지), < 400 = 짧은 탈포(2단 1초). 2023-11-06에 전환
- 검증: ① 구간표 ② 전환 전후 3주 비교 ③ 월·모델·라인 통제 로지스틱(상호작용항) ④ 9월 이전 재현 ⑤ 휴식 시간과의 관계
결과: reports/19_wait_degas.json, figs/19_wait_degas.png
"""
import pandas as pd, numpy as np, json, warnings
import statsmodels.formula.api as smf
from scipy.stats import fisher_exact
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

d = pd.read_parquet('data/features.parquet'); tr = d[d.is_train == 1].copy(); tr['y'] = tr.target.astype(int)
tr['m'] = tr.date_Dam.dt.to_period('M').astype(str); tr['short'] = (tr.fe_AC_time_total < 400).astype(int)
tr['w'] = pd.cut(tr.fe_lead_Fill2_AC, [0, 35, 45, 60, 120, 1e6], labels=['≤35', '36–45', '46–60', '61–120', '>120'])
other = (tr.equip_mismatch == 1) | (tr.fe_Fill1_zero_dispense == 1); yneg = tr['dev_Dam__Y(Stage2)'] < -0.05
s = tr[(tr.date_Dam >= '2023-09-01') & ~other & ~yneg].copy()
c = s[s.fe_lead_total <= 2].copy(); c['long'] = (c.fe_lead_Fill2_AC > 45).astype(int)
out = {}
cell = lambda g: g.y.agg(['size', 'sum', 'mean']).round(4)

# ① 대기 구간 × 탈포 레시피
G = s.groupby(['w', 'short'], observed=True).pipe(cell).reset_index(); out['grid_sep'] = G.to_dict('records')
# 탈포 레시피 구성
P = ['AutoClave__1st Pressure', 'AutoClave__1st Pressure Unit Time', 'AutoClave__2nd Pressure', 'AutoClave__2nd Pressure Unit Time',
     'AutoClave__3rd Pressure', 'AutoClave__3rd Pressure Unit Time']
out['recipe'] = s.groupby('short')[P].median().round(3).to_dict('index')
out['switch_by_month'] = pd.crosstab(s.m, s.short).to_dict('index')

# ② 전환(2023-11-06) 전후 3주
n = c[(c.date_AutoClave >= '2023-10-16') & (c.date_AutoClave < '2023-11-27')]
W = n.groupby(['long', 'short']).pipe(cell).reset_index(); out['window'] = W.to_dict('records')
for k in (0, 1):
    ct = pd.crosstab(n[n.long == k].short, n[n.long == k].y); out[f'window_fisher_long{k}'] = float(fisher_exact(ct.values)[1])

# ③ 통제 로지스틱 (2시간 규칙 제외)
f = smf.logit('y ~ long*short + C(m) + C(model) + C(line)', c).fit(disp=0); ci = np.exp(f.conf_int())
out['logit'] = {k: dict(OR=round(float(np.exp(f.params[k])), 2), lo=round(float(ci.loc[k, 0]), 2), hi=round(float(ci.loc[k, 1]), 2),
                        p=float(f.pvalues[k])) for k in ['long', 'short', 'long:short']}

# ④ 9월 이전 재현 (모두 긴 탈포) — 대기 효과의 재현
p = tr[(tr.date_Dam < '2023-09-01') & ~other & ~yneg]
out['pre_sep_long_degas'] = p[p.short == 0].groupby('w', observed=True).pipe(cell).reset_index().to_dict('records')

# ⑤ 대기 중 라인 정지 — 로그만으로 정의 (수기 데이터 미사용)
#   같은 날·같은 라인의 Fill2 완료 시각(train+test 전체, 라벨 불필요)이 20분 이상 끊기면 '정지'
#   제품의 Fill2 → 탈포 사이에 정지가 시작되면 brk=1
A_ = d[d.date_Fill2 >= '2023-09-01'][['date_Fill2', 'line']]
gaps = []
for (dy, ln), g in A_.groupby([A_.date_Fill2.dt.normalize(), 'line']):
    ts = np.sort(g.date_Fill2.unique())
    gaps += [(dy, ln, pd.Timestamp(a), pd.Timestamp(b)) for a, b in zip(ts[:-1], ts[1:]) if b - a >= np.timedelta64(20, 'm')]
GP = pd.DataFrame(gaps, columns=['day', 'line', 'a', 'b']); GP['len'] = (GP.b - GP.a).dt.total_seconds() / 60
out['stoppage'] = dict(n=len(GP), len_median=float(GP.len.median()), top_start=GP.a.dt.strftime('%H:%M').value_counts().head(8).to_dict())
gk = GP.groupby(['day', 'line'])
def stopped(r):
    try: g = gk.get_group((r.date_Fill2.normalize(), r.line))
    except KeyError: return 0
    return int(((g.a >= r.date_Fill2) & (g.a < r.date_AutoClave)).any())
c['brk'] = c.apply(stopped, axis=1)
# 정기 휴식(매일 같은 시각에 시작하는 정지: 09:50·11:50·15:00·17:50 전후) vs 비정기 정지
REG = {'09:50', '10:00', '11:50', '12:00', '15:00', '15:10', '17:50', '18:00'}
GP['reg'] = GP.a.dt.strftime('%H:%M').isin(REG); gk = GP.groupby(['day', 'line'])
out['stoppage'] |= dict(regular_share=round(float(GP.reg.mean()), 3), len_median_by_reg=GP.groupby('reg').len.median().to_dict())
def kind(r):
    try: g = gk.get_group((r.date_Fill2.normalize(), r.line))
    except KeyError: return 'none'
    h = g[(g.a >= r.date_Fill2) & (g.a < r.date_AutoClave)]
    return 'none' if h.empty else ('regular' if h.reg.all() else 'irregular')
c['stop_kind'] = c.apply(kind, axis=1)
out['stop_kind'] = c.groupby(['long', 'stop_kind']).pipe(cell).reset_index().to_dict('records')
out['stop_kind_x_degas_long_wait'] = c[c.long == 1].groupby(['short', 'stop_kind']).pipe(cell).reset_index().to_dict('records')
out['break'] = c.groupby(['long', 'brk']).pipe(cell).reset_index().to_dict('records')
out['break_x_degas_long_wait'] = c[c.long == 1].groupby(['short', 'brk']).pipe(cell).reset_index().to_dict('records')

# ⑥ 개선 여지: 짧은 탈포 × 장기 대기 제품이 긴 탈포 수준(같은 대기 구간)이었다면
for lab, lo, hi in [('46–60', 45, 60)]:
    a = c[(c.fe_lead_Fill2_AC > lo) & (c.fe_lead_Fill2_AC <= hi)]
    r0 = a[a.short == 0].y.mean(); b = a[a.short == 1]
    out['avoidable_46_60'] = dict(n=len(b), defects=int(b.y.sum()), expected_if_long=round(len(b) * r0, 1), avoidable=round(b.y.sum() - len(b) * r0, 1))

json.dump(out, open('reports/19_wait_degas.json', 'w'), ensure_ascii=False, indent=1, default=str)

# 그림
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
labs = ['≤35', '36–45', '46–60', '61–120']; x = np.arange(len(labs))
for k, (lab, col) in enumerate([('long degas (≥400s)', '#9fb6d4'), ('short degas (<400s)', '#c0392b')]):
    g = G[(G.short == k) & G.w.isin(labs)].set_index('w').reindex(labs)
    bars = ax[0].bar(x + (k - .5) * .38, g['mean'] * 100, .38, color=col, label=lab)
    for xi, (m_, n_) in zip(x + (k - .5) * .38, zip(g['mean'], g['size'])):
        if pd.notna(m_): ax[0].text(xi, m_ * 100 + 1.5, f'{m_*100:.1f}%', ha='center', fontsize=8)
ax[0].set_xticks(x); ax[0].set_xticklabels(labs); ax[0].set_xlabel('Fill2 → degas wait (min)'); ax[0].set_ylabel('defect %'); ax[0].set_ylim(0, 105)
ax[0].legend(fontsize=8, loc='upper left'); ax[0].set_title('Wait x degas recipe (2023-09+)', fontsize=10)
lab2 = ['wait ≤45\nlong degas', 'wait ≤45\nshort degas', 'wait >45\nlong degas', 'wait >45\nshort degas']
v = W.set_index(['long', 'short']).reindex([(0, 0), (0, 1), (1, 0), (1, 1)])
ax[1].bar(lab2, v['mean'] * 100, color=['#9fb6d4', '#e6a39b', '#9fb6d4', '#c0392b'])
for i, (m_, n_, s_) in enumerate(zip(v['mean'], v['size'], v['sum'])): ax[1].text(i, m_ * 100 + 1, f'{m_*100:.1f}%\n{int(s_)}/{int(n_)}', ha='center', fontsize=8.5)
ax[1].set_ylabel('defect %'); ax[1].set_ylim(0, 50); ax[1].set_title('Before/after recipe switch (2023-10-16 ~ 11-26)', fontsize=10)
B = pd.DataFrame(out['stop_kind']).set_index(['long', 'stop_kind']).reindex([(1, 'none'), (1, 'regular'), (1, 'irregular')])
lab3 = ['line kept running', 'regular break\n(09:50·11:50·15:00·17:50)', 'irregular stop']
ax[2].bar(lab3, B['mean'] * 100, color=['#9fb6d4', '#e6a39b', '#c0392b'])
for i, (m_, n_, s_) in enumerate(zip(B['mean'], B['size'], B['sum'])): ax[2].text(i, m_ * 100 + 1, f'{m_*100:.1f}%\n{int(s_)}/{int(n_)}', ha='center', fontsize=8.5)
ax[2].set_ylabel('defect %'); ax[2].set_ylim(0, 65); ax[2].set_title('Long wait (>45 min): what happened on the line', fontsize=10)
fig.subplots_adjust(wspace=.3); plt.savefig('figs/19_wait_degas.png', dpi=120, bbox_inches='tight')
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
