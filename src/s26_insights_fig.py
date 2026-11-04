"""26단계: 인사이트 보고서(16_sep_insights.md) 요약 그림 — 23단계 점검을 통과한 결과만
① 9월 불량률 하락 = Dam Y 정렬 보정 (레시피 변경 전) ② 탈포 배치 크기 ③ 견고한 위험 조건(Dam 고속 · UV 45 · 작은 배치) 중첩
결과: figs/16_sep_insights.png
"""
import pandas as pd, numpy as np
from statsmodels.stats.proportion import proportion_confint
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

d = pd.read_parquet('data/features.parquet'); d['ac_batch'] = d.groupby('date_AutoClave')['Set ID'].transform('size')
tr = d[d.is_train == 1].copy(); tr['y'] = tr.target.astype(int); tr['m'] = tr.date_Dam.dt.to_period('M').astype(str)
rule = (tr.fe_lead_total > 2) | (tr.equip_mismatch == 1) | (tr.fe_Fill1_zero_dispense == 1)
t = tr[(tr.date_Dam >= '2023-09-01') & ~rule & ~(tr['dev_Dam__Y(Stage2)'] < -0.05)].copy()
ci = lambda k, n: proportion_confint(k, n, method='wilson')

fig, ax = plt.subplots(1, 3, figsize=(19, 4.8))
mm = tr[~rule].groupby('m').agg(rate=('y', 'mean'), yneg=('dev_Dam__Y(Stage2)', lambda s: (s < -0.05).mean()))
a = ax[0]; a.bar(range(len(mm)), mm.yneg * 100, color='#9fb6d4', label='Dam Y(Stage2) negative dev. share (%)')
a2 = a.twinx(); a2.plot(range(len(mm)), mm.rate * 100, 'o-', color='#c0392b', label='defect rate (%)'); a2.set_ylim(0, 10)
a.axvline(3.5, color='grey', ls='--'); a.axvline(4.5, color='grey', ls=':'); a.text(3.6, 60, 'Y aligned\n(Sep)', fontsize=8); a.text(4.6, 80, 'new recipe\n(Oct)', fontsize=8)
a.set_xticks(range(len(mm))); a.set_xticklabels(mm.index, rotation=60, fontsize=8); a.set_ylabel('Y negative dev. share %'); a2.set_ylabel('defect %')
h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels(); a.legend(h1 + h2, l1 + l2, fontsize=8, loc='upper right')
a.set_title('① Defect drop came with Y alignment, before the recipe change', fontsize=10)

g = t.groupby(pd.cut(t.ac_batch, [0, 1, 3, 6, 10, 14, 100], labels=['1', '2-3', '4-6', '7-10', '11-14', '15+']), observed=True).y.agg(['size', 'sum'])
g['r'] = g['sum'] / g['size'] * 100; lo, hi = zip(*[ci(k, n) for k, n in zip(g['sum'], g['size'])])
b = ax[1]; b.bar(range(len(g)), g.r, color=['#c0392b', '#c0392b'] + ['#9fb6d4'] * 4)
b.errorbar(range(len(g)), g.r, yerr=[g.r - np.array(lo) * 100, np.array(hi) * 100 - g.r], fmt='none', ecolor='#555', capsize=3)
for i, (r_, n_) in enumerate(zip(g.r, g['size'])): b.text(i, np.array(hi)[i] * 100 + .5, f'{r_:.1f}%\nn={int(n_):,}', ha='center', fontsize=8)
b.set_xticks(range(len(g))); b.set_xticklabels(g.index); b.set_xlabel('products loaded into the degas chamber together'); b.set_ylabel('defect %')
b.set_ylim(0, 30); b.axhline(t.y.mean() * 100, color='grey', ls='--', lw=.8); b.set_title('② Under-filled degas loads: up to 4x defects', fontsize=10)

F = [t['Dam__Stage2 Line1 Distance Speed'] >= 6500, t['Fill2__CURE SPEED'] == 45, t.ac_batch <= 9]
n = np.minimum(sum(m.astype(int) for m in F), 2); g = t.groupby(n).y.agg(['size', 'sum']); g['r'] = g['sum'] / g['size'] * 100
c = ax[2]; c.bar(['0', '1', '2+'], g.r, color=['#9fb6d4', '#9fb6d4', '#c0392b'])
for i, (r_, n_) in enumerate(zip(g.r, g['size'])): c.text(i, r_ + .2, f'{r_:.1f}%\nn={int(n_):,}', ha='center', fontsize=9)
c.set_xlabel('# robust risk conditions: Dam speed≥6500, UV speed 45, degas load≤9'); c.set_ylabel('defect %'); c.set_ylim(0, 9)
c.axhline(t.y.mean() * 100, color='grey', ls='--', lw=.8); c.set_title('③ Robust conditions stack up (2023-09+, known causes removed)', fontsize=10)
fig.subplots_adjust(wspace=.33); plt.savefig('figs/16_sep_insights.png', dpi=120, bbox_inches='tight')
print(g)
