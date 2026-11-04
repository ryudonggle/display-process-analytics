"""25단계: 라인별 일일 불량률 p-관리도 → 관리 한계 밖(이상) 날짜 특정 → 그날 무엇이 달랐나
- p-관리도: 중심선 = 9월 이후 해당 라인 평균 불량률, 한계 = p ± 3·sqrt(p(1-p)/n_day) (일별 n 반영)
- 이상일 vs 정상일 비교: 주요 설정값·작은 배치 비율·모델 구성·작업지시 수
대상: 2023-09 이후, 규칙·Dam Y 음수 편차 제외 (train 라벨)
결과: reports/25_line_spc.json, figs/25_line_spc.png
"""
import pandas as pd, numpy as np, json, warnings
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

d = pd.read_parquet('data/features.parquet'); d['ac_batch'] = d.groupby('date_AutoClave')['Set ID'].transform('size')
tr = d[d.is_train == 1].copy(); tr['y'] = tr.target.astype(int); tr['day'] = tr.date_Dam.dt.normalize()
rule = (tr.fe_lead_total > 2) | (tr.equip_mismatch == 1) | (tr.fe_Fill1_zero_dispense == 1)
t = tr[(tr.date_Dam >= '2023-09-01') & ~rule & ~(tr['dev_Dam__Y(Stage2)'] < -0.05)].copy()
t['small'] = (t.ac_batch <= 3).astype(int); t['fast'] = (t['Dam__Stage2 Line1 Distance Speed'] >= 6500).astype(int)
t['uv45'] = (t['Fill2__CURE SPEED'] == 45).astype(int); t['m501'] = t.model.str.endswith('501').astype(int)
COMP = {'Dam 속도 ≥6500': 'fast', 'UV 속도 45': 'uv45', '탈포 배치 1-3': 'small', '주력 모델 A': 'm501',
        'Dam 택트(초)': 'Dam__Machine Tact time', 'Fill1 택트(초)': 'Fill1__Machine Tact time', '챔버 온도': 'AutoClave__Chamber Temp.',
        'Fill1 Stage2 토출량': 'Fill1__Dispense Volume(Stage2)', 'Dam 좌표편차 합': 'fe_Dam_coord_dev_abs', 'Fill1 좌표편차 합': 'fe_Fill1_coord_dev_abs'}
out = {}; fig, ax = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
for i, ln in enumerate([1, 2]):
    s = t[t.line == ln]; p = s.y.mean()
    g = s.groupby('day').agg(n=('y', 'size'), k=('y', 'sum')); g = g[g.n >= 30]; g['p'] = g.k / g.n
    g['ucl'] = p + 3 * np.sqrt(p * (1 - p) / g.n); g['out'] = g.p > g.ucl
    od = g[g.out]
    a = ax[i]; a.plot(g.index, g.p * 100, 'o-', ms=3, lw=.8, color='#2c3e50'); a.plot(g.index, g.ucl * 100, color='#c0392b', lw=.8, ls='--', label='UCL (3σ)')
    a.axhline(p * 100, color='grey', lw=.8); a.scatter(od.index, od.p * 100, color='#c0392b', zorder=3, s=30, label='out of control')
    a.set_ylabel('defect %'); a.set_title(f'Line #{ln} daily p-chart (2023-09+, known causes removed) — {len(od)} days out of {len(g)}', fontsize=10); a.legend(fontsize=8)
    # 이상일 vs 정상일 비교
    s = s.assign(out=s.day.isin(od.index))
    cmp_ = {k: dict(out=round(float(s[s.out][c].mean()), 3), normal=round(float(s[~s.out][c].mean()), 3)) for k, c in COMP.items()}
    cmp_['작업지시 수/일'] = dict(out=round(float(s[s.out].groupby('day').workorder.nunique().mean()), 2), normal=round(float(s[~s.out].groupby('day').workorder.nunique().mean()), 2))
    # 이상일 불량 중 설명되는 몫: 작은 배치·UV45·고속
    ex = s[s.out]; expl = ((ex.small == 1) | (ex.uv45 == 1) | (ex.fast == 1))
    out[f'line{ln}'] = dict(center=round(p * 100, 2), days=len(g), out_days=[str(x.date()) for x in od.index],
                            out_days_detail=od.assign(p=lambda x: (x.p * 100).round(1), ucl=lambda x: (x.ucl * 100).round(1)).reset_index().astype({'day': str}).to_dict('records'),
                            out_day_defects=int(ex.y.sum()), out_day_defects_with_known_condition=int(ex.y[expl].sum()), compare=cmp_,
                            out_day_workorders=ex.groupby('workorder').y.agg(['size', 'sum']).sort_values('sum', ascending=False).head(8).reset_index().to_dict('records'))
plt.tight_layout(); plt.savefig('figs/25_line_spc.png', dpi=110, bbox_inches='tight')
# 라인 차이를 설명하는가: 라인 효과를 조건 통제 전후로
import statsmodels.formula.api as smf
t['m'] = t.date_Dam.dt.to_period('M').astype(str); t['hr'] = t.date_Dam.dt.hour
for nm, f in [('line_only', 'y ~ C(line) + C(m) + C(model) + C(hr)'), ('line+conditions', 'y ~ C(line) + small + fast + uv45 + C(m) + C(model) + C(hr)')]:
    r = smf.logit(f, t).fit(disp=0, maxiter=300); k = 'C(line)[T.2]'
    out[nm] = dict(OR_line2=round(float(np.exp(r.params[k])), 2), p=round(float(r.pvalues[k]), 4))
json.dump(out, open('reports/25_line_spc.json', 'w'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
