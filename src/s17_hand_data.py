"""17단계: 수기 측정 데이터(hand_data.xlsx, 2024-03-19 ~ 04-28, 휴식 시간마다 기록)를 제품에 연결해 불량과의 관계 확인
- 연결: 제품의 Fill2 시각이 기록 시간대(start ~ end) 안에 들면 그 값을 붙임. OCR gap은 제품 라인(#1/#2)에 맞춰 선택
- 대상: 규칙·Dam Y 음수 편차 제외 (16단계와 같은 기준)
- 같은 시간대 제품은 같은 값을 공유 → 로지스틱은 시간대 군집 표준오차
"""
import pandas as pd, numpy as np, sys, warnings
import statsmodels.api as sm
warnings.filterwarnings('ignore')

h = pd.read_excel(sys.argv[1] if len(sys.argv) > 1 else 'data/hand_data.xlsx')
h.columns = ['day', 'st', 'et', 'dam_th', 'fill_th', 'uv_mj', 'uv_mw', 'temp', 'hum', 'g11', 'g12', 'g13', 'g21', 'g22', 'g23']
h['date'] = pd.to_datetime(h.day.astype(str), format='%y%m%d')
h['s'] = h.date + pd.to_timedelta(h.st.astype(str)); h['e'] = h.date + pd.to_timedelta(h.et.astype(str))
h = h.dropna(subset=['uv_mj']).reset_index(drop=True); h['slot'] = h.index

d = pd.read_parquet('data/features.parquet'); t = d[d.is_train == 1].copy()
t = t[~((t.fe_lead_total > 2) | (t.equip_mismatch == 1) | (t.fe_Fill1_zero_dispense == 1)) & ~(t['dev_Dam__Y(Stage2)'] < -0.05)]
t['slot'] = -1
for i, r in h.iterrows(): t.loc[(t.date_Fill2 >= r.s) & (t.date_Fill2 <= r.e), 'slot'] = i
c = t[t.slot >= 0].merge(h, on='slot'); c['y'] = c.target.astype(int)
for k in (1, 2, 3): c[f'gap_s{k}'] = np.where(c.line == 1, c[f'g1{k}'], c[f'g2{k}'])
print(f'연결 {len(c)}건 · 불량 {c.y.sum()} ({c.y.mean():.2%}) · 시간대 {c.slot.nunique()}개')

rows = []
for v in ['uv_mj', 'uv_mw', 'temp', 'hum', 'dam_th', 'fill_th', 'gap_s1', 'gap_s2', 'gap_s3']:
    X = sm.add_constant(pd.DataFrame({'z': (c[v] - c[v].mean()) / c[v].std(), 'line2': (c.line == 2).astype(float)}))
    r = sm.Logit(c.y, X).fit(disp=0, cov_type='cluster', cov_kwds={'groups': c.slot}); ci = np.exp(r.conf_int().loc['z'])
    rows.append(dict(var=v, OR_per_SD=np.exp(r.params.z), lo=ci[0], hi=ci[1], p_cluster=r.pvalues.z))
R = pd.DataFrame(rows).round(3); R.to_csv('reports/17_hand_data_logit.csv', index=False); print(R.to_string(index=False))

for v, b in [('uv_mj', [0, 1600, 1640, 1660, 1700, 1800, 3000]), ('fill_th', [0, 200, 220, 235, 400]), ('temp', [0, 23.5, 24, 24.5, 30])]:
    g = c.groupby(pd.cut(c[v], b), observed=True).y.agg(['size', 'sum', 'mean']); g['mean'] = (g['mean'] * 100).round(2); print(v); print(g.to_string())
print('UV 에너지 ↔ Fill2 경화 속도'); print(c.groupby('Fill2__CURE SPEED').agg(n=('y', 'size'), defect=('y', 'mean'), uv_mj=('uv_mj', 'median')).round(4).to_string())
