"""3단계 피처 엔지니어링: 공정 물리량 기반 파생 + 좌표 편차 + 시간/작업지시 컨텍스트"""
import pandas as pd, numpy as np, json

d = pd.read_parquet('data/clean.parquet')
F = {}  # 새 피처 → 설명
def add(name, s, desc):
    d[name] = s; F[name] = desc

# ── 1) 레진 도포(Dam·Fill1): 총량, 시간, 실제 토출률 ──
for p in ['Dam', 'Fill1']:
    vol = d[[f'{p}__Dispense Volume(Stage{i})' for i in (1, 2, 3)]]
    tim = d[[f'{p}__DISCHARGED TIME OF RESIN(Stage{i})' for i in (1, 2, 3)]]
    add(f'fe_{p}_vol_total', vol.sum(1), f'{p} 총 도포량')
    add(f'fe_{p}_time_total', tim.sum(1), f'{p} 총 토출 시간')
    add(f'fe_{p}_flow_rate', vol.sum(1) / tim.sum(1).replace(0, np.nan), f'{p} 평균 토출률(양/시간)')
    add(f'fe_{p}_vol_s2_ratio', vol.iloc[:, 1] / vol.sum(1).replace(0, np.nan), f'{p} Stage2 도포 비중')
    add(f'fe_{p}_vol_s1_s3_diff', vol.iloc[:, 0] - vol.iloc[:, 2], f'{p} Stage1-3 도포량 비대칭')
add('fe_Fill1_zero_dispense', (d['fe_Fill1_vol_total'] == 0).astype(int), 'Fill1 도포 0 (미도포)')
add('fe_Dam_Fill1_vol_ratio', d['fe_Dam_vol_total'] / d['fe_Fill1_vol_total'].replace(0, np.nan), 'Dam(가벽)/Fill1(내측) 도포량 비')

# ── 2) UV 경화(Dam·Fill2): 경화 이동거리, 경화 시간(=거리/속도) → UV 조사량 대리변수 ──
for p in ['Dam', 'Fill2']:
    dist = (d[f'{p}__CURE START POSITION X'] - d[f'{p}__CURE END POSITION X']).abs()
    add(f'fe_{p}_cure_dist', dist, f'{p} UV 경화 이동거리')
    add(f'fe_{p}_cure_time', dist / d[f'{p}__CURE SPEED'], f'{p} UV 경화 시간(거리/속도) — UV 조사량 대리변수')
add('fe_cure_time_total', d['fe_Dam_cure_time'] + d['fe_Fill2_cure_time'], 'UV 경화 시간 합')

# ── 3) Dam 헤드 이동속도 요약 ──
sp = d[[c for c in d.columns if c.startswith('Dam__Stage') and 'Distance Speed' in c]]
add('fe_Dam_speed_mean', sp.mean(1), 'Dam 헤드 이동속도 평균')
add('fe_Dam_speed_std', sp.std(1), 'Dam 헤드 이동속도 구간 편차')
add('fe_Dam_vol_per_speed', d['fe_Dam_vol_total'] / sp.mean(1) * 1e4, 'Dam 도포량/이동속도 (선당 도포 두께 대리)')
add('fe_Dam_thick_abs', d[['Dam__THICKNESS 1', 'Dam__THICKNESS 2', 'Dam__THICKNESS 3']].abs().sum(1), 'Dam 두께 편차 절대합')

# ── 4) 탈포(AutoClave): 압력×시간, 총 시간 ──
pt = 0
for k in ('1st', '2nd', '3rd'):
    v = d[f'AutoClave__{k} Pressure'] * d[f'AutoClave__{k} Pressure Unit Time']
    add(f'fe_AC_{k}_PxT', v, f'탈포 {k} 압력×시간'); pt = pt + v
add('fe_AC_PxT_total', pt, '탈포 압력×시간 총합 (탈포 에너지)')
add('fe_AC_time_total', d[[f'AutoClave__{k} Pressure Unit Time' for k in ('1st', '2nd', '3rd')]].sum(1), '탈포 총 시간')
add('fe_AC_temp_x_time', d['AutoClave__Chamber Temp.'] * d['AutoClave__Chamber Temp. Unit Time'], '챔버 온도×시간')
add('fe_AC_P3_minus_P1', d['AutoClave__3rd Pressure'] - d['AutoClave__1st Pressure'], '탈포 압력 상승폭')

# ── 5) 좌표: 라인·모델 기준 편차 (정렬 불량 Misalignment 관련) + 스테이지 간 차이 ──
coord = [c for c in d.columns if 'HEAD NORMAL COORDINATE' in c]
grp = d['line'].astype(str) + '_' + d['model']
for c in coord:
    med = d.groupby(grp)[c].transform('median')
    add('dev_' + c.replace('HEAD NORMAL COORDINATE ', '').replace(' AXIS', ''), d[c] - med, f'{c} — 라인·모델 중앙값 대비 편차')
dev = [c for c in d.columns if c.startswith('dev_')]
for p in ['Dam', 'Fill1', 'Fill2']:
    cols = [c for c in dev if c.startswith(f'dev_{p}__')]
    add(f'fe_{p}_coord_dev_abs', d[cols].abs().sum(1), f'{p} 좌표 편차 절대합')
for p in ['Dam', 'Fill1']:
    for ax in 'XYZ':
        a = d[f'{p}__HEAD NORMAL COORDINATE {ax} AXIS(Stage1)']; b = d.get(f'{p}__HEAD NORMAL COORDINATE {ax} AXIS(Stage3)')
        if b is not None: add(f'fe_{p}_{ax}_s1_s3', a - b, f'{p} {ax}축 Stage1-Stage3 위치차')

# ── 6) 시간 컨텍스트 ──
add('fe_lead_Dam_Fill1', (d.date_Fill1 - d.date_Dam).dt.total_seconds() / 60, 'Dam→Fill1 대기(분)')
add('fe_lead_Fill1_Fill2', (d.date_Fill2 - d.date_Fill1).dt.total_seconds() / 60, 'Fill1→Fill2 대기(분)')
add('fe_lead_Fill2_AC', (d.date_AutoClave - d.date_Fill2).dt.total_seconds() / 60, 'Fill2→AutoClave 대기(분)')
add('fe_lead_total', (d.date_AutoClave - d.date_Dam).dt.total_seconds() / 3600, 'Dam→AutoClave 총 리드타임(시간)')
add('fe_hour', d.date_Dam.dt.hour, '작업 시각')
add('fe_weekday', d.date_Dam.dt.weekday, '요일')
add('fe_days', (d.date_Dam - d.date_Dam.min()).dt.days, '기간 경과일 (추세 통제용)')
add('fe_recipe_new', (d.log_format != 'A').astype(int), '2023-10 이후 신규 레시피')

# ── 7) 작업지시(Workorder) 컨텍스트 (타깃 미사용) ──
add('fe_wo_size', d.groupby('workorder')['Set ID'].transform('size'), '작업지시 규모(제품 수)')
add('fe_wo_rank', d.groupby('workorder')['date_Dam'].rank(pct=True), '작업지시 내 투입 순서(0~1)')
add('fe_wo_prefix', d.workorder.str[:4], '작업지시 접두어')
add('fe_wo_suffix', d.workorder.str.split('-').str[-1].astype(str), '작업지시 차수(-1,-2..)')
add('fe_wo_span_h', d.groupby('workorder')['date_Dam'].transform(lambda s: (s.max() - s.min()).total_seconds() / 3600), '작업지시 생산 기간(시간)')
add('fe_qty_ratio', d['Dam__Production Qty'] / d['fe_wo_size'], '작업지시 내 생산 진행률')
add('fe_tact_Dam_Fill1_diff', d['Dam__Machine Tact time'] - d['Fill1__Machine Tact time'], 'Dam-Fill1 택트타임 차')

d = d.replace([np.inf, -np.inf], np.nan)
d.to_parquet('data/features.parquet', index=False)
json.dump(F, open('reports/03_feature_dict.json', 'w'), ensure_ascii=False, indent=1)

# 단변량 신호 점검: 피처별 불량률 AUC (train)
from sklearn.metrics import roc_auc_score
t = d[d.is_train == 1]
rows = []
for c in F:
    x = t[c]
    if x.dtype == object or x.dtype.name in ('str', 'string') or x.nunique() < 2: continue
    x = x.fillna(x.median()); a = roc_auc_score(t.target, x)
    rows.append((c, round(max(a, 1 - a), 4), '+' if a >= .5 else '-', F[c]))
res = pd.DataFrame(rows, columns=['feature', 'auc', 'dir', 'desc']).sort_values('auc', ascending=False)
res.to_csv('reports/03_feature_auc.csv', index=False)
print(d.shape, 'new features:', len(F))
print(res.head(30).to_string(index=False))
