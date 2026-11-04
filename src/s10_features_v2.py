"""10단계 v2 피처: 4·7단계 분석에서 나온 가설을 피처로 구현
- 흐름 단절: 설비별 직전/다음 제품 간격, 동시 기록 수, 탈포 배치 구성
- 대기 세분화: Fill2→탈포 대기 구간 플래그
- 정렬: 스테이지 간 좌표차, 공정 간 좌표 편차 차, Dam Y(Stage2) 음수 편차
- 로트 내 이상: 작업지시 중앙값 대비 편차
- 실시간 가능 작업지시 진행 정보: 착수 후 경과 시간·누적 수량
"""
import pandas as pd, numpy as np, json

d = pd.read_parquet('data/features.parquet')
F = {}
def add(n, s, desc): d[n] = s; F[n] = desc

# ── 1) 흐름 단절 (같은 설비 기준, 시간순) ──
for p, eq in [('Dam', 'equip_Dam'), ('Fill1', 'equip_Fill1'), ('Fill2', 'equip_Fill2')]:
    o = d.sort_values([eq, f'date_{p}'])
    g = o.groupby(eq)[f'date_{p}']
    add(f'v2_{p}_gap_prev', (o[f'date_{p}'] - g.shift(1)).dt.total_seconds().div(60).reindex(d.index), f'{p} 설비 직전 제품과 간격(분)')
    add(f'v2_{p}_gap_next', (g.shift(-1) - o[f'date_{p}']).dt.total_seconds().div(60).reindex(d.index), f'{p} 설비 다음 제품과 간격(분)')
    add(f'v2_{p}_same_ts', d.groupby([eq, f'date_{p}'])[f'date_{p}'].transform('size'), f'{p} 같은 시각·설비 동시 기록 수')
o = d.sort_values('date_AutoClave')
add('v2_AC_gap_prev', o.date_AutoClave.diff().dt.total_seconds().div(60).reindex(d.index), '탈포 직전 투입과 간격(분)')
add('v2_restart_flag', (d.v2_Dam_gap_prev > 60).astype(int), 'Dam 60분 이상 정지 후 재가동')

# ── 2) 탈포 배치: 같은 AutoClave 시각에 들어간 제품 묶음 ──
bk = d.groupby('date_AutoClave')
add('v2_AC_batch_size', bk['Set ID'].transform('size'), '탈포 배치 크기')
add('v2_AC_batch_wait_max', bk['fe_lead_Fill2_AC'].transform('max'), '배치 내 최대 Fill2→탈포 대기')
add('v2_AC_batch_wait_gap', d.fe_lead_Fill2_AC - bk['fe_lead_Fill2_AC'].transform('min'), '배치 내 최소 대기 대비 내 대기 초과분')
add('v2_AC_batch_n_wo', bk['workorder'].transform('nunique'), '배치 내 작업지시 수(혼합 투입)')
add('v2_AC_batch_n_line', bk['line'].transform('nunique'), '배치 내 라인 수')

# ── 3) 대기 세분화 ──
add('v2_wait_F2AC_ge40', (d.fe_lead_Fill2_AC >= 40).astype(int), 'Fill2→탈포 40분 이상')
add('v2_wait_F2AC_ge60', (d.fe_lead_Fill2_AC >= 60).astype(int), 'Fill2→탈포 60분 이상')
add('v2_wait_D_F1_ge20', (d.fe_lead_Dam_Fill1 >= 20).astype(int), 'Dam→Fill1 20분 이상')

# ── 4) 정렬: 스테이지 간 좌표차, 공정 간 차 ──
C = lambda p, ax, s: d[f'{p}__HEAD NORMAL COORDINATE {ax} AXIS(Stage{s})']
for p in ['Dam', 'Fill1']:
    add(f'v2_{p}_Y2_Y1', C(p, 'Y', 2) - C(p, 'Y', 1), f'{p} Y Stage2-1')
    add(f'v2_{p}_Y2_Y3', C(p, 'Y', 2) - C(p, 'Y', 3), f'{p} Y Stage2-3')
    add(f'v2_{p}_X1_X2', C(p, 'X', 1) - C(p, 'X', 2), f'{p} X Stage1-2')
    add(f'v2_{p}_X2_X3', C(p, 'X', 2) - C(p, 'X', 3), f'{p} X Stage2-3')
add('v2_DamY2_neg', (d['dev_Dam__Y(Stage2)'] < -0.05).astype(int), 'Dam Y(Stage2) 음수 편차')
add('v2_DamY2_minus_Fill1Y2', d['dev_Dam__Y(Stage2)'] - d['dev_Fill1__Y(Stage2)'], 'Dam·Fill1 Y(Stage2) 편차 차 (공정 간 정렬)')
add('v2_DamX2_minus_Fill1X2', d['dev_Dam__X(Stage2)'] - d['dev_Fill1__X(Stage2)'], 'Dam·Fill1 X(Stage2) 편차 차')
add('v2_coord_dev_total', d[[c for c in d.columns if c.startswith('dev_')]].abs().sum(1), '전 공정 좌표 편차 절대합')

# ── 5) 조건 플래그 (4단계 임계값) ──
add('v2_dam_speed_hi', (d.fe_Dam_speed_mean > 6000).astype(int), 'Dam 이동속도 > 6000')
add('v2_dam_s2_short', (d['Dam__DISCHARGED TIME OF RESIN(Stage2)'] < 6).astype(int), 'Dam Stage2 토출 < 6초')
add('v2_ac_temp_low', (d['AutoClave__Chamber Temp.'] <= 46).astype(int), '챔버 온도 ≤ 46℃')
add('v2_model_C', d.model.str.endswith('503').astype(int), '모델 C')
add('v2_pxx_order', d.workorder.str.contains('PXX').astype(int), '시험 생산 작업지시')

# ── 6) 로트 내 이상: 작업지시 중앙값 대비 편차 ──
for c, n in [('fe_Dam_speed_mean', 'dam_speed'), ('Dam__DISCHARGED TIME OF RESIN(Stage2)', 'dam_s2_time'), ('fe_Dam_vol_total', 'dam_vol'),
             ('fe_Fill1_vol_total', 'fill1_vol'), ('AutoClave__Chamber Temp.', 'ac_temp'), ('dev_Dam__Y(Stage2)', 'damY2'),
             ('Dam__Machine Tact time', 'dam_tact'), ('fe_lead_Fill2_AC', 'wait_f2ac')]:
    add(f'v2_wo_dev_{n}', d[c] - d.groupby('workorder')[c].transform('median'), f'{c} — 작업지시 중앙값 대비')
add('v2_wo_nuniq_dam_speed', d.groupby('workorder').fe_Dam_speed_mean.transform('nunique'), '작업지시 내 Dam 속도 설정 변경 횟수')

# ── 7) 실시간 가능 작업지시 진행 정보 (과거만 사용) ──
o = d.sort_values(['workorder', 'date_Dam'])
add('v2_wo_cum_n', o.groupby('workorder').cumcount().reindex(d.index), '작업지시 착수 후 누적 투입 수')
add('v2_wo_elapsed_h', (o.date_Dam - o.groupby('workorder').date_Dam.transform('min')).dt.total_seconds().div(3600).reindex(d.index), '작업지시 착수 후 경과(시간)')
add('v2_wo_early', (d.v2_wo_cum_n < 15).astype(int), '작업지시 착수 후 15개 이내')

d = d.replace([np.inf, -np.inf], np.nan)
d.to_parquet('data/features_v2.parquet', index=False)
json.dump(F, open('reports/10_feature_dict_v2.json', 'w'), ensure_ascii=False, indent=1)
t = d[(d.is_train == 1) & (d.equip_mismatch == 0) & (d.fe_lead_total <= 2)]
for c in ['v2_AC_batch_size', 'v2_AC_batch_n_wo', 'v2_AC_batch_wait_gap', 'v2_wo_early', 'v2_restart_flag', 'v2_wait_F2AC_ge60']:
    q = pd.qcut(t[c], 5, duplicates='drop') if t[c].nunique() > 6 else t[c]
    print(c, t.groupby(q, observed=True).target.agg(['size', 'mean']).round(4).to_dict('index'))
print(d.shape, len(F))
