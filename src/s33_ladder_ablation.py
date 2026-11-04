"""33단계: 32단계 '분석 기반 변수' 상승분을 묶음별로 쪼개기 (2단계 + 묶음 하나씩 / 3단계 − 묶음 하나씩)"""
import json, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
src = open('src/s32_feature_ladder.py', encoding='utf-8').read()
exec(src.split('MONTHS = ')[0])
MONTHS = ['2023-12', '2024-01', '2024-02', '2024-03', '2024-04']
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
GROUPS = {'대기 시간 · 규칙 (공정 간 대기, 2시간 초과, 라인 변경, Fill1 미도포)': ['fe_lead_Dam_Fill1', 'fe_lead_Fill1_Fill2', 'fe_lead_Fill2_AC', 'fe_lead_total', 'fe_long_wait', 'equip_mismatch', 'fe_Fill1_zero_dispense'],
          '휴식 · 시각 (휴식 겹침, 휴식 외 대기, 공정 시각)': RTF + ['fe_hour', 'fe_weekday'],
          '공정 조건 (저온 챔버, UV 과조사, 고속 토출)': A_F,
          '운영 (택트 차이, 시험 생산)': ['fe_tact_Dam_Fill1_diff', 'fe_pxx_order']}
base2 = COMMON + MEAS + PHYS; y = tr.y.values
def ev(fs):
    oof = np.zeros(len(tr))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(tr, y): oof[b] = fit_pred2(tr.iloc[a], tr.iloc[b], fs, True, SEED)
    lifts = []
    for mo in MONTHS:
        A, B = tr[tr.mo < mo], tr[tr.mo == mo]; p = fit_pred2(A, B, fs, True, SEED); lifts.append(average_precision_score(B.y, p))
    return round(average_precision_score(y, oof), 4), round(float(np.mean(lifts)), 4)
out = {}
for g, cols in GROUPS.items():
    out[g] = dict(add_to_stage2=ev(base2 + cols), remove_from_stage3=ev(base2 + [c for c in ANAL if c not in cols]))
    print(g, out[g], flush=True)
json.dump(out, open('reports/33_ladder_ablation.json', 'w'), ensure_ascii=False, indent=1)
