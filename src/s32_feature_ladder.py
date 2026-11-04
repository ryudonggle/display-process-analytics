"""32단계: 성능 사다리 — 분석이 만든 변수가 예측을 얼마나 올렸나 (누수 없는 기준으로 재측정)
단계
 0. 원본 그대로      : train.csv 측정값(형식 변경으로 59% 행이 한 칸 밀린 상태 그대로)
 1. 데이터 복원      : 밀림을 바로잡은 측정값
 2. + 공정 물리 변수 : 토출률·토출량 비율·경화 시간·탈포 압력×시간·좌표 어긋남(학습 구간 중앙값 기준)
 3. + 분석 기반 변수 : 공정 간 대기·라인 변경·Fill1 미도포(규칙), 휴식 겹침·휴식 외 대기, 저온·UV 과조사·고속 조건, 시험 생산
 (참고) 4. + 현장 실시간 로그: 탈포 배치 크기 등 — test 한 행만으로는 못 만듦
원칙: 모든 단계에 같은 모델(LightGBM)·같은 파라미터. 라벨·미래 정보·경과일·작업지시 그룹 통계 미사용(20단계 원칙)
평가: ① 같은 기간 5-fold (대회 방식) ② 월별 롤링 2023-12 ~ 2024-04 (현장 방식, 시드 2개)
결과: reports/32_feature_ladder.json, figs/grad_B0_ladder.png
"""
import pandas as pd, numpy as np, json, warnings
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
warnings.filterwarnings('ignore'); SEED = 42
exec(open('src/s27_predict_from_findings.py', encoding='utf-8').read().split('LOOKAHEAD = ')[0])   # d(+rt_*, fA_*, fB_*), RTF, dev_fit

raw = pd.read_csv('data/train.csv', low_memory=False)
SKIP = ('Production Qty', 'PalletID', 'Receip No', 'Collect Date', 'Insp. Seq No.')
rawnum = {}
for c in raw.columns:
    if c == 'target' or any(k in c for k in SKIP): continue
    v = pd.to_numeric(raw[c], errors='coerce')
    if v.notna().any() and v.nunique() > 1: rawnum['raw_' + c] = v.values
RAW = list(rawnum)

for c in ['model', 'equip_Dam', 'equip_Fill1', 'equip_Fill2']: d[c + '_code'] = d[c].astype('category').cat.codes
tr = d[d.is_train == 1].reset_index(drop=True); tr = pd.concat([tr, pd.DataFrame(rawnum)], axis=1)
tr['y'] = tr.target.astype(int); tr['mo'] = tr.date_Dam.dt.to_period('M').astype(str)
COMMON = ['model_code', 'equip_Dam_code', 'equip_Fill1_code', 'equip_Fill2_code']
MEAS = [c for c in tr.columns if c.startswith(('Dam__', 'Fill1__', 'Fill2__', 'AutoClave__')) and not any(k in c for k in SKIP)
        and pd.api.types.is_numeric_dtype(tr[c]) and tr[c].nunique() > 1]
PHYS = ['fe_Dam_vol_total', 'fe_Dam_time_total', 'fe_Dam_flow_rate', 'fe_Dam_vol_s2_ratio', 'fe_Dam_vol_s1_s3_diff', 'fe_Fill1_vol_total',
        'fe_Fill1_time_total', 'fe_Fill1_flow_rate', 'fe_Fill1_vol_s2_ratio', 'fe_Fill1_vol_s1_s3_diff', 'fe_Dam_Fill1_vol_ratio', 'fe_Dam_cure_dist',
        'fe_Dam_cure_time', 'fe_Fill2_cure_dist', 'fe_Fill2_cure_time', 'fe_cure_time_total', 'fe_Dam_speed_mean', 'fe_Dam_speed_std',
        'fe_Dam_vol_per_speed', 'fe_Dam_thick_abs', 'fe_AC_1st_PxT', 'fe_AC_2nd_PxT', 'fe_AC_3rd_PxT', 'fe_AC_PxT_total', 'fe_AC_time_total',
        'fe_AC_temp_x_time', 'fe_AC_P3_minus_P1', 'fe_Dam_X_s1_s3', 'fe_Dam_Y_s1_s3', 'fe_Fill1_X_s1_s3', 'fe_Fill1_Y_s1_s3', 'fe_Fill1_Z_s1_s3']
ANAL = ['fe_lead_Dam_Fill1', 'fe_lead_Fill1_Fill2', 'fe_lead_Fill2_AC', 'fe_lead_total', 'fe_long_wait', 'equip_mismatch', 'fe_Fill1_zero_dispense',
        'fe_tact_Dam_Fill1_diff', 'fe_hour', 'fe_weekday', 'fe_pxx_order'] + RTF + A_F
STAGES = [('0. 원본 그대로', COMMON + RAW, False), ('1. 데이터 복원', COMMON + MEAS, False),
          ('2. + 공정 물리 변수', COMMON + MEAS + PHYS, True), ('3. + 분석 기반 변수', COMMON + MEAS + PHYS + ANAL, True),
          ('(참고) + 현장 실시간 로그', COMMON + MEAS + PHYS + ANAL + B_F, True)]
P_ = dict(n_estimators=600, learning_rate=0.02, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
          colsample_bytree=0.5, reg_lambda=5, verbose=-1, random_state=SEED)


def fit_pred2(A, B, fs, dev, seed):
    XA, XB = A[fs], B[fs]
    if dev: XA, XB = pd.concat([XA, dev_fit(A, A)], axis=1), pd.concat([XB, dev_fit(A, B)], axis=1)
    return lgb.LGBMClassifier(**(P_ | dict(random_state=seed))).fit(XA, A.y).predict_proba(XB)[:, 1]


MONTHS = ['2023-12', '2024-01', '2024-02', '2024-03', '2024-04']
y = tr.y.values; res = []
for name, fs, dev in STAGES:
    oof = np.zeros(len(tr))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(tr, y): oof[b] = fit_pred2(tr.iloc[a], tr.iloc[b], fs, dev, SEED)
    k1 = np.argsort(-oof)[:int(len(tr) * .01)]; k3 = np.argsort(-oof)[:int(len(tr) * .03)]
    roll = []
    for mo in MONTHS:
        A, B = tr[tr.mo < mo], tr[tr.mo == mo]
        p = np.mean([fit_pred2(A, B, fs, dev, s) for s in (SEED, 7)], 0)
        k = np.argsort(-p)[:max(1, int(len(B) * .03))]
        roll.append(dict(month=mo, pr_auc=average_precision_score(B.y, p), lift=average_precision_score(B.y, p) / B.y.mean(),
                         top3_precision=float(B.y.values[k].mean())))
    R = pd.DataFrame(roll)
    r = dict(stage=name, n_features=len(fs) + (0 if not dev else 1), cv_pr_auc=round(average_precision_score(y, oof), 4),
             cv_top1_precision=round(float(y[k1].mean()), 3), cv_top3_precision=round(float(y[k3].mean()), 3),
             roll_pr_auc=round(R.pr_auc.mean(), 4), roll_lift=round(R.lift.mean(), 2), roll_top3_precision=round(R.top3_precision.mean(), 3))
    res.append(r); print(r, flush=True)
json.dump(dict(base_rate=round(float(y.mean()), 4), stages=res), open('reports/32_feature_ladder.json', 'w'), ensure_ascii=False, indent=1)
