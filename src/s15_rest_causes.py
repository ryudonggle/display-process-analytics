"""15단계: 규칙(132건) 밖 '나머지 불량'의 원인 재검증
07단계 원인 후보를 플래그로 정의 → 규칙 제외 40,374건에서
  ① 레시피별 불량률·상대위험·p (구/신 각각)  ② 월 고정효과 + 모델 + 라인 통제 로지스틱 오즈비
  ③ 초과 불량 건수(해당 구간 불량 − 기준 불량률로 기대되는 불량)
판정: 두 레시피 모두 같은 방향·유의 + 월 통제 후 유의 → '확정', 한 레시피만 → '구 레시피 한정' 등
"""
import pandas as pd, numpy as np, json, warnings
import statsmodels.api as sm
from scipy.stats import fisher_exact
warnings.filterwarnings('ignore')

d = pd.read_parquet('data/features_v2.parquet'); d = d[d.is_train == 1].copy()
d['rule'] = ((d.fe_lead_total > 2) | (d.equip_mismatch == 1) | (d.fe_Fill1_zero_dispense == 1)).astype(int)
t = d[d.rule == 0].copy(); y = t.target.astype(int)
t['recipe'] = np.where(t.fe_recipe_new == 1, 'new', 'old'); t['month'] = t.date_Dam.dt.to_period('M').astype(str)

q = lambda c, p: t[c].quantile(p)
FLAGS = {
    'Dam Y(Stage2) 음수 편차 < -0.05':      t['dev_Dam__Y(Stage2)'] < -0.05,
    'Dam 이동속도 > 6,000':                 t.fe_Dam_speed_mean > 6000,
    'Dam Stage2 토출 < 6초':                t['Dam__DISCHARGED TIME OF RESIN(Stage2)'] < 6,
    'Dam 택트 ≥ 75초':                      t['Dam__Machine Tact time'] >= 75,
    'Dam Head Zero Y > 300':                t['Dam__Head Zero Position Y'] > 300,
    'Fill1 X(Stage2) 편차 |.| > 0.05':      t['dev_Fill1__X(Stage2)'].abs() > 0.05,
    'Fill1 Y(Stage3) 편차 |.| > 0.05':      t['dev_Fill1__Y(Stage3)'].abs() > 0.05,
    'Fill1 Stage2 토출량 상위 20%':          t['Fill1__Dispense Volume(Stage2)'] > q('Fill1__Dispense Volume(Stage2)', .8),
    '탈포 챔버 ≤ 46℃':                      t['AutoClave__Chamber Temp.'] <= 46,
    '탈포 총시간 하위 20%':                  t.fe_AC_time_total < q('fe_AC_time_total', .2),
    '작업지시 생산 기간 > 24시간':           t.fe_wo_span_h > 24,
    '작업지시 60개 미만 (소량)':             t.fe_wo_size < 60,
    '작업지시 착수 첫 10%':                  t.fe_wo_rank <= 0.10,
    '작업지시 착수 후 15개 이내 (실시간)':    t.v2_wo_early == 1,
    'Dam 60분+ 정지 후 재가동':              t.v2_restart_flag == 1,
    '시험 생산 (PXX)':                       t.v2_pxx_order == 1,
    '모델 C':                              t.v2_model_C == 1,
}
base = pd.get_dummies(t[['month', 'model']], drop_first=True, dtype=float)
base['line2'] = (t.line == 2).astype(float); base = sm.add_constant(base)
p0 = y.mean(); rows = []
for name, f in FLAGS.items():
    f = f.astype(bool); r = dict(cause=name, n=int(f.sum()), rate_on=y[f].mean(), rate_off=y[~f].mean())
    for rc in ('old', 'new'):
        m = (t.recipe == rc).values; a, b = f.values & m, ~f.values & m
        r[f'n_{rc}'] = int(a.sum())
        if a.sum() < 30: r[f'rr_{rc}'] = r[f'p_{rc}'] = np.nan; continue
        r[f'on_{rc}'], r[f'off_{rc}'] = y[a].mean(), y[b].mean(); r[f'rr_{rc}'] = r[f'on_{rc}'] / r[f'off_{rc}']
        r[f'p_{rc}'] = fisher_exact([[y[a].sum(), a.sum() - y[a].sum()], [y[b].sum(), b.sum() - y[b].sum()]])[1]
    X = base.copy(); X['flag'] = f.astype(float).values
    try:
        mm = sm.Logit(y, X).fit(disp=0, maxiter=200); r['or_adj'] = np.exp(mm.params['flag']); r['p_adj'] = mm.pvalues['flag']
    except Exception:
        r['or_adj'] = r['p_adj'] = np.nan
    r['excess'] = y[f].sum() - f.sum() * y[~f].mean()   # 기준 불량률 대비 초과 불량
    rows.append(r)
R = pd.DataFrame(rows)

def verdict(r):
    sig = lambda rc: r[f'p_{rc}'] < .01 and r[f'rr_{rc}'] > 1
    adj = r.p_adj < .01 and r.or_adj > 1
    if sig('old') and sig('new') and adj: return '확정 (두 레시피·월 통제)'
    if adj and (sig('old') or sig('new')): return '구 레시피 한정' if sig('old') else '신 레시피 한정'
    if adj: return '약함 (레시피별 비유의)'
    return '기각'
R['verdict'] = R.apply(verdict, axis=1)
R = R.sort_values(['verdict', 'excess'], ascending=[True, False])
R.to_csv('reports/15_rest_causes.csv', index=False)

# 확정 원인이 설명하는 범위: 하나라도 해당하는 제품의 비중·불량 비중
conf = R[R.verdict.str.startswith('확정')].cause.tolist()
anyf = np.column_stack([FLAGS[c].values for c in conf]).any(1) if conf else np.zeros(len(t), bool)
cover = dict(confirmed=conf, share_rows=round(anyf.mean(), 4), share_defects=round(y[anyf].sum() / y.sum(), 4),
             rate_in=round(y[anyf].mean(), 4), rate_out=round(y[~anyf].mean(), 4), n_rest=len(t), defects_rest=int(y.sum()))
json.dump(cover, open('reports/15_rest_coverage.json', 'w'), ensure_ascii=False, indent=1)

pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 40)
fmt = R.assign(**{c: (R[c] * 100).round(1) for c in ['on_old', 'off_old', 'on_new', 'off_new']})
print(fmt[['cause', 'verdict', 'n_old', 'on_old', 'off_old', 'p_old', 'n_new', 'on_new', 'off_new', 'p_new', 'or_adj', 'p_adj', 'excess']]
      .round({'p_old': 4, 'p_new': 4, 'or_adj': 2, 'p_adj': 4, 'excess': 0}).to_string(index=False))
print(cover)
