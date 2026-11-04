"""22단계: 순번(생산 순서)과 정기 휴식의 영향 — 원인 분석용 (모델 피처 아님: 다른 제품 행 사용)
순번: Dam__Production Qty = 라인·일 단위 생산 카운터(하루 0부터, 99.9% 단조 증가) → 같은 10분 칸 안의 순서까지 복원
  - day_seq: 그날 그 라인의 몇 번째 제품 / wo_seq: 작업지시(라인별) 안에서 몇 번째 / wo_seq_rev: 끝에서 몇 번째
정기 휴식: 공정별 로그 공백으로 확인. Dam·Fill1·Fill2는 10:00–10:20 · 12:00–13:00 · 15:10–15:30 · 17:50–18:00에 함께 멈춤,
  탈포기는 약 30분 늦게(10:30 · 12:30–13:30 · 15:50) 빔 → 휴식 전 대기열을 처리하는 것으로 해석
  - 비정기 정지는 역인과(문제 → 정지) 가능성이 있어 이번 분석에서 쓰지 않음
  - 휴식이 걸린 공정 구간: Dam 전(휴식 직후 첫 투입) · Dam→Fill1 · Fill1→Fill2 · Fill2→탈포 · 안 걸림
  - 휴식 전후 순번: 같은 라인 Dam 투입 기준 휴식 직전 k번째 / 직후 k번째
대상: 2023-09 이후, 규칙(2시간·라인 변경·Fill1 미도포)·Dam Y 음수 편차 제외 (+ 전체 기간 재현)
결과: reports/22_sequence_breaks.json, figs/22_sequence_breaks.png
"""
import pandas as pd, numpy as np, json, warnings
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportion_confint
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
MIN = pd.Timedelta('1min')
BRK = [('10:00', '10:20'), ('12:00', '13:00'), ('15:10', '15:30'), ('17:50', '18:00')]   # Dam·Fill1·Fill2 공통 휴식

d = pd.read_parquet('data/features.parquet'); d['day'] = d.date_Dam.dt.normalize()
q = 'Dam__Production Qty'
d = d.sort_values(['line', 'day', 'date_Dam', q]).reset_index(drop=True)
d['day_seq'] = d.groupby(['line', 'day']).cumcount() + 1
d['wo_seq'] = d.sort_values(['date_Dam', q]).groupby(['workorder', 'line']).cumcount() + 1
d['wo_n'] = d.groupby(['workorder', 'line'])['Set ID'].transform('size'); d['wo_seq_rev'] = d.wo_n - d.wo_seq + 1

# 휴식이 걸린 구간
stage = pd.Series('none', index=d.index)
for a, b in BRK:
    bs, be = d.day + pd.Timedelta(a + ':00'), d.day + pd.Timedelta(b + ':00')
    for nm, x, y in [('Dam→Fill1', 'date_Dam', 'date_Fill1'), ('Fill1→Fill2', 'date_Fill1', 'date_Fill2'), ('Fill2→탈포', 'date_Fill2', 'date_AutoClave')]:
        hit = (d[x] < bs) & (d[y] >= be) & (stage == 'none')          # 휴식 시작 전에 x 완료, 휴식 끝난 뒤 y 완료 = 이 구간에서 휴식을 보냄
        stage[hit] = nm
d['brk_stage'] = stage
# 휴식 전후 순번 (같은 라인 Fill2 완료 순서): 휴식 시작 전 마지막 k번째(-k), 휴식 끝난 뒤 첫 k번째(+k)
d['brk_pos'] = np.nan
for (ln, dy), g in d.groupby(['line', 'day']):
    for a, b in BRK:
        bs, be = dy + pd.Timedelta(a + ':00'), dy + pd.Timedelta(b + ':00')
        g2 = g.sort_values(['date_Fill2', 'Fill2__Production Qty'])                  # 휴식 영향은 Fill2 이후 대기 → Fill2 완료 순서 기준
        pre = g2[g2.date_Fill2 < bs].tail(10); post = g2[g2.date_Fill2 >= be].head(10)
        if len(pre) == 0 or len(post) == 0: continue
        if (g2.date_Fill2.between(bs, be - MIN)).sum() > 3: continue  # 실제로 안 멈춘 날은 제외
        d.loc[pre.index, 'brk_pos'] = -np.arange(len(pre), 0, -1); d.loc[post.index, 'brk_pos'] = np.arange(1, len(post) + 1)

d['ac_batch'] = d.groupby('date_AutoClave')['Set ID'].transform('size')   # 같은 시각 탈포기 투입 제품 수(train+test)
tr = d[d.is_train == 1].copy(); tr['y'] = tr.target.astype(int); tr['m'] = tr.date_Dam.dt.to_period('M').astype(str)
tr['hr'] = tr.date_Dam.dt.hour
rule = (tr.fe_lead_total > 2) | (tr.equip_mismatch == 1) | (tr.fe_Fill1_zero_dispense == 1)
POPS = {'sep': tr[(tr.date_Dam >= '2023-09-01') & ~rule & ~(tr['dev_Dam__Y(Stage2)'] < -0.05)].copy(),
        'all': tr[~rule].copy()}


def rate(g):
    n, k = len(g), int(g.y.sum()); lo, hi = proportion_confint(k, n, method='wilson') if n else (np.nan, np.nan)
    return pd.Series(dict(n=n, defect=k, rate=round(k / n * 100, 2) if n else np.nan, lo=round(lo * 100, 2), hi=round(hi * 100, 2)))


out = {}
for name, t in POPS.items():
    o = {}
    o['wo_seq'] = t.groupby(pd.cut(t.wo_seq, [0, 1, 2, 3, 5, 10, 20, 50, 100, 1e6], labels=['1', '2', '3', '4-5', '6-10', '11-20', '21-50', '51-100', '>100']), observed=True).apply(rate).reset_index().to_dict('records')
    o['wo_seq_rev'] = t.groupby(pd.cut(t.wo_seq_rev, [0, 1, 3, 10, 1e6], labels=['마지막', '끝 2-3', '끝 4-10', '그 외']), observed=True).apply(rate).reset_index().to_dict('records')
    o['day_seq'] = t.groupby(pd.cut(t.day_seq, [0, 5, 10, 20, 50, 100, 200, 1e6], labels=['1-5', '6-10', '11-20', '21-50', '51-100', '101-200', '>200']), observed=True).apply(lambda g: pd.concat([rate(g), pd.Series(dict(hour_med=g.date_Dam.dt.hour.median()))])).reset_index().to_dict('records')
    o['brk_stage'] = t.groupby('brk_stage').apply(rate).reset_index().to_dict('records')
    o['brk_pos'] = t.dropna(subset=['brk_pos']).assign(p=lambda x: pd.cut(x.brk_pos, [-11, -6, -3, -2, -1, 1, 3, 6, 10], labels=['전 6-10', '전 3-5', '전 2', '직전 1', '직후 1', '후 2-3', '후 4-6', '후 7-10'])).groupby('p', observed=True).apply(lambda g: pd.concat([rate(g), pd.Series(dict(wait_med=g.fe_lead_Fill2_AC.median()))])).reset_index().to_dict('records')
    # 휴식 종류별(점심 vs 짧은 휴식) × 구간
    t['brk_kind'] = np.where(t.brk_stage == 'none', 'none', np.where(t.date_Fill2.dt.hour.isin([11, 12]), 'lunch', 'short'))
    o['brk_kind_stage'] = t.groupby(['brk_kind', 'brk_stage']).apply(rate).reset_index().to_dict('records')
    # 통제 로지스틱: 월 · 모델 · 라인 · 시각(시) 고정
    t['first_wo'] = (t.wo_seq <= 3).astype(int); t['post1'] = (t.brk_pos.between(1, 3)).astype(int)
    t['pre_brk'] = (t.brk_pos.between(-5, -1)).astype(int); t['day_first5'] = (t.day_seq <= 5).astype(int)
    f = smf.logit("y ~ pre_brk + post1 + first_wo + day_first5 + C(m) + C(model) + C(line) + C(hr)", t).fit(disp=0, maxiter=200)
    ks = [k for k in f.params.index if k in ('first_wo', 'post1', 'pre_brk', 'day_first5')]; ci = np.exp(f.conf_int())
    o['logit'] = {k.replace("C(brk_stage, Treatment('none'))[T.", '휴식@').rstrip(']'): dict(OR=round(float(np.exp(f.params[k])), 2), lo=round(float(ci.loc[k, 0]), 2),
                  hi=round(float(ci.loc[k, 1]), 2), p=round(float(f.pvalues[k]), 4)) for k in ks}
    t['b'] = pd.cut(t.ac_batch, [0, 3, 6, 9, 100], labels=['b1-3', 'b4-6', 'b7-9', 'b10+']).astype(str)
    o['ac_batch'] = t.groupby('b').apply(rate).reset_index().to_dict('records')
    o['ac_batch_wait40'] = t[t.fe_lead_Fill2_AC <= 40].groupby(pd.cut(t.ac_batch, [0, 1, 3, 6, 9, 100]).astype(str)).apply(rate).reset_index().to_dict('records')
    g = t[t.ac_batch <= 3].groupby('date_AutoClave').y.agg(['size', 'sum']); g = g[g['size'] >= 2]
    o['small_batch_all_or_nothing'] = dict(batches=len(g), all_defect=round(float((g['sum'] == g['size']).mean()), 3), zero=round(float((g['sum'] == 0).mean()), 3))
    o['batch_profile'] = t.groupby('b').agg(day_first5=('day_first5', 'mean'), pre_brk=('pre_brk', 'mean'), wait=('fe_lead_Fill2_AC', 'mean')).round(3).to_dict('index')
    f2 = smf.logit("y ~ pre_brk + post1 + first_wo + day_first5 + C(b, Treatment('b10+')) + C(m) + C(model) + C(line) + C(hr)", t).fit(disp=0, maxiter=300)
    ci2 = np.exp(f2.conf_int())
    o['logit_with_batch'] = {k.replace("C(b, Treatment('b10+'))[T.", 'batch@').rstrip(']'): dict(OR=round(float(np.exp(f2.params[k])), 2), lo=round(float(ci2.loc[k, 0]), 2),
                             hi=round(float(ci2.loc[k, 1]), 2), p=round(float(f2.pvalues[k]), 4)) for k in f2.params.index if k.startswith('C(b') or k in ('first_wo', 'post1', 'pre_brk', 'day_first5')}
    out[name] = o
    print('=====', name, len(t)); [print(k, pd.DataFrame(v).to_string(index=False) if isinstance(v, list) else v) for k, v in o.items()]

json.dump(out, open('reports/22_sequence_breaks.json', 'w'), ensure_ascii=False, indent=1, default=str)

# 그림 (9월 이후)
o = out['sep']; base_rate = POPS['sep'].y.mean() * 100
fig, ax = plt.subplots(1, 4, figsize=(21, 4.6))
def bars(a_, D, lab, title, xl, hi_idx=()):
    a_.bar(range(len(D)), D.rate, color=['#c0392b' if i in hi_idx else '#9fb6d4' for i in range(len(D))])
    a_.errorbar(range(len(D)), D.rate, yerr=[D.rate - D.lo, D.hi - D.rate], fmt='none', ecolor='#555', capsize=3)
    for i, (r_, n_) in enumerate(zip(D.rate, D.n)): a_.text(i, D.hi.iloc[i] + 0.3, f'{r_:.1f}%\nn={int(n_):,}', ha='center', fontsize=7)
    a_.set_xticks(range(len(D))); a_.set_xticklabels(lab, fontsize=8); a_.set_xlabel(xl); a_.set_ylabel('defect %'); a_.set_title(title, fontsize=10)
    a_.axhline(base_rate, color='grey', ls='--', lw=.8)
    a_.set_ylim(0, D.hi.max() * 1.18)
D = pd.DataFrame(o['day_seq']); bars(ax[0], D, D.day_seq.astype(str), 'Order within the day (per line)', 'n-th product of the day', (0,))
D = pd.DataFrame(o['brk_pos']); lab = D.p.astype(str).replace({'전 6-10': '-10~-6', '전 3-5': '-5~-3', '전 2': '-2', '직전 1': '-1', '직후 1': '+1', '후 2-3': '+2~3', '후 4-6': '+4~6', '후 7-10': '+7~10'})
bars(ax[1], D, lab, 'Around a regular break (Fill2 order)', 'position (- before / + after)', (1, 3))
D = pd.DataFrame(o['brk_stage']).set_index('brk_stage').reindex(['none', 'Fill2→탈포', 'Dam→Fill1', 'Fill1→Fill2']).dropna(subset=['n']).reset_index()
bars(ax[2], D, D.brk_stage.map({'none': 'no break', 'Fill2→탈포': 'Fill2→degas', 'Dam→Fill1': 'Dam→Fill1', 'Fill1→Fill2': 'Fill1→Fill2'}), 'Which stage the break fell in', 'stage interval', (1, 2, 3))
D = pd.DataFrame(o['ac_batch']).set_index('b').reindex(['b1-3', 'b4-6', 'b7-9', 'b10+']).reset_index()
bars(ax[3], D, ['1-3', '4-6', '7-9', '10+'], 'Degas batch size (products per load)', 'products loaded together', (0,))
fig.subplots_adjust(wspace=.3); plt.savefig('figs/22_sequence_breaks.png', dpi=120, bbox_inches='tight')
