"""24단계: 탈포 배치 크기 심층 분석 — '적게 넣어서'인가, '탈포기가 식어서'인가
- 배치 크기: 같은 탈포 시각(10분 칸)에 기록된 제품 수 (train+test, 라벨 미사용)
- 탈포기 휴지 시간: 직전 탈포 기록 이후 경과(분) — 탈포기 공통(라인 구분 없음). 길수록 챔버가 식었을 가능성 (예열 대리 지표)
- 챔버 온도: AutoClave__Chamber Temp. (설정/기록값)
검증: 용량-반응, 휴지 시간·챔버 온도 동시 통제(월·모델·라인·시각), 휴지 시간 짧은 경우에도 배치 효과가 남는지
대상: 2023-09 이후, 규칙·Dam Y 음수 편차 제외 (원인 분석용 — 다른 제품 행 사용)
결과: reports/24_batch_deepdive.json, figs/24_batch_deepdive.png
"""
import pandas as pd, numpy as np, json, warnings
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportion_confint
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

d = pd.read_parquet('data/features.parquet')
d['ac_batch'] = d.groupby('date_AutoClave')['Set ID'].transform('size')
u = np.sort(d.date_AutoClave.unique()); prev = pd.Series(np.r_[np.datetime64('NaT'), u[:-1]], index=u)
d['ac_idle'] = ((d.date_AutoClave - d.date_AutoClave.map(prev)).dt.total_seconds() / 60)   # 직전 탈포 기록 이후(분)
d.loc[d.date_AutoClave.dt.normalize() != d.date_AutoClave.map(prev).dt.normalize(), 'ac_idle'] = np.nan   # 그날 첫 탈포는 별도 표시
d['ac_first_of_day'] = d.ac_idle.isna().astype(int)
tr = d[d.is_train == 1].copy(); tr['y'] = tr.target.astype(int); tr['m'] = tr.date_Dam.dt.to_period('M').astype(str); tr['hr'] = tr.date_AutoClave.dt.hour
rule = (tr.fe_lead_total > 2) | (tr.equip_mismatch == 1) | (tr.fe_Fill1_zero_dispense == 1)
t = tr[(tr.date_Dam >= '2023-09-01') & ~rule & ~(tr['dev_Dam__Y(Stage2)'] < -0.05)].copy()
t['temp'] = t['AutoClave__Chamber Temp.']; t['b'] = pd.cut(t.ac_batch, [0, 3, 6, 9, 100], labels=['b1-3', 'b4-6', 'b7-9', 'b10+']).astype(str)
t['idle_c'] = pd.cut(t.ac_idle, [-1, 10, 20, 40, 1e4], labels=['≤10', '11-20', '21-40', '>40']).astype(str).where(t.ac_first_of_day == 0, 'day_first')


def rate(g):
    n, k = len(g), int(g.y.sum()); lo, hi = proportion_confint(k, n, method='wilson') if n else (np.nan, np.nan)
    return pd.Series(dict(n=n, rate=round(k / n * 100, 2) if n else np.nan, lo=round(lo * 100, 2), hi=round(hi * 100, 2)))


out = {}
out['dose'] = t.groupby(t.ac_batch.clip(upper=16)).apply(rate).reset_index().to_dict('records')
out['idle'] = t.groupby('idle_c').apply(rate).reset_index().to_dict('records')
out['batch_x_idle'] = t.assign(small=(t.ac_batch <= 3).map({True: 'b1-3', False: 'b4+'})).groupby(['small', 'idle_c']).apply(rate).reset_index().to_dict('records')
out['temp_by_batch'] = t.groupby('b').temp.describe()[['mean', '50%']].round(2).to_dict('index')
out['idle_by_batch'] = t.groupby('b').agg(idle_med=('ac_idle', 'median'), day_first=('ac_first_of_day', 'mean')).round(2).to_dict('index')
out['temp'] = t.groupby(pd.cut(t.temp, [0, 46, 49, 52, 54, 60])).apply(rate).reset_index().astype({'temp': str}).to_dict('records')

M = {}
for nm, f in [('batch', "y ~ C(b, Treatment('b10+')) + C(m) + C(model) + C(line) + C(hr)"),
              ('batch+idle', "y ~ C(b, Treatment('b10+')) + C(idle_c, Treatment('≤10')) + C(m) + C(model) + C(line) + C(hr)"),
              ('batch+idle+temp', "y ~ C(b, Treatment('b10+')) + C(idle_c, Treatment('≤10')) + temp + C(m) + C(model) + C(line) + C(hr)")]:
    r = smf.logit(f, t).fit(disp=0, maxiter=300); ci = np.exp(r.conf_int())
    M[nm] = {k.replace("C(b, Treatment('b10+'))[T.", 'batch@').replace("C(idle_c, Treatment('≤10'))[T.", 'idle@').rstrip(']'):
             dict(OR=round(float(np.exp(r.params[k])), 2), lo=round(float(ci.loc[k, 0]), 2), hi=round(float(ci.loc[k, 1]), 2), p=round(float(r.pvalues[k]), 4))
             for k in r.params.index if k.startswith(('C(b', 'C(idle', 'temp'))}
out['logit'] = M
# 휴지 10분 이하(탈포기가 계속 돌던 중)인 경우에도 작은 배치가 위험한가
w = t[t.idle_c == '≤10']; out['warm_only_batch'] = w.groupby('b').apply(rate).reset_index().to_dict('records')
json.dump(out, open('reports/24_batch_deepdive.json', 'w'), ensure_ascii=False, indent=1, default=str)
for k, v in out.items(): print('==', k); print(pd.DataFrame(v).to_string() if isinstance(v, list) else json.dumps(v, ensure_ascii=False, default=str))

fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
D = pd.DataFrame(out['dose']); D = D[D.n >= 50]
ax[0].errorbar(D.ac_batch, D.rate, yerr=[D.rate - D.lo, D.hi - D.rate], fmt='o-', color='#c0392b', capsize=3)
ax[0].set_xlabel('products loaded together (16 = 16+)'); ax[0].set_ylabel('defect %'); ax[0].set_title('Dose-response: degas batch size', fontsize=10)
D = pd.DataFrame(out['batch_x_idle']); order = ['≤10', '11-20', '21-40', '>40', 'day_first']; x = np.arange(len(order))
for k, (lab, col) in enumerate([('b4+', '#9fb6d4'), ('b1-3', '#c0392b')]):
    g = D[D.small == lab].set_index('idle_c').reindex(order)
    ax[1].bar(x + (k - .5) * .38, g.rate, .38, color=col, label='batch 1-3' if lab == 'b1-3' else 'batch 4+')
    for xi, (r_, n_) in zip(x + (k - .5) * .38, zip(g.rate, g.n)):
        if pd.notna(r_): ax[1].text(xi, r_ + .3, f'{r_:.1f}%\nn={int(n_)}', ha='center', fontsize=7)
ax[1].set_xticks(x); ax[1].set_xticklabels(['≤10', '11-20', '21-40', '>40', "day's first"]); ax[1].set_xlabel('degas idle before this load (min)')
ax[1].set_ylabel('defect %'); ax[1].legend(fontsize=8); ax[1].set_title('Small batch vs cold chamber', fontsize=10)
L = M['batch+idle+temp']; keys = [k for k in L if k.startswith(('batch', 'idle'))] + ['temp']
ax[2].errorbar([L[k]['OR'] for k in keys], range(len(keys)), xerr=[[L[k]['OR'] - L[k]['lo'] for k in keys], [L[k]['hi'] - L[k]['OR'] for k in keys]], fmt='o', color='#2c3e50', capsize=3)
ax[2].axvline(1, color='grey', ls='--'); ax[2].set_yticks(range(len(keys))); ax[2].set_yticklabels([k.replace('idle@day_first', "idle@day's first") for k in keys], fontsize=8)
ax[2].set_xlabel('odds ratio (month·model·line·hour adjusted)'); ax[2].set_title('Batch, idle and chamber temp together', fontsize=10)
fig.subplots_adjust(wspace=.35); plt.savefig('figs/24_batch_deepdive.png', dpi=120, bbox_inches='tight')
