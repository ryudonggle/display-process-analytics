"""4단계 통계 원인 분석: 단변량 검정(FDR) → 레시피 층화(심슨 역설 점검) → 구간별 불량률 → 교란 통제 로지스틱"""
import pandas as pd, numpy as np, json, warnings
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

d = pd.read_parquet('data/features.parquet'); t = d[d.is_train == 1].copy()
y = t.target.astype(int); t['recipe'] = np.where(t.fe_recipe_new == 1, 'new', 'old')
F = json.load(open('reports/03_feature_dict.json'))
EXCL = {'fe_days', 'fe_recipe_new', 'fe_hour', 'fe_weekday', 'line', 'equip_Dam', 'equip_Fill1', 'equip_Fill2', 'is_train', 'target'}
num = [c for c in t.columns if (c.startswith(('Dam__', 'Fill1__', 'Fill2__', 'AutoClave__', 'fe_', 'dev_'))) and c not in EXCL
       and pd.api.types.is_numeric_dtype(t[c]) and t[c].nunique() > 1]

# 1) Mann-Whitney + 효과크기(rank-biserial r = 2·AUC−1) — 전체 / 레시피별
def mw(x, yy):
    a, b = x[yy == 1].dropna(), x[yy == 0].dropna()
    if a.nunique() + b.nunique() < 3: return np.nan, np.nan
    u, p = stats.mannwhitneyu(a, b, alternative='two-sided')
    return p, 2 * u / (len(a) * len(b)) - 1
rows = []
for c in num:
    p, r = mw(t[c], y)
    po, ro = mw(t.loc[t.recipe == 'old', c], y[t.recipe == 'old'])
    pn, rn = mw(t.loc[t.recipe == 'new', c], y[t.recipe == 'new'])
    rows.append(dict(feature=c, r_all=r, p_all=p, r_old=ro, p_old=po, r_new=rn, p_new=pn,
                     med_def=t.loc[y == 1, c].median(), med_ok=t.loc[y == 0, c].median()))
R = pd.DataFrame(rows)
for s in ('all', 'old', 'new'):
    m = R[f'p_{s}'].notna()
    R.loc[m, f'q_{s}'] = multipletests(R.loc[m, f'p_{s}'], method='fdr_bh')[1]
# 레시피 안에서도 같은 방향·유의해야 "견고(robust)"
R['robust'] = (R.q_old < .05) & (R.q_new < .05) & (np.sign(R.r_old) == np.sign(R.r_new))
R['simpson_flag'] = (R.q_all < .05) & ~R.robust
R['abs_r'] = R.r_all.abs()
R = R.sort_values('abs_r', ascending=False)
R.to_csv('reports/04_univariate_tests.csv', index=False)

# 2) 범주형: 카이제곱 + Cramér's V
cat_rows = []
for c in ['model', 'line', 'equip_mismatch', 'fe_wo_suffix', 'fe_Fill1_zero_dispense', 'recipe', 'fe_weekday', 'log_format']:
    ct = pd.crosstab(t[c], y)
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
    rate = (ct[1] / ct.sum(1) * 100).round(2).to_dict()
    cat_rows.append(dict(feature=c, chi2=round(chi2, 1), p=p, cramers_v=round(v, 3), rate_pct=rate, n=ct.sum(1).to_dict()))
C = pd.DataFrame(cat_rows).sort_values('cramers_v', ascending=False)

# 레시피 효과 자체: 같은 기간 인접 월(2023-09 vs 2023-10)로 근사 비교
mo = t.date_Dam.dt.to_period('M').astype(str)
near = t[mo.isin(['2023-08', '2023-09', '2023-10', '2023-11'])]
nr = near.groupby('recipe').target.agg(['size', 'mean'])
ztest = sm.stats.proportions_ztest(near.groupby('recipe').target.sum().values, nr['size'].values)

# 3) 교란 통제 로지스틱: 각 후보 변수(표준화) + 레시피 + 경과일 + 모델 + 라인
top = R[R.robust].head(15).feature.tolist()
base = pd.get_dummies(t[['model']], drop_first=True, dtype=float)
base['recipe_new'] = t.fe_recipe_new; base['days'] = (t.fe_days - t.fe_days.mean()) / t.fe_days.std(); base['line2'] = (t.line == 2).astype(float)
adj = []
for c in top:
    X = base.copy(); X['z'] = (t[c] - t[c].mean()) / t[c].std()
    X = sm.add_constant(X.fillna(0))
    try:
        m = sm.Logit(y, X).fit(disp=0)
        ci = m.conf_int().loc['z']
        adj.append(dict(feature=c, OR_per_SD=np.exp(m.params['z']), lo=np.exp(ci[0]), hi=np.exp(ci[1]), p=m.pvalues['z']))
    except Exception as e:
        adj.append(dict(feature=c, OR_per_SD=np.nan, lo=np.nan, hi=np.nan, p=np.nan))
A = pd.DataFrame(adj).sort_values('p')
A.to_csv('reports/04_adjusted_logit.csv', index=False)

# 4) 구간별 불량률 (5분위, 레시피별) — 상위 견고 변수 6개
plot_vars = R[R.robust].head(6).feature.tolist()
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
bins_out = {}
for ax, c in zip(axes.ravel(), plot_vars):
    for rc, col in [('old', '#7f8c8d'), ('new', '#c0392b')]:
        s = t[t.recipe == rc]
        try:
            q = pd.qcut(s[c], 5, duplicates='drop')
        except ValueError:
            continue
        g = s.groupby(q, observed=True).target.agg(['mean', 'size'])
        ax.plot(range(len(g)), g['mean'] * 100, 'o-', color=col, label=f'{rc} recipe')
        bins_out[f'{c}|{rc}'] = {str(k): [round(v * 100, 2), int(n)] for k, (v, n) in g.iterrows()}
    ax.set_title(c[:45], fontsize=10); ax.set_xlabel('quintile (low → high)'); ax.set_ylabel('defect %'); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig('figs/04_binned_defect.png', dpi=110)

out = dict(n_tested=len(R), n_sig_all=int((R.q_all < .05).sum()), n_robust=int(R.robust.sum()), n_simpson=int(R.simpson_flag.sum()),
           recipe_near=nr.round(4).to_dict(), recipe_z_p=float(ztest[1]))
json.dump(dict(summary=out, bins=bins_out), open('reports/04_stats_summary.json', 'w'), ensure_ascii=False, indent=1, default=str)
pd.set_option('display.width', 250)
print(out)
print(R[['feature', 'r_all', 'q_all', 'r_old', 'q_old', 'r_new', 'q_new', 'robust', 'med_def', 'med_ok']].head(30).round(4).to_string(index=False))
print('\nSIMPSON (전체 유의, 층화 후 불일치):'); print(R[R.simpson_flag][['feature', 'r_all', 'r_old', 'q_old', 'r_new', 'q_new']].head(12).round(4).to_string(index=False))
print(C[['feature', 'chi2', 'p', 'cramers_v', 'rate_pct']].to_string(index=False))
print(A.round(4).to_string(index=False))
