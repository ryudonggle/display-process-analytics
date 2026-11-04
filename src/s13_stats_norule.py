"""13단계: 규칙 해당 132건(2시간 초과 대기·라인 변경·Fill1 미도포)을 빼고 4단계 통계 검정 재실행 → 전체 기준과 비교
규칙 행은 99% 불량이라, 리드타임과 상관된 변수가 '유의'하게 끌려 올라갔을 수 있는지 점검
"""
import pandas as pd, numpy as np, json, warnings
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore')

d = pd.read_parquet('data/features.parquet'); d = d[d.is_train == 1].copy()
d['recipe'] = np.where(d.fe_recipe_new == 1, 'new', 'old')
d['rule'] = ((d.fe_lead_total > 2) | (d.equip_mismatch == 1) | (d.fe_Fill1_zero_dispense == 1)).astype(int)
EXCL = {'fe_days', 'fe_recipe_new', 'fe_hour', 'fe_weekday', 'line', 'equip_Dam', 'equip_Fill1', 'equip_Fill2', 'is_train', 'target'}
num = [c for c in d.columns if (c.startswith(('Dam__', 'Fill1__', 'Fill2__', 'AutoClave__', 'fe_', 'dev_'))) and c not in EXCL
       and pd.api.types.is_numeric_dtype(d[c]) and d[c].nunique() > 1]

def mw(x, yy):
    a, b = x[yy == 1].dropna(), x[yy == 0].dropna()
    if a.nunique() + b.nunique() < 3: return np.nan, np.nan
    u, p = stats.mannwhitneyu(a, b, alternative='two-sided')
    return p, 2 * u / (len(a) * len(b)) - 1

def tests(t):
    y = t.target.astype(int); rows = []
    for c in num:
        p, r = mw(t[c], y)
        po, ro = mw(t.loc[t.recipe == 'old', c], y[t.recipe == 'old'])
        pn, rn = mw(t.loc[t.recipe == 'new', c], y[t.recipe == 'new'])
        rows.append(dict(feature=c, r_all=r, p_all=p, r_old=ro, p_old=po, r_new=rn, p_new=pn))
    R = pd.DataFrame(rows)
    for s in ('all', 'old', 'new'):
        m = R[f'p_{s}'].notna(); R.loc[m, f'q_{s}'] = multipletests(R.loc[m, f'p_{s}'], method='fdr_bh')[1]
    R['robust'] = (R.q_old < .05) & (R.q_new < .05) & (np.sign(R.r_old) == np.sign(R.r_new))
    R['sig_all'] = R.q_all < .05
    return R.set_index('feature')

def adj_logit(t, feats):
    y = t.target.astype(int)
    base = pd.get_dummies(t[['model']], drop_first=True, dtype=float)
    base['recipe_new'] = t.fe_recipe_new; base['days'] = (t.fe_days - t.fe_days.mean()) / t.fe_days.std(); base['line2'] = (t.line == 2).astype(float)
    out = {}
    for c in feats:
        if t[c].std() == 0: continue
        X = base.copy(); X['z'] = (t[c] - t[c].mean()) / t[c].std(); X = sm.add_constant(X.fillna(0))
        try:
            m = sm.Logit(y, X).fit(disp=0); out[c] = (np.exp(m.params['z']), m.pvalues['z'])
        except Exception:
            pass
    return pd.DataFrame(out, index=['OR', 'p']).T

full, clean = d, d[d.rule == 0]
Rf, Rc = tests(full), tests(clean)
cmp = pd.DataFrame({'r_full': Rf.r_all, 'r_clean': Rc.r_all, 'robust_full': Rf.robust, 'robust_clean': Rc.robust,
                    'sig_full': Rf.sig_all, 'sig_clean': Rc.sig_all})
cmp['rank_full'] = cmp.r_full.abs().rank(ascending=False); cmp['rank_clean'] = cmp.r_clean.abs().rank(ascending=False)
feats = cmp.index[cmp.robust_full | cmp.robust_clean].tolist()
Af, Ac = adj_logit(full, feats), adj_logit(clean, feats)
cmp = cmp.join(Af.add_suffix('_full')).join(Ac.add_suffix('_clean'))
cmp['status'] = np.select([cmp.robust_full & cmp.robust_clean, cmp.robust_full, cmp.robust_clean], ['유지', '사라짐', '새로 등장'], '')
cmp.sort_values('rank_clean').to_csv('reports/13_stats_norule.csv')

summ = dict(n_full=len(full), n_clean=len(clean), defect_full=round(full.target.mean(), 4), defect_clean=round(clean.target.mean(), 4),
            sig_full=int(cmp.sig_full.sum()), sig_clean=int(cmp.sig_clean.sum()),
            robust_full=int(cmp.robust_full.sum()), robust_clean=int(cmp.robust_clean.sum()),
            kept=cmp.index[cmp.status == '유지'].tolist(), dropped=cmp.index[cmp.status == '사라짐'].tolist(),
            new=cmp.index[cmp.status == '새로 등장'].tolist())
json.dump(summ, open('reports/13_stats_norule.json', 'w'), ensure_ascii=False, indent=1)
pd.set_option('display.width', 250)
print({k: v for k, v in summ.items() if not isinstance(v, list)})
for k in ('dropped', 'new'): print(k, summ[k])
show = cmp[cmp.status != ''].sort_values('rank_clean')
print(show[['status', 'r_full', 'r_clean', 'rank_full', 'rank_clean', 'OR_full', 'p_full', 'OR_clean', 'p_clean']].round(4).to_string())
