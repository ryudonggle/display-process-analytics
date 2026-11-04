import pandas as pd, numpy as np
tr = pd.read_csv('data/train.csv', low_memory=False)
k='HEAD NORMAL COORDINATE X AXIS(Stage1) Collect Result_Dam'
grp = np.where(tr[k].eq('OK'),'OK',np.where(tr[k].isna(),'NaN','num'))
print(pd.Series(grp).value_counts())
print(pd.crosstab(grp, tr.target, normalize='index').round(3))
for p in ['Dam','Fill1','Fill2','AutoClave']:
    cols=[c for c in tr.columns if c.endswith('_'+p) and ('Collect Result' in c or 'Unit Time' in c or 'Judge' in c)]
    cols=[c for c in cols if tr[c].notna().any()]
    out={}
    for g in ['OK','NaN','num']:
        sub=tr.loc[grp==g,cols].apply(pd.to_numeric,errors='coerce')
        out[g]=sub.median()
    d=pd.DataFrame(out); 
    print('==',p); print(d.rename(index=lambda s:s[:58]).round(1).to_string())
