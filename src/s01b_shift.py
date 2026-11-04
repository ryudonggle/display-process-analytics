import pandas as pd, numpy as np
tr = pd.read_csv('data/train.csv', low_memory=False)
k='HEAD NORMAL COORDINATE X AXIS(Stage1) Collect Result_Dam'
sh = tr[k].eq('OK')
print('shift rows', sh.sum(), 'defect rate shift/normal', (tr.target[sh]=='AbNormal').mean(), (tr.target[~sh]=='AbNormal').mean())
for p in ['Dam','Fill1','Fill2','AutoClave']:
    cols=[c for c in tr.columns if c.endswith('_'+p)]
    a=tr.loc[sh,cols].notna().mean(); b=tr.loc[~sh,cols].notna().mean()
    d=pd.DataFrame({'shift':a,'norm':b}); d=d[(d.shift!=d.norm)]
    print('==',p, len(d)); print(d.round(2).to_string()[:3000])
