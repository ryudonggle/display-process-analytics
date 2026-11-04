import pandas as pd
tr = pd.read_csv('data/train.csv', low_memory=False)
k='HEAD NORMAL COORDINATE X AXIS(Stage1) Collect Result_Dam'
sh = tr[k].eq('OK')
for p in ['Dam','Fill1','Fill2']:
    cols=[c for c in tr.columns if c.endswith('_'+p)]
    a=tr.loc[sh,cols].notna().mean(); b=tr.loc[~sh,cols].notna().mean()
    d=pd.DataFrame({'s':a,'n':b}); d=d[(d['s']-d['n']).abs()>0.01]
    print('==',p); print(d.round(2).to_string())
    # 비교: 대표 행 하나씩 non-null 열 이름-값
    m=tr.loc[sh,cols].iloc[0]; n=tr.loc[~sh,cols].iloc[0]
    i=cols.index(k.replace('_Dam','_'+p))
    print(pd.DataFrame({'shift':m.iloc[i-3:i+12].values,'normal':n.iloc[i-3:i+12].values},index=[c[:60] for c in cols[i-3:i+12]]).to_string())
