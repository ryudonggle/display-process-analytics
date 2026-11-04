import pandas as pd, numpy as np
tr = pd.read_csv('data/train.csv', low_memory=False); te = pd.read_csv('data/test.csv', low_memory=False)
def g(df,p):
    k=f'HEAD NORMAL COORDINATE X AXIS(Stage1) Collect Result_{p}'
    return np.where(df[k].isna(),'NaN',np.where(df[k].astype(str).eq('OK'),'OK','num'))
for name,df in [('train',tr),('test',te)]:
    G={p:g(df,p) for p in ['Dam','Fill1','Fill2']}
    print(name, pd.Series(G['Dam']).value_counts().to_dict(), 'same across procs', (G['Dam']==G['Fill1']).mean(), (G['Dam']==G['Fill2']).mean())
# 시간
tr['dt']=pd.to_datetime(tr['Collect Date_Dam']); te['dt']=pd.to_datetime(te['Collect Date_Dam'])
print('train range', tr.dt.min(), tr.dt.max(), 'test range', te.dt.min(), te.dt.max())
tr['grp']=g(tr,'Dam')
print(tr.groupby([tr.dt.dt.to_period('M'),'grp']).size().unstack(fill_value=0))
print(tr.groupby(['grp','Equipment_Dam']).size().unstack(fill_value=0))
# AutoClave 판정 NG 와 그룹
print(pd.crosstab(tr.grp, tr['Chamber Temp. Judge Value_AutoClave']))
print('test Set ID uniq', te['Set ID'].nunique(), te.shape[0], te['target'].isna().all() if 'target' in te else '')
