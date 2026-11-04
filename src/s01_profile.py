import pandas as pd, numpy as np, re, collections
tr = pd.read_csv('data/train.csv', low_memory=False)
te = pd.read_csv('data/test.csv', low_memory=False)
print('train', tr.shape, 'test', te.shape)
print('test cols not in train', set(te.columns)-set(tr.columns))
print('train cols not in test', set(tr.columns)-set(te.columns))
# 공정 suffix
proc = collections.Counter(c.rsplit('_',1)[-1] for c in tr.columns if '_' in c)
print('process suffix', proc)
print(tr.dtypes.value_counts())
nul = tr.isna().mean()
print('all-null', (nul==1).sum(), '>50%', ((nul>0.5)&(nul<1)).sum(), 'some', ((nul>0)&(nul<=0.5)).sum())
nun = tr.nunique(dropna=True)
print('constant(<=1 unique)', (nun<=1).sum())
obj = tr.select_dtypes('object').columns
for c in obj:
    print('OBJ', c, tr[c].nunique(), tr[c].value_counts().head(4).to_dict())
