"""1단계 EDA: 구조·품질·범주별 불량률·시간 추이·train/test 비교"""
import pandas as pd, numpy as np, json, warnings
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
plt.rcParams['font.family']='DejaVu Sans'
tr = pd.read_csv('data/train.csv', low_memory=False); te = pd.read_csv('data/test.csv', low_memory=False)
y = (tr.target=='AbNormal').astype(int)
R = {}
R['shape'] = {'train': tr.shape, 'test': te.shape}
R['defect_rate'] = float(y.mean())
nul = tr.isna().mean(); nun = tr.nunique(dropna=True)
R['all_null'] = int((nul==1).sum()); R['const'] = int(((nun<=1)&(nul<1)).sum())
R['usable'] = int((nun>1).sum())
procs=['Dam','Fill1','Fill2','AutoClave']
R['cols_per_proc'] = {p:int(sum(c.endswith('_'+p) for c in tr.columns)) for p in procs}
R['usable_per_proc'] = {p:int(sum(c.endswith('_'+p) and nun[c]>1 for c in tr.columns)) for p in procs}
R['partial_null'] = nul[(nul>0)&(nul<1)].round(3).to_dict()
# 로그 포맷(열 밀림) 그룹
k='HEAD NORMAL COORDINATE X AXIS(Stage1) Collect Result_Dam'
grp = pd.Series(np.where(tr[k].isna(),'B_NaN',np.where(tr[k].eq('OK'),'C_OK','A_num')))
dt = pd.to_datetime(tr['Collect Date_Dam'])
R['date_range'] = [str(dt.min()), str(dt.max())]
def rate(s):
    t = pd.DataFrame({'k':s.values,'y':y.values}).groupby('k')['y'].agg(['size','mean']); t.columns=['n','defect_rate']; return t
tabs = {}
tabs['log_format'] = rate(grp)
for p in ['Dam','Fill1','Fill2']:
    tabs[f'equipment_{p}'] = rate(tr[f'Equipment_{p}'])
combo = tr['Equipment_Dam'].str[-2:]+'|'+tr['Equipment_Fill1'].str[-2:]+'|'+tr['Equipment_Fill2'].str[-2:]
tabs['equipment_combo(Dam|Fill1|Fill2)'] = rate(combo)
tabs['model_suffix'] = rate(tr['Model.Suffix_Dam'])
tabs['month'] = rate(dt.dt.to_period('M').astype(str))
tabs['chamber_temp_judge'] = rate(tr['Chamber Temp. Judge Value_AutoClave'])
wo = rate(tr['Workorder_Dam']); R['workorder'] = {'n':len(wo), 'rate_q':wo['defect_rate'].quantile([.5,.9,.99]).round(3).to_dict(),
   'zero_defect_wo_share': float((wo.defect_rate==0).mean()), 'top10': wo[wo.n>=30].sort_values('defect_rate',ascending=False).head(10).round(3).to_dict('index')}
# 공정 간 동일 범주 컬럼 중복 여부
R['dup_meta'] = {c: bool((tr[f'{c}_Dam']==tr[f'{c}_AutoClave']).all()) for c in ['Model.Suffix','Workorder']}
# train/test 비교
te_dt = pd.to_datetime(te['Collect Date_Dam'])
R['test_month'] = te_dt.dt.to_period('M').astype(str).value_counts(normalize=True).sort_index().round(3).to_dict()
R['train_month'] = dt.dt.to_period('M').astype(str).value_counts(normalize=True).sort_index().round(3).to_dict()
R['test_equip_dam'] = te['Equipment_Dam'].value_counts(normalize=True).round(3).to_dict()
R['train_equip_dam'] = tr['Equipment_Dam'].value_counts(normalize=True).round(3).to_dict()
json.dump(R, open('reports/01_eda_stats.json','w'), ensure_ascii=False, indent=1, default=str)
with open('reports/01_eda_tables.md','w') as f:
    for n,t in tabs.items(): f.write(f'### {n}\n\n'+t.assign(defect_rate=lambda d:(d.defect_rate*100).round(2)).to_markdown()+'\n\n')
# 그림
fig,ax=plt.subplots(1,2,figsize=(12,4))
m=tabs['month']; ax[0].bar(m.index,m.n,color='#9bb7d4'); ax2=ax[0].twinx(); ax2.plot(m.index,m.defect_rate*100,'o-',color='#c0392b')
ax[0].set_title('Monthly volume (bar) & defect rate % (line)'); ax[0].tick_params(axis='x',rotation=60); ax2.set_ylabel('defect %')
c=tabs['equipment_combo(Dam|Fill1|Fill2)'].sort_values('defect_rate'); ax[1].barh(c.index,c.defect_rate*100,color='#c0392b'); ax[1].set_title('Defect % by equipment combo (Dam|Fill1|Fill2)')
for i,(n,r) in enumerate(zip(c.n,c.defect_rate)): ax[1].text(r*100,i,f' n={n}',va='center',fontsize=8)
plt.tight_layout(); plt.savefig('figs/01_month_equipment.png',dpi=120)
fig,ax=plt.subplots(figsize=(7,3.5)); t=tabs['model_suffix']; ax.bar([f'Model {chr(65 + i)}' for i in range(len(t))],t.defect_rate*100,color='#c0392b'); ax.set_title('Defect % by Model (anonymized)'); ax.tick_params(axis='x',rotation=30)
for i,(n,r) in enumerate(zip(t.n,t.defect_rate)): ax.text(i,r*100,f'n={n}',ha='center',va='bottom',fontsize=8)
plt.tight_layout(); plt.savefig('figs/01_model_suffix.png',dpi=120)
print(json.dumps({k:R[k] for k in ['shape','defect_rate','all_null','const','usable','cols_per_proc','usable_per_proc','date_range','dup_meta','partial_null']},default=str,ensure_ascii=False,indent=0))
print(open('reports/01_eda_tables.md').read())
print(R['workorder'])
