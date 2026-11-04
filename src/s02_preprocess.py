"""2단계 전처리: 열 밀림 보정 → 불필요 열 제거 → 타입 정리 → 기본 파생 → parquet 저장"""
import pandas as pd, numpy as np, json

PROCS = ['Dam', 'Fill1', 'Fill2', 'AutoClave']
META = ['Wip Line', 'Process Desc.', 'Equipment', 'Model.Suffix', 'Workorder', 'Collect Date', 'Insp. Seq No.', 'Insp Judge Code']
SHIFT_START = 'HEAD NORMAL COORDINATE X AXIS(Stage1) Collect Result'

tr = pd.read_csv('data/train.csv', low_memory=False)
te = pd.read_csv('data/test.csv', low_memory=False)
tr['Set ID'] = [f'TR{i:05d}' for i in range(len(tr))]
te = te.drop(columns='target')
df = pd.concat([tr.assign(is_train=1), te.assign(is_train=0)], ignore_index=True)
n0 = len(df)
log = {}

# 1) 로그 포맷 그룹: A=정상 정렬(X1 숫자), B=X1 결측, C=X1 'OK' → B·C는 한 칸 밀림
x1 = df[f'{SHIFT_START}_Dam']
fmt = np.select([x1.isna(), x1.astype(str).eq('OK')], ['B', 'C'], 'A')
shifted = fmt != 'A'
log['format_counts'] = pd.Series(fmt).value_counts().to_dict()

# 2) 열 밀림 보정: 공정별로 X1 ~ 마지막 Collect Result 열을 한 칸 왼쪽으로
for p in ['Dam', 'Fill1', 'Fill2']:
    cols = [c for c in df.columns if c.endswith('_' + p) and 'Collect Result' in c]
    span = cols[cols.index(f'{SHIFT_START}_{p}'):]
    # X1 자리의 'OK'는 밀려난 쓰레기 값이므로 숫자 변환 시 NaN이 되어도 무방
    num = df[span].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
    num[shifted] = np.concatenate([num[shifted, 1:], np.full((shifted.sum(), 1), np.nan)], axis=1)
    df[span] = num
    log[f'shift_span_{p}'] = [span[0], span[-1], len(span)]

# 3) 공정 메타 정리: 공정별 Equipment·Collect Date만 유지, 나머지 메타는 Dam 것 하나만
keep = {'Model.Suffix_Dam': 'model', 'Workorder_Dam': 'workorder'}
for p in PROCS:
    keep[f'Equipment_{p}'] = f'equip_{p}'
    keep[f'Collect Date_{p}'] = f'date_{p}'
meta = df[list(keep)].rename(columns=keep)
meta_cols_all = [f'{m}_{p}' for m in META for p in PROCS if f'{m}_{p}' in df.columns]

# 4) 측정 열: 숫자로 변환 후 전부 결측·상수 제거 (train 기준)
meas = df.drop(columns=meta_cols_all + ['target', 'Set ID', 'is_train'])
meas = meas.apply(pd.to_numeric, errors='coerce')
trm = meas[df.is_train.eq(1).values]
drop_null = trm.columns[trm.isna().all()].tolist()
nun = trm.nunique()
drop_const = [c for c in nun[nun <= 1].index if c not in drop_null]
meas = meas.drop(columns=drop_null + drop_const)
log['drop_all_null'] = len(drop_null); log['drop_const'] = len(drop_const)

# 5) 완전 중복 열 제거 (공정 간 같은 값을 기록한 열 등)
dups = meas.T.duplicated()
log['drop_duplicate'] = meas.columns[dups].tolist()
meas = meas.loc[:, ~dups.values]

# 6) 이름 단순화: "XXX Collect Result_Dam" → "Dam__XXX"
def short(c):
    name, p = c.rsplit('_', 1)
    name = name.replace(' Collect Result', '').replace('1st Pressure 1st Pressure Unit Time', '1st Pressure Unit Time')
    return f'{p}__{name}'
meas.columns = [short(c) for c in meas.columns]

# 7) 메타 파생
out = pd.concat([df[['Set ID', 'is_train']], meta, meas], axis=1)
out['target'] = (df['target'] == 'AbNormal').astype(float).where(df.is_train.eq(1))
out['log_format'] = fmt
for p in PROCS:
    out[f'date_{p}'] = pd.to_datetime(out[f'date_{p}'])
    out[f'equip_{p}'] = out[f'equip_{p}'].str.extract(r'#(\d)')[0].astype(float) if p != 'AutoClave' else 0
out = out.drop(columns='equip_AutoClave')
out['equip_mismatch'] = ((out.equip_Dam != out.equip_Fill1) | (out.equip_Fill1 != out.equip_Fill2)).astype(int)
out['line'] = out['equip_Dam'].astype(int)

# 검증
assert len(out) == n0
chk = out[out.is_train.eq(1)]
med = chk.groupby('log_format')[['Dam__HEAD NORMAL COORDINATE X AXIS(Stage1)', 'Dam__HEAD NORMAL COORDINATE X AXIS(Stage2)',
                                 'Dam__HEAD NORMAL COORDINATE Y AXIS(Stage1)', 'Fill1__HEAD NORMAL COORDINATE X AXIS(Stage1)']].median()
log['post_shift_median_check'] = med.round(1).to_dict()
log['final_shape'] = list(out.shape)
log['n_measure_cols'] = meas.shape[1]
log['measure_null_rate'] = {c: round(float(v), 3) for c, v in chk[meas.columns].isna().mean().items() if v > 0}

out.to_parquet('data/clean.parquet', index=False)
json.dump(log, open('reports/02_preprocess_log.json', 'w'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(log, ensure_ascii=False, indent=1, default=str))
print(out.dtypes.value_counts())
