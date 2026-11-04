"""38단계: 성능 사다리 단계별 혼동행렬 — 같은 기간 5-fold OOF, 위험 상위 3%(같은 개수) 재검사 기준
단계 0 원본 그대로 · 1 데이터 복원 · 2 + 공정 물리 변수 · 3 + 공정 사이 대기 시간 등 (32단계와 같은 변수·모델)
결과: reports/38_ladder_confusion.json, data/ladder_oof.parquet, figs/proj_ladder_confusion.png (가로) · proj_ladder_confusion_a4.png (2×2)
"""
import json, numpy as np, pandas as pd, warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
warnings.filterwarnings('ignore')
exec(open('src/s32_feature_ladder.py', encoding='utf-8').read().split('MONTHS = ')[0])
for w in ('Regular', 'Bold'): fm.fontManager.addfont(f'fonts/NotoSansKR-{w}.ttf')
plt.rcParams.update({'font.family': 'Noto Sans KR Static', 'axes.unicode_minus': False})
DARK = '#222222'
y = tr.y.values; OOF = {}; out = []
for name, fs, dev in STAGES[:4]:
    o = np.zeros(len(tr))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(tr, y): o[b] = fit_pred2(tr.iloc[a], tr.iloc[b], fs, dev, SEED)
    OOF[name] = o
    n = int(round(len(tr) * .03)); flag = np.zeros(len(tr), bool); flag[np.argsort(-o)[:n]] = True
    tp = int((flag & (y == 1)).sum()); fp = n - tp; fn = int(y.sum()) - tp; tn = len(tr) - n - fn
    out.append(dict(stage=name, pr_auc=round(average_precision_score(y, o), 4), inspected=n, TP=tp, FP=fp, FN=fn, TN=tn,
                    precision=round(tp / n, 3), recall=round(tp / y.sum(), 3)))
    print(out[-1], flush=True)
pd.DataFrame(OOF).assign(y=y).to_parquet('data/ladder_oof.parquet', index=False)
json.dump(out, open('reports/38_ladder_confusion.json', 'w'), ensure_ascii=False, indent=1)

LAB = ['0. 원본 그대로', '1. 데이터 복원', '2. + 공정 물리 변수', '3. + 공정 사이 대기 시간 등 (최종)']


def draw(ax, c, title, hot):
    M = np.array([[c['TP'], c['FN']], [c['FP'], c['TN']]])
    col = np.array([['#1D6A72' if hot else '#7FAEB3', '#F6DEDA'], ['#DDE0E3', '#D7E9EA']])
    nm = np.array([['TP · 잡았다', 'FN · 놓쳤다'], ['FP · 헛검사', 'TN · 통과']])
    for r in range(2):
        for k in range(2):
            w = (r, k) == (0, 0)
            ax.add_patch(plt.Rectangle((k, 1 - r), 1, 1, fc=col[r, k], ec='white', lw=3))
            ax.text(k + .5, 1 - r + .58, f'{M[r, k]:,}', ha='center', va='center', fontsize=20, weight='bold', color='white' if w else DARK)
            ax.text(k + .5, 1 - r + .25, nm[r, k], ha='center', va='center', fontsize=10.5, color='white' if w else '#333')
    ax.set_xlim(0, 2); ax.set_ylim(0, 2); ax.set_xticks([.5, 1.5]); ax.set_xticklabels(['모델: 불량 의심', '모델: 정상'], fontsize=10)
    ax.set_yticks([1.5, .5]); ax.set_yticklabels(['실제\n불량', '실제\n정상'], fontsize=10); ax.tick_params(length=0); ax.xaxis.set_ticks_position('top')
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title(f"{title}\nPR-AUC {c['pr_auc']:.3f} · 정밀도 {c['precision']*100:.0f}% · 재현율 {c['recall']*100:.0f}%", fontsize=12, weight='bold', pad=28, loc='left',
                 color='#1D6A72' if hot else DARK)


note = (f"같은 기간 5-fold 예측 · 제품 40,506개(불량 2,350개) · 네 단계 모두 위험 점수 상위 3%({out[0]['inspected']:,}개)를 재검사. "
        f"같은 검사 수로 잡은 불량 {out[0]['TP']} → {out[3]['TP']}개 (+{out[3]['TP'] - out[0]['TP']}), 헛검사 {out[0]['FP']} → {out[3]['FP']}개")
for fn, shape, size in [('proj_ladder_confusion', (1, 4), (21, 5.6)), ('proj_ladder_confusion_a4', (2, 2), (12, 11.5))]:
    fig, ax = plt.subplots(*shape, figsize=size); ax = ax.ravel()
    for i in range(4): draw(ax[i], out[i], LAB[i], i == 3)
    fig.text(.01, -.03 if shape == (1, 4) else -.02, note, fontsize=11, color='#222222')
    plt.tight_layout(h_pad=3); plt.savefig(f'figs/{fn}.png', dpi=150, bbox_inches='tight'); plt.close()
print('ok')
