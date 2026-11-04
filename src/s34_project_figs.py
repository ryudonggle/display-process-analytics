"""34단계: 프로젝트 보고서(성과 먼저 구조)용 그림 — 성능 사다리 (32 · 33단계 결과)
결과: figs/proj_ladder.png, figs/proj_contrib.png
"""
import json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for w in ('Regular', 'Bold'): fm.fontManager.addfont(f'fonts/NotoSansKR-{w}.ttf')
plt.rcParams.update({'font.family': 'Noto Sans KR Static', 'axes.unicode_minus': False, 'axes.spines.top': False, 'axes.spines.right': False})
RED, BLUE, GREY, DARK, TEAL = '#1D6A72', '#B9C0C6', '#222222', '#222222', '#1D6A72'   # 포스터 기준 색
L = json.load(open('reports/32_feature_ladder.json', encoding='utf-8')); S = L['stages'][:4]; base = L['base_rate']
lab = ['원본 그대로', '데이터 복원', '+ 공정 물리 변수', '+ 공정 사이\n대기 시간 등']

fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw=dict(width_ratios=[1.15, 1.15, 0.9]))
for a, key, ttl, rb, fmt in [(ax[0], 'cv_pr_auc', '같은 기간 예측 (대회 방식)', base, '{:.3f}'), (ax[1], 'roll_lift', '다음 달 예측 (현장 방식)', 1, '{:.1f}배')]:
    v = [s[key] for s in S]; cols = [BLUE, BLUE, BLUE, RED]
    a.bar(lab, v, color=cols, width=.62)
    for i, x in enumerate(v): a.text(i, x + max(v) * .02, fmt.format(x), ha='center', fontsize=11, weight='bold', color=RED if i == 3 else DARK)
    a.axhline(rb, color=GREY, ls='--', lw=1); a.text(-.4, rb + max(v) * .02, '무작위' if key == 'cv_pr_auc' else '무작위 = 1배', fontsize=8.5, color='#222222')
    a.set_title(ttl, fontsize=11.5, weight='bold', loc='left'); a.set_ylim(0, max(v) * 1.2); a.tick_params(axis='x', labelsize=9)
    a.set_ylabel('PR-AUC' if key == 'cv_pr_auc' else '무작위 대비 (배)')
c = ax[2]; v0, v3 = S[0]['cv_top1_precision'] * 100, S[3]['cv_top1_precision'] * 100
c.bar(['원본 그대로', '최종'], [v0, v3], color=[BLUE, RED], width=.55)
for i, x in enumerate([v0, v3]): c.text(i, x + 2, f'{x:.0f}%', ha='center', fontsize=14, weight='bold', color=RED if i else DARK)
c.set_ylim(0, 100); c.set_ylabel('%'); c.set_title('위험 상위 1% 재검사 시\n진짜 불량 비율', fontsize=11.5, weight='bold', loc='left')
fig.suptitle(f"원본 {S[0]['cv_pr_auc']:.3f} → 최종 {S[3]['cv_pr_auc']:.3f} (+{(S[3]['cv_pr_auc']/S[0]['cv_pr_auc']-1)*100:.0f}%) · 다음 달 예측 {S[0]['roll_lift']:.1f}배 → {S[3]['roll_lift']:.1f}배",
             fontsize=13.5, weight='bold', color=DARK, x=.01, ha='left', y=1.03)
plt.tight_layout(); plt.savefig('figs/proj_ladder.png', dpi=150, bbox_inches='tight'); plt.close()

# 단계별 기여 (같은 기간 PR-AUC 증가분)
inc = [S[1]['cv_pr_auc'] - S[0]['cv_pr_auc'], S[2]['cv_pr_auc'] - S[1]['cv_pr_auc'], S[3]['cv_pr_auc'] - S[2]['cv_pr_auc']]
names = ['데이터 복원\n(밀린 59% 행 바로잡기)', '공정 물리 변수\n(토출률 · 압력×시간 · 좌표 어긋남)', '공정 사이 대기 시간\n(대기 · 휴식 겹침 · 라인 변경)']
fig, ax = plt.subplots(figsize=(9, 3.4)); ax.barh(names[::-1], inc[::-1], color=[RED, BLUE, '#DDE0E3'], height=.55)
for i, x in enumerate(inc[::-1]): ax.text(max(x, 0) + .002, i, f'{x:+.3f}', va='center', fontsize=10.5, weight='bold')
ax.axvline(0, color=GREY, lw=1); ax.set_xlabel('같은 기간 PR-AUC 증가분'); ax.set_xlim(-.01, .12)
ax.set_title('성능을 끌어올린 것은 "공정 사이에서 보낸 시간"', fontsize=12, weight='bold', loc='left')
plt.tight_layout(); plt.savefig('figs/proj_contrib.png', dpi=150, bbox_inches='tight'); plt.close(); print('ok')
