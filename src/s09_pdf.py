"""보고서 00~25 마크다운 → 단일 PDF (WeasyPrint, Noto Sans KR)"""
import markdown, re, glob, datetime
from weasyprint import HTML

FILES = sorted(glob.glob('reports/[0-2][0-9]_*.md'))   # 00 ~ 25단계 전체 (16·19·23단계 이후 정정 반영)
# 이모지는 한글 폰트에 글리프가 없어 텍스트 기호로 치환
EMOJI = {'🔴': '<b class="r">[높음]</b>', '🟠': '<b class="o">[중간]</b>', '🟡': '<b class="y">[낮음]</b>', '🟢': '<b class="g">[참고]</b>',
         '✅': '<b class="g">OK</b>', '⚠️': '<b class="o">주의</b>', '⭐': '★', '❌': 'X', '✗': 'X', '△': '△'}

def fix_lists(t):
    """문단 바로 다음 줄의 목록 앞에 빈 줄 삽입 (python-markdown은 빈 줄이 있어야 목록으로 인식)"""
    out, prev = [], ''
    for line in t.split('\n'):
        is_item = bool(re.match(r'^\s*([-*]|\d+\.)\s', line))
        prev_item = bool(re.match(r'^\s*([-*]|\d+\.)\s', prev))
        if is_item and prev.strip() and not prev_item and not prev.lstrip().startswith(('|', '>', '#')): out.append('')
        out.append(line); prev = line
    return '\n'.join(out)


def conv(path):
    t = fix_lists(open(path, encoding='utf-8').read())
    for k, v in EMOJI.items(): t = t.replace(k, v)
    t = t.replace('\\|', '&#124;')
    h = markdown.markdown(t, extensions=['tables', 'fenced_code', 'sane_lists'])
    return f'<section class="chap">{h}</section>'

def title(f):
    first = open(f, encoding='utf-8').readline().strip()
    return re.sub(r'^# ', '', first) if first.startswith('# ') else f.split('/')[-1][:-3] + ' (부록 표)'
toc = ''.join(f'<li>{title(f)}</li>' for f in FILES)
body = ''.join(conv(f) for f in FILES)
html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><style>
@page {{ size: A4; margin: 18mm 16mm 18mm 16mm;
  @bottom-center {{ content: counter(page) " / " counter(pages); font-size: 8pt; color: #888; }} }}
@page :first {{ @bottom-center {{ content: none; }} }}
body {{ font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif; font-size: 9.5pt; line-height: 1.55; color: #222; }}
.cover {{ height: 240mm; display: flex; flex-direction: column; justify-content: center; page-break-after: always; }}
.cover h1 {{ font-size: 24pt; border: none; margin: 0 0 6mm; color: #1a2b4c; }}
.cover .sub {{ font-size: 12pt; color: #555; }} .cover ol {{ margin-top: 14mm; font-size: 10.5pt; line-height: 1.9; }}
.chap {{ page-break-before: always; }}
h1 {{ font-size: 17pt; color: #1a2b4c; border-bottom: 2px solid #1a2b4c; padding-bottom: 2mm; margin-top: 0; }}
h2 {{ font-size: 13pt; color: #1a2b4c; margin-top: 7mm; border-left: 4px solid #c0392b; padding-left: 2.5mm; }}
h3 {{ font-size: 11pt; color: #333; margin-top: 5mm; }}
table {{ border-collapse: collapse; width: 100%; margin: 2.5mm 0 4mm; font-size: 8.3pt; page-break-inside: auto; }}
tr {{ page-break-inside: avoid; }}
th {{ background: #1a2b4c; color: #fff; padding: 1.6mm 2mm; text-align: left; font-weight: 600; }}
td {{ border-bottom: 1px solid #ddd; padding: 1.4mm 2mm; vertical-align: top; }}
tr:nth-child(even) td {{ background: #f6f7f9; }}
blockquote {{ margin: 3mm 0; padding: 2mm 4mm; background: #f3f5f8; border-left: 3px solid #8a9bb5; color: #444; }}
blockquote p {{ margin: 1mm 0; }}
code {{ font-family: 'DejaVu Sans Mono', monospace; font-size: 8pt; background: #f1f1f1; padding: 0 1mm; }}
pre {{ background: #f6f7f9; padding: 3mm; font-size: 8pt; white-space: pre-wrap; }}
img {{ max-width: 100%; display: block; margin: 3mm auto; page-break-inside: avoid; }}
hr {{ border: none; border-top: 1px solid #ccc; margin: 5mm 0; }}
b.r, b.o, b.y, b.g {{ white-space: nowrap; }} b.r {{ color: #c0392b; }} b.o {{ color: #d35400; }} b.y {{ color: #b7950b; }} b.g {{ color: #1e8449; }}
</style></head><body>
<div class="cover"><h1>디스플레이 합착 공정<br>불량 원인 분석 · 예측 모델링</h1>
<div class="sub">LG Aimers 5기 본선 데이터 · Sub Assembly Line (Dam → Fill1 → Fill2 → AutoClave)</div>
<div class="sub">{datetime.date.today():%Y-%m-%d}</div><ol>{toc}</ol></div>
{body}</body></html>"""
HTML(string=html, base_url='reports/').write_pdf('reports/lg5_불량원인분석_보고서.pdf')
print('ok')
