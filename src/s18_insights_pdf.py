"""16단계 인사이트 보고서(16_sep_insights.md) → PDF (s09_pdf.py와 같은 스타일)"""
import re, markdown, datetime
from weasyprint import HTML

def fix_lists(t):
    """문단 바로 다음 줄의 목록 앞에 빈 줄 삽입 (python-markdown은 빈 줄이 있어야 목록으로 인식)"""
    out, prev = [], ''
    for line in t.split('\n'):
        is_item = bool(re.match(r'^\s*([-*]|\d+\.)\s', line))
        prev_item = bool(re.match(r'^\s*([-*]|\d+\.)\s', prev))
        if is_item and prev.strip() and not prev_item and not prev.lstrip().startswith(('|', '>', '#')): out.append('')
        out.append(line); prev = line
    return '\n'.join(out)


t = fix_lists(open('reports/16_sep_insights.md', encoding='utf-8').read())
for k, v in {'⭐': '★', '→': '→'}.items(): t = t.replace(k, v)
t = t.replace('\\|', '&#124;')
body = markdown.markdown(t, extensions=['tables', 'fenced_code', 'sane_lists'])
html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><style>
@page {{ size: A4; margin: 16mm 15mm 16mm 15mm;
  @bottom-center {{ content: counter(page) " / " counter(pages); font-size: 8pt; color: #888; }} }}
body {{ font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif; font-size: 9.5pt; line-height: 1.55; color: #222; }}
h1 {{ font-size: 18pt; color: #1a2b4c; border-bottom: 2px solid #1a2b4c; padding-bottom: 2mm; margin-top: 0; }}
h2 {{ font-size: 12.5pt; color: #1a2b4c; margin-top: 6mm; border-left: 4px solid #c0392b; padding-left: 2.5mm; page-break-after: avoid; }}
table {{ border-collapse: collapse; width: 100%; margin: 2.5mm 0 4mm; font-size: 8.3pt; }}
tr {{ page-break-inside: avoid; }}
th {{ background: #1a2b4c; color: #fff; padding: 1.6mm 2mm; text-align: left; font-weight: 600; }}
td {{ border-bottom: 1px solid #ddd; padding: 1.4mm 2mm; vertical-align: top; }}
tr:nth-child(even) td {{ background: #f6f7f9; }}
blockquote {{ margin: 3mm 0; padding: 2.5mm 4mm; background: #f3f5f8; border-left: 3px solid #c0392b; color: #333; font-size: 10.5pt; }}
blockquote p {{ margin: 1mm 0; }}
p strong:first-child {{ color: #c0392b; }}
code {{ font-family: 'DejaVu Sans Mono', monospace; font-size: 8pt; background: #f1f1f1; padding: 0 1mm; }}
img {{ max-width: 100%; display: block; margin: 3mm auto; page-break-inside: avoid; }}
hr {{ border: none; border-top: 1px solid #ccc; margin: 4mm 0; }}
.date {{ color: #777; font-size: 8.5pt; text-align: right; }}
</style></head><body><div class="date">{datetime.date.today():%Y-%m-%d}</div>{body}</body></html>"""
HTML(string=html, base_url='reports/').write_pdf('reports/lg5_9월이후_인사이트_보고서.pdf')
print('ok')
