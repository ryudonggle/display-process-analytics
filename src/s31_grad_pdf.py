"""31단계: 졸업작품 보고서 A · B 마크다운 → PDF (s18 스타일, 표지·A4)"""
import re, markdown, datetime, sys
from weasyprint import HTML

DOCS = [('reports/project_불량예측.md', 'reports/프로젝트_불량예측과원인규명.pdf')]
CSS = """
@page { size: A4; margin: 17mm 16mm 17mm 16mm; @bottom-center { content: counter(page) " / " counter(pages); font-size: 8pt; color: #888; } }
body { font-family: 'Noto Sans KR', sans-serif; font-size: 10pt; line-height: 1.65; color: #222; }
h1 { font-size: 24pt; color: #1a2b4c; margin: 0 0 1mm; border: none; }
h1 + h2 { font-size: 13pt; color: #555; border: none; padding: 0; margin: 0 0 5mm; font-weight: 500; }
h2 { font-size: 14pt; color: #1a2b4c; margin-top: 8mm; border-left: 5px solid #c0392b; padding-left: 3mm; page-break-after: avoid; }
h3 { font-size: 11.5pt; color: #333; margin-top: 5mm; page-break-after: avoid; }
table { border-collapse: collapse; width: 100%; margin: 2.5mm 0 4mm; font-size: 8.8pt; }
tr { page-break-inside: avoid; }
th { background: #1a2b4c; color: #fff; padding: 1.8mm 2.2mm; text-align: left; font-weight: 600; }
td { border-bottom: 1px solid #ddd; padding: 1.6mm 2.2mm; vertical-align: top; }
td:first-child, th:first-child { min-width: 22mm; }
tr:nth-child(even) td { background: #f6f7f9; }
blockquote { margin: 3mm 0; padding: 3mm 4.5mm; background: #f3f5f8; border-left: 4px solid #c0392b; color: #222; font-size: 10.5pt; }
blockquote p { margin: 1mm 0; }
img { max-width: 100%; display: block; margin: 3mm auto 4mm; page-break-inside: avoid; }
figure { margin: 3mm 0 5mm; page-break-inside: avoid; }
figure img { margin: 0 auto 1.5mm; }
figcaption { font-size: 8.5pt; color: #666; text-align: center; }
hr { border: none; border-top: 1px solid #ddd; margin: 5mm 0; }
code { font-size: 8.5pt; background: #f1f1f1; padding: 0 1mm; }
"""
def fix_lists(t):
    """문단 바로 다음 줄의 목록 앞에 빈 줄 삽입 (python-markdown은 빈 줄이 있어야 목록으로 인식)"""
    out, prev = [], ''
    for line in t.split('\n'):
        is_item = bool(re.match(r'^\s*([-*]|\d+\.)\s', line))
        prev_item = bool(re.match(r'^\s*([-*]|\d+\.)\s', prev))
        if is_item and prev.strip() and not prev_item and not prev.lstrip().startswith(('|', '>', '#')): out.append('')
        out.append(line); prev = line
    return '\n'.join(out)


for src, dst in DOCS:
    t = fix_lists(open(src, encoding='utf-8').read()).replace('⭐', '★').replace('\\|', '&#124;')
    body = markdown.markdown(t, extensions=['tables', 'sane_lists'])
    # 그림 바로 다음 기울임 문단 = 캡션 → 한 덩어리로 묶어 페이지가 갈라지지 않게
    body = re.sub(r'<p>(<img [^>]+>)</p>\s*<p><em>(.+?)</em></p>', r'<figure>\1<figcaption>\2</figcaption></figure>', body)
    HTML(string=f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><style>{CSS}</style></head><body>{body}'
                f'<p style="color:#888;font-size:8pt;text-align:right">작성 {datetime.date.today():%Y-%m-%d}</p></body></html>',
         base_url='reports/').write_pdf(dst)
    print(dst)
