"""그림용 한글 폰트: 설치된 Noto Sans KR 가변 폰트에서 Regular(400) · Bold(700) 고정 굵기 폰트를 만든다.
matplotlib은 가변 폰트의 굵기를 고르지 못해 Thin으로만 그리기 때문 (30 · 34 ~ 39단계 그림에서 사용)
사용: python src/make_fonts.py [가변 폰트 경로]   →   fonts/NotoSansKR-Regular.ttf, fonts/NotoSansKR-Bold.ttf
Noto Sans KR은 SIL Open Font License.
"""
import sys
from pathlib import Path
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

src = sys.argv[1] if len(sys.argv) > 1 else str(Path.home() / '.local/share/fonts/NotoSansKR.ttf')
Path('fonts').mkdir(exist_ok=True)
for w, nm in [(400, 'Regular'), (700, 'Bold')]:
    inst = instantiateVariableFont(TTFont(src), {'wght': w})
    for rec in inst['name'].names:
        if rec.nameID in (1, 16): rec.string = 'Noto Sans KR Static'
        if rec.nameID in (2, 17): rec.string = nm
        if rec.nameID == 4: rec.string = f'Noto Sans KR Static {nm}'
        if rec.nameID == 6: rec.string = f'NotoSansKRStatic-{nm}'
    inst['OS/2'].usWeightClass = w
    if nm == 'Bold':
        inst['OS/2'].fsSelection = (inst['OS/2'].fsSelection | 0x20) & ~0x40; inst['head'].macStyle |= 1
    inst.save(f'fonts/NotoSansKR-{nm}.ttf'); print(f'fonts/NotoSansKR-{nm}.ttf')
