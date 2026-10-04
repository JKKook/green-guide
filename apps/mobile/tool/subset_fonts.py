"""번들 폰트 서브셋 — Pretendard 4종·푸른숲체 Bold 를 KS X 1001 한글 2350자 +
앱 소스에 쓰인 음절 + 라틴·기호로 줄인다 (원본 1.57MB → 약 0.45MB/종).

원본은 upstream 에서 받아 `assets/fonts/_src/` 에 두고 실행한다:
  python3 tool/subset_fonts.py            # fontTools 필요 (pip install fonttools)
2350자 밖의 드문 음절(예: 뷁)은 기기 시스템 글꼴로 글자 단위 폴백된다.
OFL Reserved Font Name 규정에 따라 내부 이름은 'GreenGuide Sans' / 'GreenGuide Display' 로 바꾼다
(pubspec 의 family 키 'Pretendard'/'PureunSup' 는 앱 내부 조회 키라 그대로 둔다).
"""
from __future__ import annotations

import glob
import os
import sys

from fontTools import subset
from fontTools.ttLib import TTFont

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
SRC = os.path.join(APP, "assets", "fonts", "_src")
DST = os.path.join(APP, "assets", "fonts")

FONTS = {  # 파일명 → 내부 패밀리 이름
    "Pretendard-Regular.otf": "GreenGuide Sans",
    "Pretendard-Medium.otf": "GreenGuide Sans",
    "Pretendard-SemiBold.otf": "GreenGuide Sans",
    "Pretendard-Bold.otf": "GreenGuide Sans",
    "PureunSup-Bold.otf": "GreenGuide Display",
}

# 라틴·라틴 확장·구두점·통화·화살표·수학·원문자·도형·기호·CJK 구두점·호환 자모·전각
RANGES = [(0x20, 0x7E), (0xA0, 0x24F), (0x2000, 0x206F), (0x20A0, 0x20CF),
          (0x2100, 0x214F), (0x2190, 0x21FF), (0x2200, 0x22FF), (0x2460, 0x24FF),
          (0x25A0, 0x25FF), (0x2600, 0x26FF), (0x3000, 0x303F), (0x3130, 0x318F),
          (0xFF00, 0xFFEF)]


def ks_x_1001() -> set[int]:
    out = set()
    for c in range(0xAC00, 0xD7A4):
        try:
            chr(c).encode("iso2022_kr")
            out.add(c)
        except UnicodeEncodeError:
            pass
    return out


def app_syllables() -> set[int]:
    out = set()
    for p in glob.glob(os.path.join(APP, "lib", "**", "*.dart"), recursive=True):
        with open(p, encoding="utf-8") as f:
            out.update(ord(ch) for ch in f.read() if 0xAC00 <= ord(ch) <= 0xD7A3)
    return out


def rename(font: TTFont, family: str) -> None:
    name = font["name"]
    for rec in name.names:
        if rec.nameID in (1, 16):
            rec.string = family
        elif rec.nameID == 4:
            rec.string = f"{family} {rec.toUnicode().split(' ')[-1]}"
        elif rec.nameID == 6:
            rec.string = family.replace(" ", "") + "-" + rec.toUnicode().split("-")[-1]


def main() -> int:
    unicodes = sorted(set().union(*[set(range(a, b + 1)) for a, b in RANGES]) | ks_x_1001() | app_syllables())
    for fname, family in FONTS.items():
        src = os.path.join(SRC, fname)
        if not os.path.exists(src):
            print(f"skip (원본 없음): {src}")
            continue
        opts = subset.Options()
        opts.layout_features = ["*"]
        opts.name_IDs = ["*"]
        opts.hinting = False
        font = TTFont(src)
        s = subset.Subsetter(opts)
        s.populate(unicodes=unicodes)
        s.subset(font)
        rename(font, family)
        dst = os.path.join(DST, fname)
        font.save(dst)
        print(f"{fname}: {os.path.getsize(src)//1024}KB → {os.path.getsize(dst)//1024}KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
