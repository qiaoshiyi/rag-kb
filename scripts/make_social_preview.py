"""生成 GitHub 仓库的 Social Preview 图（1280x640）。

    python scripts/make_social_preview.py                       # 输出 assets/social-preview.png
    python scripts/make_social_preview.py shot.png out.png

为什么单独写一个脚本：Social preview 只有上传那一刻用一次，但每次截图更新（换了语料、改了界面）
都得重做一张。脚本里把版式固定下来，重做只要一条命令，不用再打开设计软件。
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
W, H = 1280, 640

INK = (28, 32, 36)
INK2 = (91, 100, 112)
INK3 = (139, 148, 159)
LINE = (228, 231, 236)
ACCENT = (59, 91, 219)
ACCENT_SOFT = (238, 242, 255)
BG = (246, 247, 249)

# 不同系统上找得到的中文字体，取第一个可用的
FONT_CANDIDATES = {
    "regular": [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ],
    "bold": [
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    ],
}


def font(kind: str, size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES[kind]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


def rounded(im: Image.Image, radius: int) -> Image.Image:
    mask = Image.new("L", im.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, *im.size], radius=radius, fill=255)
    out = im.convert("RGBA")
    out.putalpha(mask)
    return out


def build(screenshot: Path, out: Path) -> None:
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)

    # 右上角一团极淡的强调色，避免整张图死平
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(glow).ellipse([820, -260, 1500, 340], fill=(*ACCENT_SOFT, 255))
    canvas = Image.alpha_composite(canvas.convert("RGBA"), glow.filter(ImageFilter.GaussianBlur(90)))
    draw = ImageDraw.Draw(canvas)

    # ---- 左栏 ----
    x = 72
    pill = "RAG · 知识库问答"
    f_pill = font("bold", 18)
    pw = draw.textlength(pill, font=f_pill)
    draw.rounded_rectangle([x, 96, x + pw + 36, 140], radius=22, fill=ACCENT_SOFT)
    draw.text((x + 18, 107), pill, font=f_pill, fill=ACCENT)

    draw.text((x - 4, 170), "rag-kb", font=font("bold", 82), fill=INK)

    f_sub = font("regular", 23)
    draw.text((x, 280), "四类文档解析 · 混合检索 · 多轮改写", font=f_sub, fill=INK2)
    draw.text((x, 318), "带引用作答 · 全链路可量化评测", font=f_sub, fill=INK2)

    # 三个核心数字
    metrics = [("1.000", "检索 hit@5"), ("0.748", "拒答阈值"), ("142", "离线测试")]
    f_num, f_lab = font("bold", 34), font("regular", 15)
    my = 398
    for i, (num, label) in enumerate(metrics):
        cx = x + i * 172
        draw.text((cx, my), num, font=f_num, fill=INK)
        draw.text((cx, my + 46), label, font=f_lab, fill=INK3)

    draw.text((x, 524), "12 份企业制度 · 2.6 万字 · 四类格式", font=font("regular", 16), fill=INK3)

    # ---- 右栏：界面截图 ----
    shot = Image.open(screenshot).convert("RGB")
    cw, ch = 580, round(580 * shot.height / shot.width)
    shot = shot.resize((cw, ch), Image.LANCZOS)
    cx, cy = 636, 96

    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle(
        [cx + 6, cy + 14, cx + cw + 6, cy + ch + 14], radius=16, fill=(16, 24, 40, 55)
    )
    canvas = Image.alpha_composite(canvas, shadow.filter(ImageFilter.GaussianBlur(18)))

    card = rounded(shot, 14)
    canvas.paste(card, (cx, cy), card)
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle([cx, cy, cx + cw, cy + ch], radius=14, outline=LINE, width=2)

    canvas.convert("RGB").save(out, optimize=True)
    print(f"{out}  {out.stat().st_size // 1024} KB  {canvas.width}x{canvas.height}")


if __name__ == "__main__":
    shot = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "assets" / "screenshot.png"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "assets" / "social-preview.png"
    build(shot, out)
