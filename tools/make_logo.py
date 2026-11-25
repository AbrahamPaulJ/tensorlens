"""Draw docs/logo.png (16:9): a small model graph seen through a lens, next to the name."""
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H, S = 1280, 720, 2  # draw at 2x, downsample for smooth edges
BG = (17, 24, 39)
INK = (229, 231, 235)
ACCENT = (56, 189, 248)
WARM = (251, 146, 60)
DIM = (75, 85, 99)

img = Image.new("RGB", (W * S, H * S), BG)
d = ImageDraw.Draw(img)


def p(x, y):
    return (x * S, y * S)


# graph: three layers of nodes
cx, cy, r = 330, 360, 210
layers = [[-110, 0, 110], [-150, -50, 50, 150], [-60, 60]]
xs = [cx - 120, cx, cx + 120]
nodes = [[(xs[i], cy + dy) for dy in layer] for i, layer in enumerate(layers)]
for a, b in zip(nodes, nodes[1:]):
    for n1 in a:
        for n2 in b:
            inside = math.dist(n1, (cx, cy)) < r and math.dist(n2, (cx, cy)) < r
            d.line([p(*n1), p(*n2)], fill=ACCENT if inside else DIM, width=3 * S)
for layer in nodes:
    for (x, y) in layer:
        rr = 16
        d.ellipse([p(x - rr, y - rr), p(x + rr, y + rr)], fill=BG, outline=INK, width=4 * S)
# the highlighted node: the one whose kernel we jump to
hx, hy = nodes[1][1]
d.ellipse([p(hx - 16, hy - 16), p(hx + 16, hy + 16)], fill=WARM)
# lens
d.ellipse([p(cx - r, cy - r), p(cx + r, cy + r)], outline=INK, width=10 * S)
a = math.radians(45)
hx0, hy0 = cx + r * math.cos(a), cy + r * math.sin(a)
d.line([p(hx0, hy0), p(hx0 + 110, hy0 + 110)], fill=INK, width=26 * S)

font_dir = Path("C:/Windows/Fonts")
big = ImageFont.truetype(str(font_dir / "bahnschrift.ttf"), 118 * S)
small = ImageFont.truetype(str(font_dir / "consola.ttf"), 34 * S)
d.text(p(620, 250), "TensorLens", font=big, fill=INK)
d.text(p(626, 410), "ML models and runtimes,", font=small, fill=(156, 163, 175))
d.text(p(626, 456), "inside the binary, in IDA", font=small, fill=(156, 163, 175))

out = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/logo.png")
out.parent.mkdir(parents=True, exist_ok=True)
img.resize((W, H), Image.LANCZOS).save(out, optimize=True)
print(out, out.stat().st_size)
