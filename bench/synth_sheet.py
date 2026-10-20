"""A synthetic 42 x 30-inch drawing sheet, vector PDF, so benchmarks need no client data.

    python bench/synth_sheet.py sheet.pdf

Grid lines with bubbles, walls, hatched areas, door swings, columns, dimension
text and notes, in a handful of pen colours and weights. Fixed seed.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np

rng = np.random.default_rng(42)
W, H = 42, 30  # inches; 1 unit = 1 inch
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
PENS = ["black", "#c00000", "#0050c0", "#008040", "#7a7a7a", "#a05000"]

ax.add_patch(patches.Rectangle((0.5, 0.5), W - 1, H - 1, fill=False, lw=2.5, color="black"))  # border
ax.add_patch(patches.Rectangle((W - 7.5, 0.5), 7, 5, fill=False, lw=1.5, color="black"))  # title block
for i, t in enumerate(["PROJECT  SYNTHETIC TOWER", "SHEET  S-201", "LEVEL 2 FRAMING PLAN", "SCALE 1/8\" = 1'-0\""]):
    ax.text(W - 7.2, 4.8 - i * 1.1, t, fontsize=14, family="monospace")

xs, ys = np.linspace(3, W - 10, 9), np.linspace(3, H - 4, 7)
for k, x in enumerate(xs):  # grid lines with bubbles
    ax.plot([x, x], [1.8, H - 2.2], color="#7a7a7a", lw=0.6, ls=(0, (12, 3, 2, 3)))
    ax.add_patch(patches.Circle((x, H - 1.6), 0.45, fill=False, lw=1, color="black"))
    ax.text(x, H - 1.6, str(k + 1), ha="center", va="center", fontsize=13)
for k, y in enumerate(ys):
    ax.plot([1.8, W - 9.2], [y, y], color="#7a7a7a", lw=0.6, ls=(0, (12, 3, 2, 3)))
    ax.add_patch(patches.Circle((1.3, y), 0.45, fill=False, lw=1, color="black"))
    ax.text(1.3, y, "ABCDEFG"[k], ha="center", va="center", fontsize=13)
for x in xs:  # columns
    for y in ys:
        ax.add_patch(patches.Rectangle((x - 0.12, y - 0.12), 0.24, 0.24, color="black"))
for x0, x1 in zip(xs[:-1], xs[1:]):  # beams along the grid, with size tags
    for y in ys:
        if rng.random() < 0.8:
            ax.plot([x0 + 0.15, x1 - 0.15], [y, y], color="black", lw=2.2)
            ax.text((x0 + x1) / 2, y + 0.12, f"W{rng.choice([12, 14, 16, 18, 21])}x{rng.choice([26, 35, 40, 50, 62])}", ha="center", fontsize=8)
for x in xs:
    for y0, y1 in zip(ys[:-1], ys[1:]):
        if rng.random() < 0.6:
            ax.plot([x, x], [y0 + 0.15, y1 - 0.15], color="black", lw=2.2)
for _ in range(40):  # joists: thin parallel lines in random bays
    i, j = rng.integers(0, len(xs) - 1), rng.integers(0, len(ys) - 1)
    for yy in np.linspace(ys[j] + 0.3, ys[j + 1] - 0.3, 8):
        ax.plot([xs[i] + 0.2, xs[i + 1] - 0.2], [yy, yy], color="#0050c0", lw=0.4)
for _ in range(14):  # hatched areas: thin diagonals, the hard case for 4-neighbour labelling
    i, j = rng.integers(0, len(xs) - 1), rng.integers(0, len(ys) - 1)
    x0, y0, w, h = xs[i] + 0.3, ys[j] + 0.3, (xs[i + 1] - xs[i]) * 0.5, (ys[j + 1] - ys[j]) * 0.5
    ax.add_patch(patches.Rectangle((x0, y0), w, h, fill=False, lw=1, color="#c00000"))
    for d in np.arange(-h, w, 0.18):
        a, b = max(0, d), min(w, d + h)
        if a < b:
            ax.plot([x0 + a, x0 + b], [y0 + a - d, y0 + b - d], color="#c00000", lw=0.3)
for _ in range(30):  # door swings
    x, y, r = rng.uniform(4, W - 12), rng.uniform(4, H - 5), rng.uniform(0.4, 0.9)
    ax.add_patch(patches.Arc((x, y), 2 * r, 2 * r, theta1=0, theta2=90, lw=0.7, color="#008040"))
    ax.plot([x, x + r], [y, y], color="#008040", lw=0.7)
for x0, x1 in zip(xs[:-1], xs[1:]):  # dimension strings
    ax.annotate("", (x0, 2.2), (x1, 2.2), arrowprops=dict(arrowstyle="<->", lw=0.5))
    ax.text((x0 + x1) / 2, 2.35, f"{rng.integers(18, 32)}'-{rng.integers(0, 12)}\"", ha="center", fontsize=9)
for k in range(25):  # general notes
    ax.text(W - 8.8, H - 3 - k * 0.8, f"{k + 1}. " + " ".join(rng.choice(["VERIFY", "ALL", "BEAM", "CONNECTIONS", "PER", "DETAIL", "TYP.", "SEE", "S-501", "U.N.O.", "ELEV.", "TOS"], 6)), fontsize=9)
fig.savefig(sys.argv[1])
print("wrote", sys.argv[1])
