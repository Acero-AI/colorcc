"""Label an RGB image and save the regions in random colours.

    python examples/label_png.py drawing.png labels.png
"""
import sys

import numpy as np
import torch
from PIL import Image

import colorcc

src, dst = sys.argv[1], sys.argv[2]
dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else sys.exit("needs an NVIDIA (CUDA) or Apple (MPS) GPU")
rgb = np.array(Image.open(src).convert("RGB"))
labels, n = colorcc.label(torch.from_numpy(rgb).to(dev), conn=8, ignore=(255, 255, 255))
print(f"{n} regions")
colours = np.random.default_rng(0).integers(0, 256, (n + 1, 3), dtype=np.uint8)
colours[0] = 255  # ignored pixels (label 0) stay white
Image.fromarray(colours[labels.cpu().numpy()]).save(dst)
