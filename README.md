<div align="center">

# colorcc

**Connected components of every colour at once, on the GPU.**

Exact region labels for RGB images in PyTorch, on NVIDIA and Apple GPUs.

[![PyPI](https://img.shields.io/pypi/v/colorcc)](https://pypi.org/project/colorcc/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/colorcc/)
[![test](https://github.com/Acero-AI/colorcc/actions/workflows/test.yml/badge.svg)](https://github.com/Acero-AI/colorcc/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/Acero-AI/colorcc/blob/main/LICENSE)

**[How it works, with live figures: read the post on the Fablnx engineering blog →](https://blogs.fablnx.com/posts/every-colour/)**

<img src="https://raw.githubusercontent.com/Acero-AI/colorcc/main/docs/hero.png" alt="A drawing and colorcc's labels: every region of every colour" width="100%">

</div>

> **New here?** [*Every colour at once*](https://blogs.fablnx.com/posts/every-colour/) builds this labeller from
> scratch: union-find, warps, atomics, why the white paper of a drawing is the hardest region of all, and how it
> compares with NVIDIA's libraries. Every figure is live, and no GPU background is needed.

## Why

Most connected-component labellers (OpenCV, SciPy, cuCIM's binary mode) see black and white. Real images, and
construction drawings in particular, are made of many exact colours, with anti-aliased edges in between. colorcc
gives every exact colour its own regions, in one pass over the image, with no loop over colours:

<img src="https://raw.githubusercontent.com/Acero-AI/colorcc/main/docs/colour-vs-binary.png" alt="The same pixels: a black-and-white labeller makes all ink one region; colorcc separates each exact colour" width="100%">

Two pixels belong to the same region when they touch (4 or 8 neighbours) and have exactly the same RGB value.
Labels run 1..n in raster order of each region's first pixel, so the output is identical, value for value, to
`cc3d.connected_components` on the packed colours. One colour, such as white paper, can be ignored: it gets
label 0 and joins nothing.

## Install

Install [PyTorch](https://pytorch.org) first, then:

```bash
pip install "colorcc[cuda12]"    # NVIDIA, PyTorch built for CUDA 12
pip install "colorcc[cuda13]"    # NVIDIA, PyTorch built for CUDA 13
pip install colorcc              # Apple silicon
```

Nothing is compiled at install time and no CUDA toolkit is needed: the wheel is pure Python. The kernels are
compiled at the first call for the GPU you have, by NVRTC through CuPy on NVIDIA GPUs and by Metal through PyTorch
on Apple GPUs (PyTorch 2.6 or later). CuPy caches the result, so only the first call in a fresh environment pays
for it. Python 3.10 or later.

## Quickstart

```python
import numpy as np, torch, colorcc
from PIL import Image

rgb = np.array(Image.open("drawing.png").convert("RGB"))
img = torch.from_numpy(rgb).to("cuda")                          # or "mps"
labels, n = colorcc.label(img, conn=8, ignore=(255, 255, 255))  # labels: [H, W] int32, 1..n; paper 0
```

[`examples/label_png.py`](https://github.com/Acero-AI/colorcc/blob/main/examples/label_png.py) does this and saves the regions as a colour-coded PNG
(`pip install pillow`):

```bash
python examples/label_png.py drawing.png labels.png
```

## Benchmarks

On an RTX 3070, with the picture already on the GPU, the full 806-megapixel drawing takes **89 ms**:

| | 3 MP render | 806 MP sheet |
|---|---|---|
| colorcc | 0.50 ms | 88.9 ms |
| NVIDIA cuCIM `label()` | 2.73 ms (**5.4x** slower) | 151 ms (**1.7x** slower) |
| NVIDIA NPP `LabelMarkersUF` | wrong labels, **2x** the memory | buffer-size query overflows |

The same sheet takes 151 ms on an Apple M5 Pro (17x faster than cc3d on one CPU core) and 49 ms on an A100.
cuCIM gives the same regions in the same memory; NPP splits regions at every size it ran. NVIDIA's CV-CUDA only
joins 4 neighbours, and there it is 1.6–1.9x *faster* than colorcc's 4-neighbour mode.

> **Every number**, at every size, with the machines, versions and method: [`results/README.md`](https://github.com/Acero-AI/colorcc/blob/main/results/README.md).
> The same data as live charts, with what each idea is worth: [chapter 9 of the post](https://blogs.fablnx.com/posts/every-colour/#c9).

## How it works

<img src="https://raw.githubusercontent.com/Acero-AI/colorcc/main/docs/four-stages.gif" alt="The four stages, thread by thread, with the CUDA kernel line each thread runs" width="100%">

Four stages, the same CUDA ([`colorcc/cuda/kernels.cu`](https://github.com/Acero-AI/colorcc/blob/main/colorcc/cuda/kernels.cu)) and Metal
([`colorcc/metal/kernels.metal`](https://github.com/Acero-AI/colorcc/blob/main/colorcc/metal/kernels.metal)) code line for line, launched by one Python function
(`_stages` in [`colorcc/__init__.py`](https://github.com/Acero-AI/colorcc/blob/main/colorcc/__init__.py)) on either GPU:

1. **Start** (`init`): each pixel points at a parent. A warp vote finds runs of one colour among its 32 pixels,
   and each pixel points at the first pixel of its run.
2. **Connect** (`merge`): union-find. Each pixel joins its tree to the trees of its same-colour neighbours with an
   atomic minimum, the row above first, then the left neighbour. Joins another thread already makes are skipped,
   and path halving keeps the trees shallow.
3. **Collapse** (`flatten`): each pixel points straight at its leader.
4. **Number** (`rank_roots`, `number`): each leader gets its rank in a 256-pixel chunk; after a prefix sum, each
   pixel takes its leader's number 1..n.

The animation is Fig. 17 of [the post](https://blogs.fablnx.com/posts/every-colour/), where you can step through
it yourself, switch each idea on and off, and see what each one is worth.

## API

```python
colorcc.label(img, conn=8, ignore=None, flags=colorcc.DEFAULT, stages=4) -> (labels, n)
```

| Argument | |
|---|---|
| `img` | `[H, W, 3]` `uint8` tensor on a CUDA or MPS device |
| `conn` | 4 or 8 neighbours |
| `ignore` | an `(r, g, b)` colour that gets label 0 and joins nothing, or `None` |
| `flags` | which optimisations to switch on; see below |
| `stages` | stop early, for timing: 1 init, 2 merge, 3 flatten (labels then hold root pixel indices, -1 for ignored pixels), 4 numbered labels |

Returns `labels`, an `[H, W]` `int32` tensor on the same device, and `n`, the
number of components as a Python int (0 when `stages` is below 4). Bad input
raises `ValueError`; an empty image returns an empty label tensor and 0.

Flags (combine with `|`):

| Flag | |
|---|---|
| `RUNS` | init: a warp ballot finds horizontal runs of one colour, so most left joins disappear |
| `SKIP` | merge: skip joins another thread already makes |
| `PACKED` | init stores each colour as one int; merge reads one word instead of three bytes |
| `HALVE` | merge: path halving keeps the trees shallow |
| `DEFAULT` | `RUNS \| SKIP \| HALVE` |
| `ALL` | all four |

`PACKED` is off by default. Added to the default switches it was slower in 39
of 40 measurements (all 14 on the M5 Pro, all 12 on the RTX 3070, 13 of 14 on
the A100, which was 1.3x faster at 600 dpi with the paper ignored), by up to 38%,
and it doubles the memory. On its own, without the other switches, it was
faster about as often as slower. The flags exist so each idea can be
measured on its own; every combination gives the same labels.

`colorcc.reference` has CPU versions (cc3d, and one OpenCV call per colour)
used as ground truth in the tests and as baselines in the benchmarks.

## Limits

- Images up to 2^31 pixels (labels are 32-bit pixel indices).
- 3-channel `uint8` input only.
- Tested only on the three GPUs above.

## Correctness

[`tests/test_colorcc.py`](https://github.com/Acero-AI/colorcc/blob/main/tests/test_colorcc.py) compares every flag combination, 4 and 8 neighbours, with and
without an ignored colour, against cc3d: the same labels, value for value, on ten test images (noise,
checkerboards, one-pixel diagonals, a spiral, single rows and columns and others), plus a test that labels a large
image many times to catch races. All 285 tests pass on an RTX 3070 against the released wheel ([`results/rtx3070/tests.txt`](https://github.com/Acero-AI/colorcc/blob/main/results/rtx3070/tests.txt)) and on an Apple M5 Pro against colorcc 0.1.1 installed from PyPI ([`results/m5/tests.txt`](https://github.com/Acero-AI/colorcc/blob/main/results/m5/tests.txt)). In every benchmark the labels were also identical to cc3d at every size.

## Development

```bash
git clone https://github.com/Acero-AI/colorcc && cd colorcc
pip install -e ".[cuda12,test]"   # or cuda13; nothing extra on a Mac
pytest tests                      # CUDA if present, else the Apple GPU; with neither, only the input checks run
```

The benchmarks (`pip install -e ".[cuda12,bench]"`):

```bash
python bench/synth_sheet.py sheet.pdf
python bench/bench.py sheet.pdf --out sheet.json                  # every flag, and cc3d, 50-800 dpi
python bench/nvidia.py cache/sheet.pdf.*.npy --out nvidia.json    # vs NPP, cuCIM, CV-CUDA (whichever are installed)
python bench/e2e.py cache/sheet.pdf.*.npy --out e2e.json          # with copies to and from the GPU
python bench/tables.py                                            # the tables in results/README.md
```

`bench.py` caches each render as `.npy` in `--cache` (default `cache/`); an 800 dpi render is 2.4 GB.
`bench/profile.py` times each stage and each library's kernels at one size (it compiles its kernel variants with a
CUDA toolkit), and `bench/check_labels.py` checks a label file against its image on the CPU with SciPy.

Issues and pull requests are welcome. Releases are listed in [`CHANGELOG.md`](https://github.com/Acero-AI/colorcc/blob/main/CHANGELOG.md).

## Citing

If colorcc helps your work, please cite it: GitHub's *Cite this repository* button (from
[`CITATION.cff`](https://github.com/Acero-AI/colorcc/blob/main/CITATION.cff)) gives BibTeX and APA.

## Licence

MIT. See [`LICENSE`](https://github.com/Acero-AI/colorcc/blob/main/LICENSE).

---

Gokul Adethya, [Fablnx](https://fablnx.com). How it works, with live figures:
[the Fablnx engineering blog](https://blogs.fablnx.com/posts/every-colour/).
