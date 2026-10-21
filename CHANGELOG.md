# Changelog

## 0.1.2 (2026-10-21)

- README and `results/`: a rerun on an A100 SXM4 with 0.1.1 from PyPI (49 ms for 806 megapixels; cuCIM 2.7x to 19x
  slower, NPP 2.7x to 11x slower and wrong) leads the benchmarks; the 4-neighbour CV-CUDA comparison is dropped. No code changes.

## 0.1.1 (2026-10-20)

- The README renders on PyPI too: images and links point at GitHub. No code changes.

## 0.1.0 (2026-10-20)

First release.

- `colorcc.label(img, conn=8, ignore=None)`: exact-colour connected components of an `[H, W, 3]` `uint8`
  PyTorch tensor, labels 1..n in raster order (identical to `cc3d`), on NVIDIA (CUDA) and Apple (Metal) GPUs.
- One pure-Python wheel: the CUDA kernels compile at first use with NVRTC (through CuPy) for the GPU present,
  the Metal kernels with PyTorch. `pip install "colorcc[cuda12]"`, `"colorcc[cuda13]"`, or `colorcc` on a Mac.
- Benchmarks against NVIDIA cuCIM, NPP and CV-CUDA, with every result file: `bench/`, `results/`.
