# Changelog

## 0.1.0 (2026-10-20)

First release.

- `colorcc.label(img, conn=8, ignore=None)`: exact-colour connected components of an `[H, W, 3]` `uint8`
  PyTorch tensor, labels 1..n in raster order (identical to `cc3d`), on NVIDIA (CUDA) and Apple (Metal) GPUs.
- One pure-Python wheel: the CUDA kernels compile at first use with NVRTC (through CuPy) for the GPU present,
  the Metal kernels with PyTorch. `pip install "colorcc[cuda12]"`, `"colorcc[cuda13]"`, or `colorcc` on a Mac.
- Benchmarks against NVIDIA cuCIM, NPP and CV-CUDA, with every result file: `bench/`, `results/`.
