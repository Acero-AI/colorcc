# Results

The numbers behind [*Every colour at once*](https://blogs.fablnx.com/posts/every-colour/), the post that explains colorcc, as tables. Every number in the
post and in the README comes from a file in this folder, and the tables below are printed from those files by
[`bench/tables.py`](../bench/tables.py).

> [!TIP]
> The post shows the same data as live charts, with the story of each idea:
> [chapter 8, making it fast](https://blogs.fablnx.com/posts/every-colour/#c8) and [chapter 9, results](https://blogs.fablnx.com/posts/every-colour/#c9).

## The headline, exactly

On an RTX 3070, with the picture already on the GPU:

- The full 806-megapixel sheet takes **89 ms** (706 ms counting the copies in and out).
- **1.7x faster than NVIDIA cuCIM** on the full sheet, rising to 5.4x on a 3-megapixel render, where cuCIM's
  fixed per-call cost dominates. Same regions, same GPU memory (cuCIM also needs a 32-bit copy of a colour picture first; colorcc reads RGB).
- **NVIDIA NPP** gave wrong labels at every size, so it is not a reference (it also used twice the memory).
- **NVIDIA NPP** is 2.5x slower on the 3-megapixel render and 24x slower at 454 megapixels.
- NVIDIA CV-CUDA's `label` joins only 4 neighbours, which splits diagonal lines, so it is not compared.

On the Apple M5 Pro the sheet takes 151 ms, 17x faster than cc3d on one CPU core of the same Mac (2.59 s).

## The tables

### The full sheet: 800 dpi, 806 million pixels, paper labelled

| GPU | time | with copies in and out |
|---|---|---|
| Apple M5 Pro | 150.9 ms | 232.4 ms |
| RTX 3070 | 89.0 ms | 706.2 ms |
| A100 PCIe | 48.8 ms | 569.8 ms |

### Each idea, 800 dpi, Apple M5 Pro

|  | time | joins |
|---|---|---|
| plain, first three stages | 715.3 ms | 3.21 billion |
| + warp vote (runs) | 205.0 ms | 2.43 billion |
| + skip joins | 302.6 ms | 0.81 billion |
| + path halving | 207.3 ms | 3.21 billion |
| all three | 107.3 ms | 0.03 billion |
| + numbering (the full call) | 150.9 ms |  |

### Each idea, 800 dpi, RTX 3070

|  | time | joins |
|---|---|---|
| plain, first three stages | 143.9 ms | 3.21 billion |
| + warp vote (runs) | 120.6 ms | 2.43 billion |
| + skip joins | 199.3 ms | 0.81 billion |
| + path halving | 89.9 ms | 3.21 billion |
| all three | 65.8 ms | 0.03 billion |
| + numbering (the full call) | 89.0 ms |  |

### Each idea, 800 dpi, A100 PCIe

|  | time | joins |
|---|---|---|
| plain, first three stages | 90.4 ms | 3.21 billion |
| + warp vote (runs) | 69.3 ms | 2.43 billion |
| + skip joins | 89.3 ms | 0.81 billion |
| + path halving | 67.9 ms | 3.21 billion |
| all three | 39.5 ms | 0.03 billion |
| + numbering (the full call) | 48.8 ms |  |

### Against NVIDIA's libraries, RTX 3070

| dpi | MP | ours | cuCIM | cuCIM / ours | NPP (wrong labels) | NPP / ours | memory: ours | cuCIM | NPP |
|---|---|---|---|---|---|---|---|---|---|
| 50 | 3 | 0.50 ms | 2.73 ms | 5.4x | 1.26 ms | 2.5x | 12.7 MB | 13.2 MB | 25.2 MB |
| 100 | 13 | 1.60 ms | 6.83 ms | 4.3x | 6.73 ms | 4.2x | 50.6 MB | 51.3 MB | 100.8 MB |
| 200 | 50 | 5.61 ms | 15.7 ms | 2.8x | 43.0 ms | 7.7x | 202.4 MB | 203.1 MB | 403.2 MB |
| 300 | 113 | 12.4 ms | 28.8 ms | 2.3x | 139.9 ms | 11.3x | 455.4 MB | 455.7 MB | 907.2 MB |
| 400 | 202 | 22.6 ms | 47.7 ms | 2.1x | 332.5 ms | 14.7x | 810.6 MB | 810.2 MB | 1.61 GB |
| 600 | 454 | 49.6 ms | 91.9 ms | 1.9x | 1.18 s | 23.7x | 1.82 GB | 1.82 GB | 3.63 GB |
| 800 | 806 | 88.9 ms | 151.5 ms | 1.7x | – | – | 3.24 GB | 3.23 GB | – |

### Against NVIDIA's libraries, A100 PCIe (earlier benchmark; small-size ratios approximate)

| dpi | MP | ours | cuCIM | cuCIM / ours | NPP (wrong labels) | NPP / ours | memory: ours | cuCIM | NPP |
|---|---|---|---|---|---|---|---|---|---|
| 50 | 3 | 0.21 ms | 5.65 ms | 26.6x | 1.17 ms | 5.5x | 12.7 MB | 13.2 MB | 25.2 MB |
| 100 | 13 | 1.20 ms | 12.7 ms | 10.6x | 5.92 ms | 4.9x | 50.6 MB | 51.3 MB | 100.8 MB |
| 200 | 50 | 4.40 ms | 34.8 ms | 7.9x | 30.5 ms | 6.9x | 202.4 MB | 203.1 MB | 403.2 MB |
| 300 | 113 | 10.4 ms | 59.8 ms | 5.8x | 55.0 ms | 5.3x | 455.4 MB | 455.7 MB | 907.2 MB |
| 400 | 202 | 18.5 ms | 69.4 ms | 3.7x | 96.0 ms | 5.2x | 810.6 MB | 810.2 MB | 1.61 GB |
| 600 | 454 | 39.1 ms | 86.3 ms | 2.2x | 292.6 ms | 7.5x | 1.82 GB | 1.82 GB | 3.63 GB |
| 800 | 806 | 48.9 ms | 133.2 ms | 2.7x | – | – | 3.24 GB | 3.23 GB | – |

## How it was measured

Every time is the median wall-clock time between two GPU synchronisations, after a warm-up, with the picture
already on the GPU unless a column says copies are included. `bench/bench.py` (the sweep: the first two kinds of
table) takes the median of 10 calls, 5 at 600 and 800 dpi; cc3d's, 5, and 2 above 60 megapixels.
The "each idea" rows time the first three stages; numbering, the fourth, is added on the last row. Every
timing comes from one synthetic sheet (`bench/synth_sheet.py`), mostly paper; pictures with less background will
behave differently. cc3d's times in `rtx3070/sheet.json` (50 s at 800 dpi) come from the rented box's shared server
CPU and are not quoted; the cc3d reference is one core of the M5 Pro (2.59 s).
`bench/nvidia.py` (the library comparison) runs each library in its own process, warms it up for at least half a
second so the GPU is at its working clock, and takes the median of 5 calls (3 at 600 and 800 dpi). NPP and cuCIM
get the picture already packed into one 32-bit number per pixel before the clock starts, the RGB picture
is then freed, and NPP's stream context is made once: each library is timed on its own work. Memory is the peak
GPU allocation of one more call, beyond each method's input. Copies to and from the GPU depend on the machine's PCIe link, about 10 GB/s on the RTX 3070 here.

## The machines and the files

| folder | machine | what |
|---|---|---|
| `rtx3070/` | RTX 3070 8 GB, driver 580, CUDA 13.0, PyTorch 2.14.1, CuPy 14.2.0, NPP 13.0.1, cuCIM 26.10.0 | the published wheel, installed fresh: `tests.txt` (285 passed), `sheet.json` (sweep), `nvidia.json` (comparison), `e2e.json` (copies), `freeze.txt` (every package version); `warmup-check/`: the same comparison with a one-call warm-up, which left the GPU below its working clock at the smallest sizes |
| `m5/` | Apple M5 Pro | `m5pro_sheet.json` (sweep), `e2e.json`, `tests.txt` (285 passed, colorcc 0.1.1 installed from PyPI) |
| `a100-pcie/` | A100 PCIe 40 GB, driver 595, CUDA 13.2 | `sheet.json`, `e2e.json`, `nvidia.json`, `tests.txt`, from an earlier version of the package and of `nvidia.py` (which timed cuCIM through PyTorch's allocator); on one RTX 3070 that benchmark version's cuCIM ratios were within 1% at 200 to 600 dpi but up to 20% high at 50 and 100 dpi, so the A100's small-size ratios are approximate |
| `rtx3070-profile/` | RTX 3070 | `layouts.json`: the join-order profile (`bench/profile.py`) behind "joining the row above first is about 10x faster" |
| `earlier/` | two earlier RTX 3070 runs | before the pure-Python launcher and the fixed comparison; kept for the record |
