"""Exact-colour connected components for PyTorch, on CUDA and Apple GPUs.

    labels, n = colorcc.label(img)            # img: [H, W, 3] uint8 tensor
    labels, n = colorcc.label(img, conn=4, ignore=(255, 255, 255))

Pixels that touch and have exactly the same RGB value share a label.
Labels run 1..n in raster order of each component's first pixel (the same
numbering cc3d and OpenCV use); the ignored colour, if any, gets 0.
"""
import functools
import numbers
import pathlib

import torch

__all__ = ["label", "pack", "RUNS", "SKIP", "PACKED", "HALVE", "DEFAULT", "ALL"]

RUNS, SKIP, PACKED, HALVE = 1, 2, 4, 8
ALL = RUNS | SKIP | PACKED | HALVE
DEFAULT = RUNS | SKIP | HALVE  # PACKED measured slower: its extra write costs more than it saves

_KERNELS = pathlib.Path(__file__).parent
_compiled = {}


def pack(rgb):
    """(r, g, b) -> the int the kernels compare: r | g << 8 | b << 16."""
    r, g, b = rgb
    return int(r) | int(g) << 8 | int(b) << 16


def _metal():
    """Metal kernels, compiled by PyTorch at first use. Returns run(name, threads, group, *bufs)."""
    if "mps" not in _compiled:
        if not hasattr(torch.mps, "compile_shader"):
            raise RuntimeError("colorcc on Apple GPUs needs PyTorch >= 2.6 (torch.mps.compile_shader)")
        lib = torch.mps.compile_shader((_KERNELS / "metal" / "kernels.metal").read_text())

        def run(name, threads, group, *args):
            getattr(lib, name)(*args, threads=threads, group_size=group)
        _compiled["mps"] = run
    return _compiled["mps"]


def _cuda(device):
    """CUDA kernels, compiled by NVRTC (via CuPy) for this GPU at first use. Returns run(...)."""
    if device not in _compiled:
        try:
            import cupy
            import numpy
        except ImportError as e:
            raise RuntimeError("colorcc on NVIDIA GPUs needs CuPy: pip install 'colorcc[cuda12]' "
                               "(or [cuda13], to match torch.version.cuda)") from e
        src = (_KERNELS / "cuda" / "kernels.cu").read_text()
        with cupy.cuda.Device(device.index):
            mod = cupy.RawModule(code=src, options=("-std=c++17",))
            names = ("init", "merge", "flatten", "rank_roots", "number")
            kernels = {k: mod.get_function(k) for k in names}

        streams = {}

        def run(name, threads, group, *args):
            blocks = tuple((t + g - 1) // g for t, g in zip(threads, group))
            ts = torch.cuda.current_stream(device)
            if ts.cuda_stream not in streams:
                streams[ts.cuda_stream] = cupy.cuda.Stream.from_external(ts)
            ptrs = tuple(numpy.uint64(a.data_ptr()) for a in args)  # pointers by value: no wrapping per launch
            if cupy.cuda.runtime.getDevice() == device.index:
                kernels[name](blocks, group, ptrs, stream=streams[ts.cuda_stream])
            else:
                with cupy.cuda.Device(device.index):
                    kernels[name](blocks, group, ptrs, stream=streams[ts.cuda_stream])
        _compiled[device] = run
    return _compiled[device]


@functools.lru_cache(maxsize=64)
def _params(device, H, W, conn, flags, ignore):
    """The kernels' parameter buffer, made once per image size and setting rather than copied on every call."""
    return torch.tensor([H, W, conn, flags, ignore], dtype=torch.int32, device=device)


def _stages(run, img, conn, ignore, flags, stages):
    """The same launches on both GPUs; only run() differs."""
    H, W = img.shape[:2]
    i32 = dict(dtype=torch.int32, device=img.device)
    P = _params(img.device, H, W, conn, flags, ignore)
    label = torch.empty((H, W), **i32)
    key = torch.empty((H, W) if flags & PACKED else (1,), **i32)
    n = 0
    # x padded to 32, so a warp (a SIMD-group on Apple) is 32 pixels of one row
    rows, group = ((W + 31) // 32 * 32, H), (32, 8)
    run("init", rows, group, img, key, label, P)
    if stages >= 2:
        run("merge", rows, group, img, key, label, P)
    if stages >= 3:
        run("flatten", rows, group, label, P)
    if stages >= 4:
        chunks = (H * W + 255) // 256
        count = torch.empty(chunks, **i32)
        run("rank_roots", (chunks * 256,), (256,), label, count, P)
        count.cumsum_(0)
        run("number", (chunks * 256,), (256,), label, count, P)
        n = count[-1]
    return label, n


def label(img, conn=8, ignore=None, flags=DEFAULT, stages=4):
    """Label an [H, W, 3] uint8 CUDA or MPS tensor. Returns ([H, W] int32 labels, n).

    conn    4 or 8 neighbours
    ignore  an (r, g, b) colour that gets label 0 and joins nothing, e.g. white paper
    flags   optimisations to switch on (RUNS | SKIP | PACKED | HALVE); DEFAULT
            is all but PACKED, which measured slower
    stages  stop early, for timing: 1 init, 2 merge, 3 flatten (labels then
            hold root pixel indices, -1 for ignored), 4 numbered labels
    """
    if not (isinstance(img, torch.Tensor) and img.dtype == torch.uint8
            and img.dim() == 3 and img.shape[2] == 3):
        raise ValueError("img must be an [H, W, 3] uint8 torch tensor")
    if conn not in (4, 8):
        raise ValueError("conn must be 4 or 8")
    if ignore is not None and not (isinstance(ignore, (tuple, list)) and len(ignore) == 3 and all(
            isinstance(c, numbers.Integral) and 0 <= c <= 255 for c in ignore)):
        raise ValueError("ignore must be None or an (r, g, b) tuple of ints 0..255")
    if not (isinstance(flags, numbers.Integral) and flags & ~ALL == 0):
        raise ValueError("flags must combine RUNS, SKIP, PACKED and HALVE")
    if stages not in (1, 2, 3, 4):
        raise ValueError("stages must be 1, 2, 3 or 4")
    if img.shape[0] * img.shape[1] >= 2**31:
        raise ValueError("image has more than 2^31 pixels")
    if img.numel() == 0:
        return torch.zeros(img.shape[:2], dtype=torch.int32, device=img.device), 0
    img = img.contiguous()
    ig = -1 if ignore is None else pack(ignore)
    if img.device.type == "cuda":
        run = _cuda(img.device)
    elif img.device.type == "mps":
        run = _metal()
    else:
        raise ValueError("img must be on a CUDA or MPS device; colorcc.reference has CPU versions")
    out, n = _stages(run, img, conn, ig, flags, stages)
    return out, int(n)
