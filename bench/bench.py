"""Time and memory of exact-colour connected components on a drawing rendered at several resolutions.

    python bench/bench.py drawing.pdf --dpi 50 100 200 300 400 600 800 --out results.json
    python bench/bench.py scan.png --out results.json     # one image as it is

Methods
  loop    one cv2.connectedComponents call per distinct colour (the old way)
  cc3d    all colours in one CPU pass (cc3d, single thread)
  gpu:*   this package on the GPU, one row per switch setting
Times are medians after a warm-up. 'gpu:<variant>' is init + merge + flatten
(labels are root indices); 'gpu:stage<k>' stops after stage k with the default
switches; 'gpu:all+n' adds numbering 1..n; 'gpu:e2e' adds the copy in from
host memory and the labels back out.

Memory
  mem:<cpu method>  peak resident memory of a fresh process running it,
                    minus the same process with only the image loaded
  mem:gpu           GPU memory one label() call allocates beyond the image
  mem:gpu_e2e       the same plus the image itself (what a GPU must hold)
CPU baselines run in their own processes (python bench.py --cpu-job ...), so
their peak memory is theirs alone; renders are cached as .npy in --cache.
"""
import argparse
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

import colorcc
from colorcc.reference import label_cc3d, label_per_colour, pack

DEV = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else sys.exit("needs an NVIDIA (CUDA) or Apple (MPS) GPU")
sync = torch.cuda.synchronize if DEV == "cuda" else torch.mps.synchronize
WHITE = (255, 255, 255)
SWITCHES = {"runs": colorcc.RUNS, "skip": colorcc.SKIP, "packed": colorcc.PACKED, "halve": colorcc.HALVE}
# 'base' is the plain per-pixel union-find; '+x' turns one switch on,
# 'all-x' turns one off, so each switch is measured alone and in company.
VARIANTS = {"base": 0, **{"+" + k: f for k, f in SWITCHES.items()},
            "all": colorcc.ALL, **{"all-" + k: colorcc.ALL & ~f for k, f in SWITCHES.items()}}


def maxrss():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r if sys.platform == "darwin" else r * 1024  # macOS reports bytes, Linux KiB


def render(path, page, dpi):
    if not path.lower().endswith(".pdf"):
        import cv2
        return np.ascontiguousarray(cv2.imread(path, cv2.IMREAD_COLOR)[..., ::-1])
    import pypdfium2 as pdfium
    bmp = pdfium.PdfDocument(path)[page].render(scale=dpi / 72)
    return np.ascontiguousarray(bmp.to_numpy()[..., :3][..., ::-1])  # BGR -> RGB


def clock(fn, reps, gpu=False):
    fn()
    ts = []
    for _ in range(reps):
        if gpu:
            sync()
        t = time.perf_counter()
        fn()
        if gpu:
            sync()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def cpu_job(npy, method, conn, ignore, reps, save):
    """Run one CPU baseline in this (fresh) process; print time and memory as JSON."""
    img = np.load(npy)
    base = maxrss()
    fn = {"cc3d": label_cc3d, "loop": label_per_colour}[method]
    ig = WHITE if ignore else None
    out, n = fn(img, conn, ig)  # warm-up, and the answer
    peak = maxrss()
    if save:
        np.save(save, out)
    del out
    t = clock(lambda: fn(img, conn, ig), reps) if reps else None
    print(json.dumps({"time": t, "n": n, "mem": peak - base, "mem_total": peak}))


def run_cpu(npy, method, conn, ignore, reps, save=""):
    cmd = [sys.executable, os.path.abspath(__file__), "--cpu-job", npy, method, str(conn), str(int(ignore)), str(reps), save]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-2000:])
    return json.loads(r.stdout.strip().splitlines()[-1])


def joins(img, conn, ignore, flags, band=2048):
    """How many union calls the merge kernel makes, counted with NumPy one band of rows at a time."""
    H, W = img.shape[:2]
    ig = None if ignore is None else int(pack(np.array(ignore, np.uint8)[None, None])[0, 0])
    x = np.arange(W)[None, :]
    total = 0
    for y0 in range(0, H, band):
        lo = max(0, y0 - 1)
        k = pack(img[lo:y0 + band])
        ok = np.ones(k.shape, bool) if ig is None else k != ig
        left = np.zeros_like(ok); left[:, 1:] = k[:, 1:] == k[:, :-1]
        left &= ok
        u = np.zeros_like(ok); u[1:] = k[1:] == k[:-1]
        ul = np.zeros_like(ok); ul[1:, 1:] = k[1:, 1:] == k[:-1, :-1]
        ur = np.zeros_like(ok); ur[1:, :-1] = k[1:, :-1] == k[:-1, 1:]
        u, ul, ur = u & ok, ul & ok, ur & ok
        s = slice(y0 - lo, None)  # count only this band's own rows
        left, u, ul, ur = left[s], u[s], ul[s], ur[s]
        total += int((left & ((x % 32 == 0) if flags & colorcc.RUNS else True)).sum())
        skip = flags & colorcc.SKIP
        if conn == 4:
            total += int((u & ~(left & ul) if skip else u).sum())
        elif not skip:
            total += int(ul.sum() + u.sum() + ur.sum())
        else:
            total += int((left & ur & ~u).sum() + (~left & u).sum() + (~left & ~u & ul).sum() + (~left & ~u & ur).sum())
    return total


def gpu_mem(fn):
    """Peak GPU memory a call allocates, in bytes. CUDA keeps a peak counter.
    MPS does not, so we read the allocated total after every torch.empty /
    zeros / tensor the call makes (every buffer label() owns comes from one)."""
    sync()
    if DEV == "cuda":
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        r = fn(); sync()
        return torch.cuda.max_memory_allocated() - base
    now = torch.mps.current_allocated_memory
    base = now(); peak = [base]
    orig = {k: getattr(torch, k) for k in ("empty", "zeros", "tensor")}

    def watch(f):
        def g(*a, **kw):
            out = f(*a, **kw)
            peak[0] = max(peak[0], now())
            return out
        return g
    for k, f in orig.items():
        setattr(torch, k, watch(f))
    try:
        r = fn(); sync()
        peak[0] = max(peak[0], now())
    finally:
        for k, f in orig.items():
            setattr(torch, k, f)
    del r
    return peak[0] - base


def same_labels(got, saved, band=4096):
    ref = np.load(saved, mmap_mode="r")
    return all(np.array_equal(got[y:y + band], ref[y:y + band]) for y in range(0, got.shape[0], band))


def run(npy, conn, ignore, reps, loop_max_mpx, tmp, count_joins=True):
    img = np.load(npy)
    H, W = img.shape[:2]
    mpx = H * W / 1e6
    row = {"H": H, "W": W, "mpx": round(mpx, 2), "colours": int(len(np.unique(pack(img[::2, ::2]))))}
    cpu_reps = 5 if mpx < 60 else 2
    saved = os.path.join(tmp, "cc3d_labels.npy")
    c = run_cpu(npy, "cc3d", conn, ignore, cpu_reps, saved)
    row.update({"cc3d": c["time"], "components": c["n"], "mem:cc3d": c["mem"], "mem:cc3d_total": c["mem_total"]})
    if mpx <= loop_max_mpx:
        c = run_cpu(npy, "loop", conn, ignore, 1)
        row.update({"loop": c["time"], "mem:loop": c["mem"]})
    ig = WHITE if ignore else None
    t = torch.from_numpy(img).to(DEV)
    counted = {}  # only RUNS and SKIP change how many joins merge makes
    for name, f in VARIANTS.items():
        try:
            row["gpu:" + name] = clock(lambda: colorcc.label(t, conn, ig, f, stages=3), reps, gpu=True)
        except torch.OutOfMemoryError:  # PACKED needs 4 more bytes per pixel; at 800 dpi that is past 8 GB
            row["gpu:" + name] = None
            torch.cuda.empty_cache()
        key = f & (colorcc.RUNS | colorcc.SKIP)
        if count_joins:
            row["joins:" + name] = counted[key] = counted.get(key) or joins(img, conn, ig, f)
    for k in (1, 2, 3):
        row[f"gpu:stage{k}"] = clock(lambda: colorcc.label(t, conn, ig, stages=k), reps, gpu=True)
    row["gpu:all+n"] = clock(lambda: colorcc.label(t, conn, ig), reps, gpu=True)
    row["mem:gpu"] = gpu_mem(lambda: colorcc.label(t, conn, ig))
    try:
        row["mem:gpu_packed"] = gpu_mem(lambda: colorcc.label(t, conn, ig, colorcc.ALL))
    except torch.OutOfMemoryError:
        row["mem:gpu_packed"] = None
        torch.cuda.empty_cache()
    row["mem:image"] = img.nbytes
    del t
    row["gpu:e2e"] = clock(lambda: colorcc.label(torch.from_numpy(img).to(DEV), conn, ig)[0].cpu(), reps, gpu=True)
    row["mem:gpu_e2e"] = gpu_mem(lambda: colorcc.label(torch.from_numpy(img).to(DEV), conn, ig))
    got, n = colorcc.label(torch.from_numpy(img).to(DEV), conn, ig)
    got = got.cpu().numpy()
    row["identical_to_cc3d"] = bool(n == row["components"] and same_labels(got, saved))
    os.remove(saved)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", help="a PDF to render at each --dpi, or an image file used as it is")
    ap.add_argument("--page", type=int, default=0)
    ap.add_argument("--dpi", type=int, nargs="+", default=[50, 100, 200, 300, 400, 600, 800])
    ap.add_argument("--conn", type=int, default=8)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--loop-max-mpx", type=float, default=13)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--no-joins", action="store_true", help="skip counting joins (CPU-only, slow at 800 dpi)")
    a = ap.parse_args()
    os.makedirs(a.cache, exist_ok=True)
    dev = torch.cuda.get_device_name() if DEV == "cuda" else platform.processor() + " GPU (MPS)"
    res = {"device": dev, "torch": torch.__version__, "conn": a.conn, "pdf": a.pdf, "rows": []}
    for dpi in a.dpi if a.pdf.lower().endswith(".pdf") else [None]:
        npy = os.path.join(a.cache, f"{os.path.basename(a.pdf)}.{dpi}.npy")
        if not os.path.exists(npy):
            np.save(npy, render(a.pdf, a.page, dpi))
        reps = a.reps if os.path.getsize(npy) < 1e9 else max(3, a.reps // 2)
        for ignore in (False, True):
            row = {"dpi": dpi, "ignore_white": ignore, **run(npy, a.conn, ignore, reps, a.loop_max_mpx, a.cache, not a.no_joins)}
            res["rows"].append(row)
            print(json.dumps(row), flush=True)
            json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--cpu-job":
        npy, method, conn, ignore, reps = sys.argv[2:7]
        cpu_job(npy, method, int(conn), int(ignore), int(reps), sys.argv[7] if len(sys.argv) > 7 else "")
    else:
        main()
