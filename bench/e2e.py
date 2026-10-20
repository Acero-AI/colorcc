"""Copies in and out: a fresh host buffer per call vs one reused (pinned on CUDA) buffer.

    python bench/e2e.py cache/sheet.pdf.*.npy --out e2e.json

naive   torch.from_numpy(img).to(gpu), label, .cpu()   (allocates new host memory each call)
reuse   copy into a GPU buffer, label, copy into a host buffer allocated once
        (pinned on CUDA, so copies run at full PCIe speed and can overlap)
"""
import argparse
import json
import os
import statistics
import sys
import time

import numpy as np
import torch

import colorcc

DEV = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else sys.exit("needs an NVIDIA (CUDA) or Apple (MPS) GPU")
sync = torch.cuda.synchronize if DEV == "cuda" else torch.mps.synchronize


def clock(fn, reps):
    fn()
    ts = []
    for _ in range(reps):
        sync(); t = time.perf_counter(); fn(); sync(); ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npys", nargs="+")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default="e2e.json")
    a = ap.parse_args()
    rows = []
    for npy in a.npys:
        img = np.load(npy)
        H, W = img.shape[:2]
        pin = DEV == "cuda"

        def naive():
            return colorcc.label(torch.from_numpy(img).to(DEV))[0].cpu()
        t_naive = clock(naive, a.reps)  # before the reused buffers exist, so both fit on an 8 GB GPU
        hin = torch.from_numpy(img)
        if pin:
            hin = hin.pin_memory()
        hout = torch.empty((H, W), dtype=torch.int32, pin_memory=pin)
        dimg = torch.empty((H, W, 3), dtype=torch.uint8, device=DEV)

        def reuse():
            dimg.copy_(hin, non_blocking=pin)
            lab, _ = colorcc.label(dimg)
            hout.copy_(lab, non_blocking=pin)

        def kernels():
            colorcc.label(dimg)

        row = {"file": os.path.basename(npy), "H": H, "W": W, "mpx": round(H * W / 1e6, 2),
               "naive": t_naive, "reuse": clock(reuse, a.reps), "kernels": clock(kernels, a.reps)}
        rows.append(row)
        print(json.dumps(row), flush=True)
        json.dump({"device": DEV, "rows": rows}, open(a.out, "w"), indent=1)
        del hin, hout, dimg


if __name__ == "__main__":
    main()
