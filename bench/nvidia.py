"""colorcc against what NVIDIA GPUs already offer, on the same renders.

    python bench/nvidia.py cache/sheet.pdf.*.npy --out nvidia.json

Methods (each run in its own process, so one running out of GPU memory does
not take the others with it, and peak memory is its own):
  ours       colorcc.label, 8 neighbours, labels 1..n
  ours4      colorcc.label, 4 neighbours (to compare with the old kernel)
  npp        NVIDIA NPP nppiLabelMarkersUF_32u (8 neighbours) + CompressMarkerLabelsUF,
             on colours packed into one uint32 per pixel
  npp_raw    the same without the compress (renumbering) step
  cucim      RAPIDS cuCIM skimage.measure.label(connectivity=2) on packed colours
  cvcuda4    CV-CUDA cvcuda.label, 4 neighbours (it refuses 8), on packed colours
  old        our earlier kernel, cc_torch.cc_2d_color (4 neighbours, root labels;
             not published, so not in the default list)
Time: median of --reps after a warm-up of at least half a second, GPU only (image already on the GPU). The
packed picture that NPP, cuCIM and CV-CUDA need is made before timing, the RGB
picture is then freed, and NPP's stream context is made once, so each library is
timed on its own work with the whole GPU to itself. cuCIM is timed with CuPy's
own memory pool, as a user would call it.
Memory: peak GPU memory of one more call beyond its input (for cuCIM that call
routes CuPy's allocations through PyTorch's allocator so they are counted;
CV-CUDA allocates outside it and is not counted). A colour picture also needs the packed copy
(4 bytes per pixel) before NPP, cuCIM or CV-CUDA can read it; ours reads RGB.
Agreement: every method's grouping is compared with ours (same connectivity)
on the CPU: each of our labels maps to exactly one of theirs at every pixel (no
region split) and both have the same number of labels (no regions merged). When
our own job failed, the row records that agreement was not checked.
"""
import argparse
import itertools
import json
import os
import statistics
import subprocess
import sys
import time

import numpy as np

NPP_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <nppcore.h>
#include <nppi_filtering_functions.h>

static NppStreamContext make_context() {
    NppStreamContext c{};
    c.hStream = c10::cuda::getCurrentCUDAStream();
    cudaGetDevice(&c.nCudaDeviceId);
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, c.nCudaDeviceId);
    c.nMultiProcessorCount = p.multiProcessorCount;
    c.nMaxThreadsPerMultiProcessor = p.maxThreadsPerMultiProcessor;
    c.nMaxThreadsPerBlock = p.maxThreadsPerBlock;
    c.nSharedMemPerBlock = p.sharedMemPerBlock;
    cudaDeviceGetAttribute(&c.nCudaDevAttrComputeCapabilityMajor, cudaDevAttrComputeCapabilityMajor, c.nCudaDeviceId);
    cudaDeviceGetAttribute(&c.nCudaDevAttrComputeCapabilityMinor, cudaDevAttrComputeCapabilityMinor, c.nCudaDeviceId);
    cudaStreamGetFlags(c.hStream, &c.nStreamFlags);
    return c;
}

int64_t npp_version() {
    const NppLibraryVersion* v = nppGetLibVersion();
    return v->major * 10000 + v->minor * 100 + v->build;
}

static const NppStreamContext& context() {
    static const NppStreamContext c = make_context();
    return c;
}

// key: [H, W] int32 (packed colours). Returns labels and their count (0 if not compressed).
std::tuple<torch::Tensor, int64_t> npp_label(torch::Tensor key, int64_t conn, bool compress) {
    const int H = key.size(0), W = key.size(1);
    NppiSize roi{W, H};
    int bytes = 0;
    TORCH_CHECK(nppiLabelMarkersUFGetBufferSize_32u_C1R(roi, &bytes) == NPP_SUCCESS);
    auto buf = torch::empty({bytes}, key.options().dtype(torch::kUInt8));
    auto out = torch::empty({H, W}, key.options().dtype(torch::kInt32));
    const auto& c = context();
    NppStatus s = nppiLabelMarkersUF_32u_C1R_Ctx((Npp32u*)key.data_ptr(), W * 4, (Npp32u*)out.data_ptr(), W * 4, roi,
                                                 conn == 8 ? nppiNormInf : nppiNormL1, buf.data_ptr<uint8_t>(), c);
    TORCH_CHECK(s == NPP_SUCCESS, "nppiLabelMarkersUF failed: ", s);
    int n = 0;
    if (compress) {
        buf = torch::Tensor();
        TORCH_CHECK(nppiCompressMarkerLabelsGetBufferSize_32u_C1R(W * H, &bytes) == NPP_SUCCESS);
        auto cbuf = torch::empty({bytes}, key.options().dtype(torch::kUInt8));
        s = nppiCompressMarkerLabelsUF_32u_C1IR_Ctx((Npp32u*)out.data_ptr(), W * 4, roi, W * H, &n, cbuf.data_ptr<uint8_t>(), c);
        TORCH_CHECK(s == NPP_SUCCESS, "nppiCompressMarkerLabelsUF failed: ", s);
    }
    return {out, n};
}
"""


def job(method, npy, save, reps):
    import torch
    sync = torch.cuda.synchronize
    img = torch.from_numpy(np.load(npy)).cuda()
    H, W = img.shape[:2]

    def pack():  # one int per colour, r | g << 8 | b << 16, as a 4-byte view (no 32-bit copies of channels)
        x = torch.zeros((H, W, 4), dtype=torch.uint8, device="cuda")
        x[..., :3] = img
        return x.view(torch.int32)[..., 0]

    key, lib = None, None
    if method.startswith("npp") or method in ("cucim", "cvcuda4"):  # packed before timing; the RGB picture is not needed after
        key = pack()
        if method == "cucim":
            key += 1  # cuCIM treats 0 as background; black would be 0
        del img
        torch.cuda.empty_cache()
    if method in ("ours", "ours4"):
        import colorcc
        conn = 8 if method == "ours" else 4
        run = lambda: colorcc.label(img, conn)
    elif method.startswith("npp"):
        from torch.utils.cpp_extension import load_inline
        cuda = os.environ.get("CUDA_HOME", "/usr/local/cuda")
        mod = load_inline("npp_label", cpp_sources="#include <torch/extension.h>\nstd::tuple<torch::Tensor, int64_t> npp_label(torch::Tensor key, int64_t conn, bool compress);\nint64_t npp_version();",
                          cuda_sources=NPP_SRC, functions=["npp_label", "npp_version"],
                          extra_include_paths=[f"{cuda}/include"], extra_ldflags=[f"-L{cuda}/lib64", "-lnppif", "-lnppc"],
                          build_directory=None, verbose=False)
        compress = method == "npp"
        v = mod.npp_version()
        lib = f"NPP {v // 10000}.{v // 100 % 100}.{v % 100}"
        run = lambda: mod.npp_label(key, 8, compress)
    elif method == "cucim":
        import cupy as cp
        import cucim
        from cucim.skimage.measure import label as culabel
        lib = f"cuCIM {cucim.__version__}, CuPy {cp.__version__}"

        def torch_alloc(size):  # CuPy memory from PyTorch's allocator, so its peak counter sees it
            t = torch.empty(size, dtype=torch.uint8, device="cuda")
            return cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(t.data_ptr(), size, t), 0)
        ckey = cp.from_dlpack(key)
        run = lambda: culabel(ckey, connectivity=2, return_num=True)
    elif method == "cvcuda4":
        import cvcuda
        lib = f"CV-CUDA {cvcuda.__version__}"
        src = cvcuda.as_tensor(key.view(torch.uint32), "HW")   # its unsigned 32-bit input
        run = lambda: (torch.as_tensor(cvcuda.label(src, cvcuda.CONNECTIVITY_4_2D, cvcuda.LABEL.FAST)[0].cuda(), device="cuda"), None)
    elif method == "old":
        import cc_torch
        run = lambda: cc_torch.cc_2d_color(img, 16, 16)
    t0 = time.perf_counter()  # warm-up: at least 3 calls and half a second, so the GPU reaches its working clock
    for k in itertools.count():
        out = run(); sync(); del out
        if k >= 2 and time.perf_counter() - t0 > 0.5:
            break
    ts = []
    for _ in range(reps):
        sync(); t = time.perf_counter(); out = run(); sync(); ts.append(time.perf_counter() - t)
        del out
    if method == "cucim":  # the memory call: CuPy allocations through PyTorch's allocator, so the peak counter sees them
        cp.get_default_memory_pool().free_all_blocks()
        cp.cuda.set_allocator(torch_alloc)
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out = run(); sync()
    peak = torch.cuda.max_memory_allocated() - base
    lab, n = (out if isinstance(out, tuple) else (out, None))
    lab = torch.as_tensor(lab, device="cuda") if not isinstance(lab, torch.Tensor) else lab
    host = lab.cpu().numpy()
    if method == "old":
        n = int((host.ravel() == np.arange(1, host.size + 1, dtype=host.dtype)).sum())  # roots: label == own index + 1
    np.save(save, host)
    print(json.dumps({"time": statistics.median(ts), "times": ts, "peak": peak, "n": int(n) if n is not None else None,
                      "H": H, "W": W, "lib": lib}))


def consistent(a, b):
    """Same grouping: a label map a -> b that holds at every pixel (b splits nothing) and as many labels in b as in a (b merges nothing)."""
    a, b = a.ravel(), b.ravel().astype(np.int64)
    m = np.zeros(int(a.max()) + 1, np.int64)
    m[a] = b
    nb = int(len(np.unique(b)))
    return bool((m[a] == b).all()) and nb == len(np.unique(a)), nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npys", nargs="+")
    ap.add_argument("--methods", nargs="+", default=["ours", "ours4", "npp", "npp_raw", "cucim", "cvcuda4"])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--tmp", default="/dev/shm")
    ap.add_argument("--out", default="nvidia.json")
    a = ap.parse_args()
    rows = []
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    for npy in a.npys:
        row = {"file": os.path.basename(npy)}
        reps = a.reps if os.path.getsize(npy) < 1e9 else 3
        for m in a.methods:
            save = os.path.join(a.tmp, f"lab_{m}.npy")
            r = subprocess.run([sys.executable, os.path.abspath(__file__), "--job", m, npy, save, str(reps)],
                               capture_output=True, text=True)
            if r.returncode:
                err = r.stderr.strip().splitlines()
                row[m] = {"error": next((l for l in reversed(err) if "Error" in l or "error" in l), err[-1] if err else "?")[-300:]}
            else:
                row[m] = json.loads(r.stdout.strip().splitlines()[-1])
            row["H"], row["W"] = row[m].get("H", row.get("H")), row[m].get("W", row.get("W"))
        for m in a.methods:  # agreement with ours, same connectivity
            if m in ("ours", "ours4") or "error" in row[m]:
                continue
            ref = "ours4" if m in ("old", "cvcuda4") else "ours"
            if "error" in row.get(ref, {"error": 1}):
                row[m]["same_grouping_as_" + ref] = "not checked (reference failed)"
                continue
            ok, nb = consistent(np.load(os.path.join(a.tmp, f"lab_{ref}.npy")), np.load(os.path.join(a.tmp, f"lab_{m}.npy")))
            row[m]["same_grouping_as_" + ref] = ok
            row[m]["distinct_labels"] = nb
        for m in a.methods:
            p = os.path.join(a.tmp, f"lab_{m}.npy")
            if os.path.exists(p):
                os.remove(p)
        rows.append(row)
        print(json.dumps(row), flush=True)
        json.dump({"gpu": smi, "rows": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "--job":
        job(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
    else:
        main()
