"""Why is cuCIM's plain union-find faster than ours, and how fast is everyone once packing is taken out of the timing?

    python bench/profile.py cache/sheet.pdf.300.npy cache/sheet.pdf.600.npy --out profile.json

Jobs (each in its own process, so peak memory is its own; the 32-bit packed picture is made before timing):
  ours        colorcc.label per stage (init, merge, flatten, number), base (no switches) and default switches
  cucim       cuCIM 26.x label() on the pre-packed picture, total and per kernel (init, connect, count,
              labels, sort, finalize), with its own kernels from cucim.skimage.measure._label_kernels
  npp         nppiLabelMarkersUF_32u on the pre-packed picture, stream context and scratch made once
  cvcuda      cvcuda.label (8 neighbours) on the pre-packed picture
  variants    one union-find merge kernel compiled 32 ways, flipping the five things ours and cuCIM's do
              differently: the join (atomicMin on any parent / atomicCAS on a root only), the neighbour order,
              the launch shape (32 x 8 tiles / 1-D blocks of 128), the index arithmetic (x and y / div and mod per
              neighbour) and the colour read (packed int / three bytes); plus counters of find steps, atomics
              and failed atomics for the joins and orders
Time: median of --reps after a warm-up, GPU events around the GPU work only.
Agreement: every method's grouping is compared with ours on the CPU.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from nvidia import consistent  # noqa: E402


def packed(torch, img):
    """One int32 per pixel, r | g << 8 | b << 16, plus 1 so that no pixel is 0 (cuCIM's background)."""
    x = torch.zeros(img.shape[:2] + (4,), dtype=torch.uint8, device="cuda")
    x[..., :3] = img
    return x.view(torch.int32)[..., 0] + 1


def timed(torch, fn, reps):
    fn(); torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); out = fn(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) / 1e3)
        del out
    return statistics.median(ts)


def peak(torch, fn):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
    out = fn(); torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - base, out


def cupy_from_torch_allocator(torch, cp):
    """CuPy allocations through PyTorch's allocator, so PyTorch's peak counter sees them."""
    def alloc(size):
        t = torch.empty(size, dtype=torch.uint8, device="cuda")
        return cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(t.data_ptr(), size, t), 0)
    cp.cuda.set_allocator(alloc)


def job_ours(torch, img, reps, save):
    import colorcc
    res = {}
    for name, flags in (("base", 0), ("default", colorcc.DEFAULT)):
        cum = [timed(torch, lambda s=s: colorcc.label(img, 8, None, flags, s), reps) for s in (1, 2, 3, 4)]
        res[name] = {"init": cum[0], "merge": cum[1] - cum[0], "flatten": cum[2] - cum[1], "number": cum[3] - cum[2], "total": cum[3]}
    res["peak"], (lab, n) = peak(torch, lambda: colorcc.label(img, 8))
    res["n"] = int(n)
    np.save(save, lab.cpu().numpy())
    return res


def job_cucim(torch, img, reps, save):
    import cupy as cp
    from cucim.skimage.measure import label
    from cucim.skimage.measure import _label_kernels as K
    import cucim
    cupy_from_torch_allocator(torch, cp)
    key = cp.from_dlpack(packed(torch, img)); torch.cuda.synchronize()
    res = {"version": cucim.__version__, "total": timed(torch, lambda: label(key, connectivity=2), reps)}
    res["peak"], lab = peak(torch, lambda: label(key, connectivity=2))
    np.save(save, cp.asnumpy(lab)); del lab

    # The same steps as cucim's _label(), one event between each, so the time of every kernel shows.
    structure = np.ones((3, 3), bool)
    elems = np.where(structure != 0)
    vecs = [elems[d] - 1 for d in range(2)]
    offset = vecs[0] * 3 + vecs[1]
    idx = np.where(offset < 0)[0]
    dirs = cp.array([[vecs[d][r] for d in range(2)] for r in idx], dtype=np.int32)
    steps = {k: [] for k in ("init", "connect", "count", "sync", "labels", "sort", "finalize")}
    for rep in range(reps + 1):
        y = cp.empty(key.shape, dtype=cp.int32)
        shape = cp.array(y.shape, dtype=np.int32)
        count = cp.zeros(2, dtype=y.dtype)
        ev = [cp.cuda.Event() for _ in range(8)]
        ev[0].record(); K._kernel_init()(key, y)
        ev[1].record(); K._kernel_connect(True)(key, shape, dirs, len(idx), 2, y, size=y.size)
        ev[2].record(); K._kernel_count()(y, count, size=y.size)
        ev[3].record(); maxlabel = int(count[0])
        ev[4].record(); labels = cp.empty(maxlabel, dtype=y.dtype); K._kernel_labels()(y, count, labels, size=y.size)
        ev[5].record(); srt = cp.sort(labels)
        ev[6].record(); K._kernel_finalize()(maxlabel, srt, y, size=y.size)
        ev[7].record(); ev[7].synchronize()
        if rep:
            for k, (a, b) in zip(steps, zip(ev, ev[1:])):
                steps[k].append(cp.cuda.get_elapsed_time(a, b) / 1e3)
        del y, labels, srt
    res["steps"] = {k: statistics.median(v) for k, v in steps.items()}
    res["n"] = maxlabel
    return res


NPP_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <nppi_filtering_functions.h>

static NppStreamContext ctx;
static torch::Tensor scratch, cscratch;

void npp_setup(int64_t H, int64_t W, torch::Tensor like) {
    ctx = NppStreamContext{};
    ctx.hStream = c10::cuda::getCurrentCUDAStream();
    cudaGetDevice(&ctx.nCudaDeviceId);
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, ctx.nCudaDeviceId);
    ctx.nMultiProcessorCount = p.multiProcessorCount;
    ctx.nMaxThreadsPerMultiProcessor = p.maxThreadsPerMultiProcessor;
    ctx.nMaxThreadsPerBlock = p.maxThreadsPerBlock;
    ctx.nSharedMemPerBlock = p.sharedMemPerBlock;
    cudaDeviceGetAttribute(&ctx.nCudaDevAttrComputeCapabilityMajor, cudaDevAttrComputeCapabilityMajor, ctx.nCudaDeviceId);
    cudaDeviceGetAttribute(&ctx.nCudaDevAttrComputeCapabilityMinor, cudaDevAttrComputeCapabilityMinor, ctx.nCudaDeviceId);
    cudaStreamGetFlags(ctx.hStream, &ctx.nStreamFlags);
    NppiSize roi{(int)W, (int)H};
    int bytes = 0;
    TORCH_CHECK(nppiLabelMarkersUFGetBufferSize_32u_C1R(roi, &bytes) == NPP_SUCCESS && bytes > 0, "buffer size ", bytes);
    scratch = torch::empty({bytes}, like.options().dtype(torch::kUInt8));
    TORCH_CHECK(nppiCompressMarkerLabelsGetBufferSize_32u_C1R(W * H, &bytes) == NPP_SUCCESS && bytes > 0, "compress buffer size ", bytes);
    cscratch = torch::empty({bytes}, like.options().dtype(torch::kUInt8));
}

int64_t npp_run(torch::Tensor key, torch::Tensor out, bool compress) {
    const int H = key.size(0), W = key.size(1);
    NppiSize roi{W, H};
    TORCH_CHECK(nppiLabelMarkersUF_32u_C1R_Ctx((Npp32u*)key.data_ptr(), W * 4, (Npp32u*)out.data_ptr(), W * 4, roi,
                                                nppiNormInf, scratch.data_ptr<uint8_t>(), ctx) == NPP_SUCCESS);
    int n = 0;
    if (compress)
        TORCH_CHECK(nppiCompressMarkerLabelsUF_32u_C1IR_Ctx((Npp32u*)out.data_ptr(), W * 4, roi, W * H, &n,
                                                             cscratch.data_ptr<uint8_t>(), ctx) == NPP_SUCCESS);
    return n;
}
"""


def job_npp(torch, img, reps, save):
    from torch.utils.cpp_extension import load_inline
    cuda = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    mod = load_inline("npp_fair", cpp_sources="#include <torch/extension.h>\nvoid npp_setup(int64_t H, int64_t W, torch::Tensor like);\nint64_t npp_run(torch::Tensor key, torch::Tensor out, bool compress);",
                      cuda_sources=NPP_SRC, functions=["npp_setup", "npp_run"],
                      extra_include_paths=[f"{cuda}/include"], extra_ldflags=[f"-L{cuda}/lib64", "-lnppif", "-lnppc"])
    key = packed(torch, img)
    H, W = key.shape
    mod.npp_setup(H, W, key)
    out = torch.empty((H, W), dtype=torch.int32, device="cuda")
    res = {"raw": timed(torch, lambda: mod.npp_run(key, out, False), reps),
           "compressed": timed(torch, lambda: mod.npp_run(key, out, True), reps)}
    mod.npp_run(key, out, False); torch.cuda.synchronize()
    np.save(save, out.cpu().numpy())
    return res


def job_cvcuda(torch, img, reps, save):
    import cvcuda
    key = packed(torch, img).view(torch.uint32) if hasattr(torch, "uint32") else packed(torch, img)
    H, W = key.shape
    errs = []
    for layout, view in (("HW", key), ("NHW", key.view(1, H, W)), ("NHWC", key.view(1, H, W, 1))):
        try:
            t = cvcuda.as_tensor(view, layout)
            run = lambda: cvcuda.label(t, cvcuda.CONNECTIVITY_4_2D, cvcuda.LABEL.FAST)
            run(); torch.cuda.synchronize()
            break
        except Exception as e:  # noqa: BLE001 - record which layouts this cvcuda build refuses
            errs.append(f"{layout}: {e}"[:200])
    else:
        raise RuntimeError("; ".join(errs))
    res = {"version": getattr(cvcuda, "__version__", "?"), "layout": layout, "refused": errs,
           "time": timed(torch, run, reps),
           "time_with_count": timed(torch, lambda: cvcuda.label(t, cvcuda.CONNECTIVITY_4_2D, cvcuda.LABEL.FAST, count=True), reps)}
    try:  # 8 neighbours, which is what we need; recorded if refused
        cvcuda.label(t, cvcuda.CONNECTIVITY_8_2D, cvcuda.LABEL.FAST); torch.cuda.synchronize()
        res["eight_neighbours"] = "ran"
    except Exception as e:  # noqa: BLE001
        res["eight_neighbours"] = str(e)[:200]
    import colorcc  # against ours with the same 4 neighbours
    res["ours4_time"] = timed(torch, lambda: colorcc.label(img, 4), reps)
    res["ours4_time_no_numbering"] = timed(torch, lambda: colorcc.label(img, 4, None, colorcc.DEFAULT, 3), reps)
    ours4, _ = colorcc.label(img, 4)
    torch.cuda.synchronize(); free0 = torch.cuda.mem_get_info()[0]
    out = run(); torch.cuda.synchronize()
    res["held_after_call"] = free0 - torch.cuda.mem_get_info()[0]  # what the call keeps (output and any cached workspace)
    lab = torch.as_tensor(out[0].cuda(), device="cuda").reshape(H, W)
    res["same_grouping_as_ours4"], res["distinct_labels"] = consistent(ours4.cpu().numpy(), lab.cpu().numpy())
    return res


# One merge kernel, compiled once per combination of the five differences between our base kernel and cuCIM's.
UF_SRC = r"""
#ifdef COUNT
#define BUMP(k) (c[k]++)
#else
#define BUMP(k)
#endif

__device__ __forceinline__ bool same(const int* key, const unsigned char* rgb, int a, int b) {
#ifdef RGB
    return rgb[3 * a] == rgb[3 * b] && rgb[3 * a + 1] == rgb[3 * b + 1] && rgb[3 * a + 2] == rgb[3 * b + 2];
#else
    return key[a] == key[b];
#endif
}

__device__ __forceinline__ int find(const int* L, int i, unsigned long long* c) {
    int p = L[i];
    while (p != i) { i = p; p = L[i]; BUMP(0); }
    return i;
}

__device__ void unite(int* L, int a, int b, unsigned long long* c) {
#ifdef CAS
    while (true) {
        a = find(L, a, c); b = find(L, b, c);
        if (a == b) return;
        BUMP(1);
        if (a < b) { int old = atomicCAS(L + b, b, a); if (old == b) return; BUMP(2); b = old; }
        else       { int old = atomicCAS(L + a, a, b); if (old == a) return; BUMP(2); a = old; }
    }
#else
    while (true) {
        a = find(L, a, c); b = find(L, b, c);
        if (a == b) return;
        if (a > b) { int t = a; a = b; b = t; }
        BUMP(1);
        int old = atomicMin(L + b, a);
        if (old == b) return;
        BUMP(2);
        b = old;
    }
#endif
}

extern "C" __global__ void init(int* L, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) L[i] = i;
}

// Neighbour (dy, dx) of pixel i, or -1 outside the picture.
__device__ __forceinline__ int nb(int i, int x, int y, int dy, int dx, int W, int H) {
#ifdef DIVMOD
    x = i % W; y = i / W;
#endif
    int xx = x + dx, yy = y + dy;
    return (xx < 0 || xx >= W || yy < 0 || yy >= H) ? -1 : yy * W + xx;
}

extern "C" __global__ void merge(const int* key, const unsigned char* rgb, int* L, int W, int H, unsigned long long* st) {
#ifdef LAYOUT_1D
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= W * H) return;
    int x = i % W, y = i / W;
#else
    int x = blockIdx.x * 32 + threadIdx.x, y = blockIdx.y * 8 + threadIdx.y;
    if (x >= W || y >= H) return;
    int i = y * W + x;
#endif
#ifdef ORDER_CUCIM
    const int D[4][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}};   // up-left, up, up-right, left
#else
    const int D[4][2] = {{0, -1}, {-1, -1}, {-1, 0}, {-1, 1}};   // left, up-left, up, up-right
#endif
    unsigned long long c[3] = {0, 0, 0};   // find steps, atomics, failed atomics: summed once per thread
    for (int d = 0; d < 4; d++) {
        int k = nb(i, x, y, D[d][0], D[d][1], W, H);
        if (k >= 0 && same(key, rgb, i, k)) unite(L, i, k, c);
    }
#ifdef COUNT
    for (int k = 0; k < 3; k++) if (c[k]) atomicAdd(st + k, c[k]);
#endif
}

// cuCIM 26.10's connect kernel, line for line (cucim/skimage/measure/_label_kernels.py, greyscale mode, 2-D,
// 8 neighbours, int32 labels), one thread per pixel as CuPy launches it, with the same counters as above.
extern "C" __global__ void cucim_connect(const int* x, int* y, int W, int H, unsigned long long* st) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= W * H) return;
    const int shape[2] = {H, W};
    const int dirs[4][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}};
    unsigned long long c[3] = {0, 0, 0};
    if (y[i] < 0) return;
    for (int dr = 0; dr < 4; dr++) {
        int j = i, rest = j, stride = 1, k = 0;
        for (int dm = 1; dm >= 0; dm--) {
            int pos = rest % shape[dm] + dirs[dr][dm];
            if (pos < 0 || pos >= shape[dm]) { k = -1; break; }
            k += pos * stride;
            rest /= shape[dm];
            stride *= shape[dm];
        }
        if (k < 0) continue;
        if (y[k] < 0) continue;
        if (x[k] != x[j]) continue;
        while (1) {
            while (j != y[j]) { j = y[j]; BUMP(0); }
            while (k != y[k]) { k = y[k]; BUMP(0); }
            if (j == k) break;
            BUMP(1);
            if (j < k) {
                int old = atomicCAS(&y[k], k, j);
                if (old == k) break;
                BUMP(2);
                k = old;
            } else {
                int old = atomicCAS(&y[j], j, k);
                if (old == j) break;
                BUMP(2);
                j = old;
            }
        }
    }
#ifdef COUNT
    for (int q = 0; q < 3; q++) if (c[q]) atomicAdd(st + q, c[q]);
#endif
}

extern "C" __global__ void roots(const int* L, int n, unsigned long long* count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && L[i] == i) atomicAdd(count, 1ULL);
}
"""


def job_variants(torch, img, reps, save):
    import cupy as cp
    import itertools
    rgb = cp.from_dlpack(img.contiguous().view(-1))
    key = cp.from_dlpack(packed(torch, img))
    H, W = key.shape
    n = H * W
    L = cp.empty(n, dtype=cp.int32)
    st = cp.zeros(3, dtype=cp.uint64)
    out = []
    names = ("CAS", "ORDER_CUCIM", "LAYOUT_1D", "DIVMOD", "RGB")
    combos = list(itertools.product((0, 1), repeat=5))
    if os.environ.get("JOINS_ONLY"):  # for the hardware counters: the joins and orders, on our layout
        combos = [c for c in combos if not any(c[2:])]
    if os.environ.get("JOINS_LAYOUTS"):  # the joins and orders on both layouts, with counters
        combos = [c for c in combos if not any(c[3:])]
    for on in combos:
        opts = tuple(f"-D{nm}" for nm, v in zip(names, on) if v)
        mod = cp.RawModule(code=UF_SRC, options=opts)
        init, merge, roots = mod.get_function("init"), mod.get_function("merge"), mod.get_function("roots")
        grid = ((n + 127) // 128,) if "-DLAYOUT_1D" in opts else ((W + 31) // 32, (H + 7) // 8)
        block = (128,) if "-DLAYOUT_1D" in opts else (32, 8)
        args = (key, rgb, L, np.int32(W), np.int32(H), st)

        def run():
            init(((n + 255) // 256,), (256,), (L, np.int32(n)))
            merge(grid, block, args)
        ts = []
        for rep in range(reps + 1):
            a, b = cp.cuda.Event(), cp.cuda.Event()
            a.record(); run(); b.record(); b.synchronize()
            if rep:
                ts.append(cp.cuda.get_elapsed_time(a, b) / 1e3)
        c = cp.zeros(1, dtype=cp.uint64)
        roots(((n + 255) // 256,), (256,), (L, np.int32(n), c))
        row = {"join": "cas" if on[0] else "min", "order": "cucim" if on[1] else "ours", "layout": "1d" if on[2] else "2d",
               "index": "divmod" if on[3] else "xy", "colour": "rgb" if on[4] else "packed",
               "time": statistics.median(ts), "regions": int(c[0])}
        if not on[3] and not on[4]:  # count the work for each join, order and layout
            cmod = cp.RawModule(code=UF_SRC, options=opts + ("-DCOUNT",))
            st[:] = 0
            cmod.get_function("init")(((n + 255) // 256,), (256,), (L, np.int32(n)))
            cmod.get_function("merge")(grid, block, args)
            s = cp.asnumpy(st)
            row.update(find_steps=int(s[0]), atomics=int(s[1]), failed_atomics=int(s[2]))
        out.append(row)
        print(json.dumps(row), flush=True)
    for count in (False, True):  # cuCIM's own kernel, timed, then counted
        mod = cp.RawModule(code=UF_SRC, options=("-DCOUNT",) if count else ())
        init, conn, roots = mod.get_function("init"), mod.get_function("cucim_connect"), mod.get_function("roots")
        st[:] = 0
        ts = []
        for rep in range(1 if count else reps + 1):
            a, b = cp.cuda.Event(), cp.cuda.Event()
            a.record()
            init(((n + 255) // 256,), (256,), (L, np.int32(n)))
            conn(((n + 127) // 128,), (128,), (key, L, np.int32(W), np.int32(H), st))
            b.record(); b.synchronize()
            if rep and not count:
                ts.append(cp.cuda.get_elapsed_time(a, b) / 1e3)
        c = cp.zeros(1, dtype=cp.uint64)
        roots(((n + 255) // 256,), (256,), (L, np.int32(n), c))
        if count:
            s = cp.asnumpy(st)
            cuc.update(find_steps=int(s[0]), atomics=int(s[1]), failed_atomics=int(s[2]))
        else:
            cuc = {"join": "cucim's own kernel", "time": statistics.median(ts), "regions": int(c[0])}
    out.append(cuc)
    print(json.dumps(cuc), flush=True)
    return {"variants": out}


JOBS = {"ours": job_ours, "cucim": job_cucim, "npp": job_npp, "cvcuda": job_cvcuda, "variants": job_variants}


def job(method, npy, save, reps):
    import torch
    img = torch.from_numpy(np.load(npy)).cuda()
    res = JOBS[method](torch, img, reps, save)
    res.update(H=int(img.shape[0]), W=int(img.shape[1]))
    print(json.dumps(res))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npys", nargs="+")
    ap.add_argument("--methods", nargs="+", default=list(JOBS))
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--tmp", default="/dev/shm")
    ap.add_argument("--out", default="profile.json")
    a = ap.parse_args()
    rows = []
    for npy in a.npys:
        row = {"file": os.path.basename(npy)}
        for m in a.methods:
            save = os.path.join(a.tmp, f"prof_{m}.npy")
            r = subprocess.run([sys.executable, os.path.abspath(__file__), "--job", m, npy, save, str(a.reps)],
                               capture_output=True, text=True)
            if r.returncode:
                err = r.stderr.strip().splitlines()
                row[m] = {"error": "\n".join(err[-6:])[-1200:]}
            else:
                row[m] = json.loads(r.stdout.strip().splitlines()[-1])
            print(m, json.dumps(row[m])[:400], flush=True)
        if os.path.exists(os.path.join(a.tmp, "prof_ours.npy")):
            ref = np.load(os.path.join(a.tmp, "prof_ours.npy"))
            for m in ("cucim", "npp"):
                p = os.path.join(a.tmp, f"prof_{m}.npy")
                if m in row and "error" not in row[m] and os.path.exists(p):
                    row[m]["same_grouping_as_ours"], row[m]["distinct_labels"] = consistent(ref, np.load(p))
            if "variants" in row and "error" not in row["variants"]:
                for v in row["variants"]["variants"]:
                    v["right_count"] = v["regions"] == row["ours"]["n"]
        for m in a.methods:
            p = os.path.join(a.tmp, f"prof_{m}.npy")
            if os.path.exists(p):
                os.remove(p)
        rows.append(row)
        json.dump({"rows": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "--job":
        job(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
    else:
        main()
