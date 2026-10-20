// Exact-colour connected components on the GPU (Metal).
// The same code as ../cuda/kernels.cu, line for line; read that file's
// comments. Metal differences:
//   - a warp is a SIMD-group (32 lanes on Apple GPUs); __ballot_sync is simd_ballot
//   - atomicMin is atomic_fetch_min_explicit on the same memory seen as atomic_int
//   - parameters arrive in a small int buffer: H, W, conn, flags, ignore
// Labels are read with plain loads, not atomic ones: on Apple GPUs a relaxed
// atomic load skips the cache, and a stale read is harmless here (see find_root
// in kernels.cu).
#include <metal_stdlib>
using namespace metal;

constant int RUNS = 1, SKIP = 2, PACKED = 4, HALVE = 8;

inline int rgb(device const uchar* img, uint i) {
    device const uchar* p = img + 3 * ulong(i);
    return int(p[0]) | (int(p[1]) << 8) | (int(p[2]) << 16);
}

inline int colour(device const uchar* img, device const int* key, uint i, int flags) {
    return (flags & PACKED) ? key[i] : rgb(img, i);
}

inline int atomic_min(device int* label, int i, int v) {
    return atomic_fetch_min_explicit((device atomic_int*)(label + i), v, memory_order_relaxed);
}

inline int find_root(device const int* label, int i) {
    int p = label[i];
    while (p != i) {
        i = p;
        p = label[i];
    }
    return i;
}

inline int find_root_halving(device int* label, int i) {
    while (true) {
        const int p = label[i];
        if (p == i) return i;
        const int gp = label[p];
        if (gp == p) return p;
        atomic_min(label, i, gp);
        i = gp;
    }
}

inline void unite(device int* label, int a, int b, bool halve) {
    while (true) {
        a = halve ? find_root_halving(label, a) : find_root(label, a);
        b = halve ? find_root_halving(label, b) : find_root(label, b);
        if (a == b) return;
        if (a > b) { int t = a; a = b; b = t; }
        int old = atomic_min(label, b, a);
        if (old == b) return;
        b = old;
    }
}

// Dispatch with threadgroups of (32, 8) over (W rounded up to 32, H).
kernel void init(device const uchar* img, device int* key, device int* label,
                 device const int* P, uint2 g [[thread_position_in_grid]],
                 uint lane [[thread_index_in_simdgroup]]) {
    const int H = P[0], W = P[1], flags = P[3], ignore = P[4];
    const int x = g.x, y = g.y;
    const bool in = x < W && y < H;
    const int i = y * W + x;
    const int k = in ? rgb(img, i) : -2;
    int parent = i;
    if (flags & RUNS) {
        const bool start = !in || lane == 0 || rgb(img, i - 1) != k;
        const uint starts = uint(ulong(simd_ballot(start)));
        const int first = 31 - int(clz(starts & (0xffffffffu >> (31 - lane))));
        parent = i - (int(lane) - first);
    }
    if (!in) return;
    if (flags & PACKED) key[i] = k;
    label[i] = (k == ignore) ? -1 : parent;
}

// Joins to the row above. If the left neighbour has the same colour (left),
// it already reaches up-left and up, so with SKIP one join per run above is enough.
inline void unite_above(device const uchar* img, device const int* key, device int* label,
                        int W, int conn, int f, int i, int x, int k, bool left) {
    const bool halve = f & HALVE;
    const int up = i - W;
    const bool u = colour(img, key, up, f) == k;
    if (conn == 4) {
        const bool covered = (f & SKIP) && left && colour(img, key, up - 1, f) == k;
        if (u && !covered) unite(label, i, up, halve);
        return;
    }
    const bool ul = x > 0 && colour(img, key, up - 1, f) == k;
    const bool ur = x + 1 < W && colour(img, key, up + 1, f) == k;
    if (!(f & SKIP)) {
        if (ul) unite(label, i, up - 1, halve);
        if (u) unite(label, i, up, halve);
        if (ur) unite(label, i, up + 1, halve);
    } else if (left) {
        if (ur && !u) unite(label, i, up + 1, halve);
    } else if (u) {
        unite(label, i, up, halve);
    } else {
        if (ul) unite(label, i, up - 1, halve);
        if (ur) unite(label, i, up + 1, halve);
    }
}

// The row above first, then the left neighbour (see kernels.cuh).
kernel void merge(device const uchar* img, device const int* key, device int* label,
                  device const int* P, uint2 g [[thread_position_in_grid]],
                  uint lane [[thread_index_in_simdgroup]]) {
    const int H = P[0], W = P[1], conn = P[2], f = P[3], ignore = P[4];
    const int x = g.x, y = g.y;
    if (x >= W || y >= H) return;
    const int i = y * W + x;
    const int k = colour(img, key, i, f);
    if (k == ignore) return;

    const bool left = x > 0 && colour(img, key, i - 1, f) == k;
    if (y > 0) unite_above(img, key, label, W, conn, f, i, x, k, left);
    if (left && !((f & RUNS) && lane != 0)) unite(label, i, i - 1, f & HALVE);
}

kernel void flatten(device int* label, device const int* P,
                    uint2 g [[thread_position_in_grid]]) {
    const int H = P[0], W = P[1];
    const int x = g.x, y = g.y;
    if (x >= W || y >= H) return;
    const int i = y * W + x;
    const int l = label[i];
    if (l >= 0 && l != i) {
        const int r = find_root(label, l);
        if (r != l) label[i] = r;
    }
}

// Dispatch 1-D, threadgroups of CHUNK threads over the H * W pixels.
constant int CHUNK = 256;

kernel void rank_roots(device int* label, device int* count, device const int* P,
                       uint i [[thread_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                       uint c [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                       uint warp [[simdgroup_index_in_threadgroup]]) {
    threadgroup int warp_roots[CHUNK / 32];
    const int n = P[0] * P[1];
    const bool root = int(i) < n && label[i] == int(i);
    const uint votes = uint(ulong(simd_ballot(root)));
    if (lane == 0) warp_roots[warp] = popcount(votes);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int rank = popcount(votes & ((1u << lane) - 1));
    for (uint w = 0; w < warp; w++) rank += warp_roots[w];
    if (root) label[i] = -(rank + 2);
    if (t == 0) {
        int total = 0;
        for (int w = 0; w < CHUNK / 32; w++) total += warp_roots[w];
        count[c] = total;
    }
}

kernel void number(device int* label, device const int* last, device const int* P,
                   uint i [[thread_position_in_grid]]) {
    const int n = P[0] * P[1];
    if (int(i) >= n) return;
    const int l = label[i];
    if (l == -1) { label[i] = 0; return; }
    const int r = l < 0 ? int(i) : l;
    const int v = l < 0 ? l : label[r];
    const int c = r / CHUNK;
    label[i] = v > 0 ? v : (c ? last[c - 1] : 0) + (-v - 2) + 1;
}
