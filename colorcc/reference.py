"""CPU versions, used as ground truth and as the baselines in the benchmark."""
import numpy as np


def pack(img):
    """[H, W, 3] uint8 -> [H, W] uint32 colour key, r | g << 8 | b << 16.

    Built as a 4-byte view (little-endian), so an 800-dpi sheet needs one
    3.2 GB array, not three 32-bit copies of every channel."""
    H, W = img.shape[:2]
    x = np.zeros((H, W, 4), np.uint8)
    x[..., :3] = img
    return x.view(np.uint32)[..., 0]


def _ignored(ignore):
    return int(ignore[0]) | int(ignore[1]) << 8 | int(ignore[2]) << 16


def label_cc3d(img, conn=8, ignore=None):
    """One pass over all colours at once with cc3d (multi-label union-find on the CPU)."""
    import cc3d
    key = pack(img)
    key += 1  # cc3d treats 0 as background; colours stay below 2^24, so no overflow
    if ignore is not None:
        key[key == _ignored(ignore) + 1] = 0
    out, n = cc3d.connected_components(key, connectivity=conn, return_N=True)
    return out, int(n)


def label_per_colour(img, conn=8, ignore=None):
    """The slow way: one binary connected-components call per distinct colour."""
    import cv2
    key = pack(img)
    out = np.zeros(key.shape, np.int32)
    n = 0
    skip = None if ignore is None else _ignored(ignore)
    for c in np.unique(key):
        if c == skip:
            continue
        m, lab = cv2.connectedComponents((key == c).view(np.uint8), connectivity=conn, ltype=cv2.CV_32S)
        fg = lab > 0
        out[fg] = lab[fg] + n
        n += m - 1
    return out, n


def same_partition(a, b):
    """True when a and b group the pixels the same way (label values may differ)."""
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    if ((a == 0) != (b == 0)).any():
        return False
    pairs = np.unique(np.stack([a, b]), axis=1).shape[1]
    return pairs == len(np.unique(a)) == len(np.unique(b))
