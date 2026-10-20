"""GPU labels must group pixels exactly like cc3d does, for every switch setting.

Run: pytest tests   (uses CUDA if present, else the Apple GPU)
"""
import itertools

import numpy as np
import pytest
import torch

import colorcc
from colorcc.reference import label_cc3d, label_per_colour, same_partition

DEV = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else None
gpu = pytest.mark.skipif(DEV is None, reason="needs a CUDA or Apple GPU")
WHITE = (255, 255, 255)


def palette(idx, colours):
    return np.asarray(colours, np.uint8)[idx]


def cases():
    rng = np.random.default_rng(0)
    two = [(0, 0, 0), WHITE]
    yield "noise-2", palette(rng.integers(0, 2, (61, 97)), two)
    yield "noise-5", palette(rng.integers(0, 5, (130, 70)), [(0, 0, 0), WHITE, (1, 0, 0), (0, 1, 0), (0, 0, 1)])
    yield "checker", palette(np.indices((40, 67)).sum(0) % 2, two)  # 8-conn: 2 components; 4-conn: every pixel
    stripes = np.zeros((50, 200), int)
    for d in range(-200, 50, 7):  # one-pixel diagonals: only 8-connectivity keeps them whole
        r = np.arange(50); c = r + d; ok = (c >= 0) & (c < 200); stripes[r[ok], c[ok]] = 1
    yield "diagonals", palette(stripes, [WHITE, (0, 0, 0)])
    sp = np.zeros((65, 65), int); y = x = 32; dy, dx, step = 0, 1, 1  # a spiral: one long chain
    while 0 <= y < 65 and 0 <= x < 65:
        for _ in range(2):
            for _ in range(step):
                if 0 <= y < 65 and 0 <= x < 65: sp[y, x] = 1
                y += dy; x += dx
            dy, dx = dx, -dy
        step += 2
    yield "spiral", palette(sp, [WHITE, (200, 0, 0)])
    yield "flat", np.full((33, 129, 3), 7, np.uint8)
    yield "one-row", palette(rng.integers(0, 3, (1, 300)), [(0, 0, 0), WHITE, (9, 9, 9)])
    yield "one-col", palette(rng.integers(0, 3, (300, 1)), [(0, 0, 0), WHITE, (9, 9, 9)])
    blob = rng.integers(0, 4, (24, 24)).repeat(5, 0).repeat(5, 1)[:117, :101]  # blocky regions, odd size
    yield "blocks", palette(blob, [(0, 0, 0), WHITE, (128, 128, 128), (255, 0, 0)])
    big = rng.integers(0, 2, (300, 1000)).repeat(3, 0).repeat(3, 1)  # wider than many warps
    yield "wide", palette(big, two)


FLAGS = [0, colorcc.RUNS, colorcc.SKIP, colorcc.PACKED, colorcc.HALVE, colorcc.DEFAULT, colorcc.ALL]


@gpu
@pytest.mark.parametrize("name,img", list(cases()), ids=[c[0] for c in cases()])
@pytest.mark.parametrize("conn,flags,ignore", list(itertools.product((4, 8), FLAGS, (None, WHITE))))
def test_matches_cc3d(name, img, conn, flags, ignore):
    want, n_want = label_cc3d(img, conn, ignore)
    got, n = colorcc.label(torch.from_numpy(img).to(DEV), conn, ignore, flags)
    got = got.cpu().numpy()
    assert n == n_want
    assert got.max() == n and (got[got > 0].min() == 1 if n else True)
    assert same_partition(got, want)
    assert np.array_equal(got, want)  # same numbering too: raster order of first pixel


def test_per_colour_loop_agrees():
    img = next(c for c in cases() if c[0] == "noise-5")[1]
    a, na = label_per_colour(img, 8)
    b, nb = label_cc3d(img, 8)
    assert na == nb and same_partition(a, b)


@gpu
def test_repeatable_on_a_big_image():
    """Races show up as rare wrong answers, so label a large image many times."""
    rng = np.random.default_rng(1)
    img = palette(rng.integers(0, 3, (500, 700)).repeat(3, 0).repeat(3, 1), [(0, 0, 0), WHITE, (5, 5, 5)])
    t = torch.from_numpy(img).to(DEV)
    for conn in (4, 8):
        want, n_want = label_cc3d(img, conn)
        for _ in range(10):
            got, n = colorcc.label(t, conn)
            assert n == n_want and np.array_equal(got.cpu().numpy(), want)


@gpu
@pytest.mark.parametrize("ignore", (None, WHITE))
def test_stage3_roots(ignore):
    """stages=3 leaves each pixel holding its root's pixel index, -1 where ignored, and n = 0."""
    img = next(c for c in cases() if c[0] == "blocks")[1]
    want, _ = label_cc3d(img, 8, ignore)
    got, n = colorcc.label(torch.from_numpy(img).to(DEV), 8, ignore, colorcc.DEFAULT, 3)
    got = got.cpu().numpy()
    assert n == 0
    assert same_partition(got + 1, want)  # -1 (ignored) becomes 0, as in cc3d's output
    roots = got[got >= 0]
    assert (got.ravel()[roots] == roots).all()  # a root points at itself


def test_empty_and_bad_input():
    lab, n = colorcc.label(torch.zeros((0, 5, 3), dtype=torch.uint8))
    assert n == 0 and lab.shape == (0, 5)
    for bad in (torch.zeros((4, 4, 4), dtype=torch.uint8), torch.zeros((4, 4, 3)), torch.zeros((4, 4), dtype=torch.uint8)):
        with pytest.raises(ValueError):
            colorcc.label(bad)
    ok = torch.zeros((4, 4, 3), dtype=torch.uint8)
    for kw in ({"conn": 6}, {"ignore": (256, 0, 0)}, {"ignore": (-1, 0, 0)}, {"ignore": (1, 2)}, {"ignore": 255},
               {"flags": 16}, {"stages": 0}, {"stages": 5}):
        with pytest.raises(ValueError):
            colorcc.label(ok, **kw)
    with pytest.raises(ValueError):
        colorcc.label(np.zeros((4, 4, 3), np.uint8))
    with pytest.raises(ValueError, match="CUDA or MPS"):
        colorcc.label(torch.zeros((4, 4, 3), dtype=torch.uint8))
