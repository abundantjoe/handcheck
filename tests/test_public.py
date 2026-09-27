"""Offline tests of the public-data helpers (no network)."""

import numpy as np

from handcheck import public as P


def test_slice_stats_and_centroid_curve():
    m = np.zeros((50, 80), bool)
    m[10:30, 20:60] = True  # 20 x 40 box: centre (x 39.5, y 19.5), axis ratio 0.5
    st = P.slice_stats(m)
    assert st["n"] == 800 and abs(st["cx"] - 39.5) < 1e-9 and abs(st["cy"] - 19.5) < 1e-9
    assert abs(st["axis_ratio"] - 0.5) < 0.01
    rows = [dict(z=float(z), n=800, cx=100.0 + 0.01 * z, cy=50.0, axis_ratio=0.5) for z in range(0, 1000, 8)]
    rows += [dict(z=2000.0, n=10, cx=900.0, cy=900.0, axis_ratio=1.0)]  # near-empty end slice: dropped
    c, info = P.centroid_curve(rows, sigma=16.0)
    assert info["n_used"] == len(rows) - 1
    x, y = c(np.array([500.0]))
    assert abs(x[0] - 105.0) < 0.1 and abs(y[0] - 50.0) < 1e-9


def test_affine_inverse_lookup():
    M = np.array([[2.0, 0, 0, 10], [0, 2.0, 0, 20], [0, 0, 2.0, 30]])
    cat = {"samples": {"S": {"sample": {"properties": {"volume_transforms": [
        {"from_volume_id": "a", "transforms": [{"to_volume_id": "b", "matrix": M.tolist()}]}]}}}}}
    fwd = P.affine(cat, "S", "a", "b")
    inv = P.affine(cat, "S", "b", "a")
    p = np.array([1.0, 2.0, 3.0])
    q = fwd[:, :3] @ p + fwd[:, 3]
    assert np.allclose(inv[:, :3] @ q + inv[:, 3], p)
    assert P.affine(cat, "S", "a", "c") is None
