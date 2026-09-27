"""TEST HELPER ONLY - not for production rendering.

Trilinear sampling of a synthetic volume along vertex normals, plus
synthetic-volume painters used by the geometry tests.

Volume indexing follows the official convention: a volume array is indexed
``vol[z, y, x]`` while mesh points are (x, y, z).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

from .tifxyz import valid_mask


def slice_offsets(n: int) -> np.ndarray:
    """Offsets -(n//2) .. such that slice index n//2 has offset 0."""
    if n < 1:
        raise ValueError("n must be >= 1")
    return np.arange(n, dtype=np.float64) - (n // 2)


def sample_volume(vol: np.ndarray, xyz: np.ndarray, cval: float = 0.0) -> np.ndarray:
    """Trilinear sample ``vol[z, y, x]`` at (..., 3) xyz points."""
    q = np.asarray(xyz, dtype=np.float64)
    coords = np.stack([q[..., 2].ravel(), q[..., 1].ravel(), q[..., 0].ravel()])
    out = map_coordinates(np.asarray(vol, dtype=np.float32), coords, order=1, mode="constant", cval=cval)
    return out.reshape(q.shape[:-1])


def render_along_normals(
    vol: np.ndarray,
    points: np.ndarray,
    normals: np.ndarray,
    n: int = 31,
    spacing: float = 1.0,
    invalid_value: float = np.nan,
) -> np.ndarray:
    """(n, H, W) stack; slice k samples ``p + (k - n//2) * spacing * n_hat``.
    Slice n//2 lies on the mesh."""
    p = np.asarray(points, dtype=np.float64)
    nn = np.asarray(normals, dtype=np.float64)
    ok = valid_mask(p) & np.isfinite(nn).all(-1)
    nn = np.where(ok[..., None], nn, 0.0)
    offs = slice_offsets(n) * spacing
    stack = np.empty((n, *p.shape[:2]), dtype=np.float32)
    for i, o in enumerate(offs):
        s = sample_volume(vol, p + o * nn)
        s[~ok] = invalid_value
        stack[i] = s
    return stack


def glyph_F(h: int = 20, w: int = 14) -> np.ndarray:
    """Asymmetric 'F' template (float32, 1 = ink), readable as displayed."""
    g = np.zeros((h, w), dtype=np.float32)
    t = max(2, h // 7)
    g[2:h - 2, 2:2 + t] = 1.0            # stem
    g[2:2 + t, 2:w - 2] = 1.0            # top bar
    mid = h // 2 - t // 2
    g[mid:mid + t, 2:w - 4] = 1.0        # middle bar (shorter)
    return g


def cylinder_glyph_volume(
    shape_zyx: tuple[int, int, int],
    centre_xy: tuple[float, float],
    radius: float,
    glyph: np.ndarray,
    *,
    phi0: float,
    z0: float,
    pix_vox: float,
    sigma: float = 1.2,
    base: float = 0.05,
    readable_from_outside: bool = True,
) -> np.ndarray:
    """Volume with a thin cylindrical sheet (axis along z) on which ``glyph``
    is painted.

    Glyph row r maps to z = z0 + r * pix_vox (glyph top at low z). Glyph
    column c maps to arc s = c * pix_vox; with ``readable_from_outside`` the
    glyph reads correctly for a viewer outside the cylinder looking inward
    with low z at the top (phi = phi0 - s / R), otherwise from inside
    (phi = phi0 + s / R).
    """
    Z, Y, X = shape_zyx
    zz, yy, xx = np.meshgrid(np.arange(Z, dtype=np.float64), np.arange(Y, dtype=np.float64),
                             np.arange(X, dtype=np.float64), indexing="ij")
    dx, dy = xx - centre_xy[0], yy - centre_xy[1]
    r = np.hypot(dx, dy)
    shell = np.exp(-0.5 * ((r - radius) / sigma) ** 2)
    phi = np.arctan2(dy, dx)
    dphi = (phi - phi0 + np.pi) % (2 * np.pi) - np.pi
    s = (-dphi if readable_from_outside else dphi) * radius
    col = s / pix_vox
    row = (zz - z0) / pix_vox
    ink = map_coordinates(np.asarray(glyph, dtype=np.float32), np.stack([row.ravel(), col.ravel()]),
                          order=1, mode="constant", cval=0.0).reshape(zz.shape)
    return (shell * (base + ink)).astype(np.float32)


def plane_volume(shape_zyx: tuple[int, int, int], x_positions, sigma: float = 0.7,
                 amplitudes=None) -> np.ndarray:
    """Volume with bright planes x = const (Gaussian profile along x)."""
    Z, Y, X = shape_zyx
    x = np.arange(X, dtype=np.float64)
    prof = np.zeros(X)
    amps = amplitudes if amplitudes is not None else [1.0] * len(list(x_positions))
    for x0, a in zip(x_positions, amps):
        prof += a * np.exp(-0.5 * ((x - x0) / sigma) ** 2)
    return np.broadcast_to(prof[None, None, :], (Z, Y, X)).astype(np.float32).copy()


def cylinder_grid(
    radius: float,
    centre_xy: tuple[float, float],
    shape_hw: tuple[int, int],
    step: float,
    angle_deg: float,
    *,
    s0: float = 0.0,
    z0: float = 0.0,
    phi_ref: float = 0.0,
) -> np.ndarray:
    """(H, W, 3) grid on a cylinder of the given radius (axis along z).

    The grid is an isometric parameterisation of the unrolled cylinder,
    rotated by ``angle_deg`` relative to iso-z:
    arc s = step*(u cos a - v sin a) + s0, z = step*(u sin a + v cos a) + z0,
    phi = phi_ref + s / R. With this orientation dp/du x dp/dv points
    outward for a in (-90, 90) deg.
    """
    H, W = shape_hw
    a = np.radians(angle_deg)
    vv, uu = np.mgrid[0:H, 0:W].astype(np.float64)
    s = step * (uu * np.cos(a) - vv * np.sin(a)) + s0
    z = step * (uu * np.sin(a) + vv * np.cos(a)) + z0
    phi = phi_ref + s / radius
    return np.stack([centre_xy[0] + radius * np.cos(phi), centre_xy[1] + radius * np.sin(phi), z],
                    axis=-1).astype(np.float32)
