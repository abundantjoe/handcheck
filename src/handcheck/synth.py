"""Synthetic spiral scrolls with a planted font-rendered glyph (TEST HELPER
ONLY). The glyph is written on the RECTO (inner, umbilicus-facing) face of
one winding, reading correctly for a viewer on the umbilicus side with the
text top at LOW z (so reading direction = +phi, counter-clockwise x -> y).

Volume indexing vol[z, y, x]; mesh points (x, y, z). Used only by
tests/test_rule.py; not for production rendering.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import map_coordinates


def font_glyph(ch: str = "R", scale: float = 1.2, thickness: int = 1, pad: int = 2) -> np.ndarray:
    """Font-rendered capital (cv2 Hershey simplex), cropped, float32 in [0, 1],
    1 = ink, oriented as it reads (row 0 = top)."""
    canvas = np.zeros((80, 80), np.uint8)
    cv2.putText(canvas, ch, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, scale, 255, thickness, cv2.LINE_AA)
    ys, xs = np.nonzero(canvas)
    g = canvas[ys.min() - pad:ys.max() + pad + 1, xs.min() - pad:xs.max() + pad + 1]
    return (g.astype(np.float32) / 255.0)


@dataclass
class Spiral:
    centre: tuple[float, float]  # (cx, cy)
    r_outer: float
    pitch: float  # radial distance between windings
    turns: float
    phi_start: float  # azimuth of the outer end
    w: int  # +1: moving inward rotates counter-clockwise (phi increases); -1 opposite

    def radius(self, theta: np.ndarray) -> np.ndarray:
        return self.r_outer - self.pitch * np.asarray(theta) / (2 * np.pi)

    def theta_at(self, phi: np.ndarray, k: int) -> np.ndarray:
        """Unwrapped spiral parameter of winding k (0 = outermost) at azimuth phi."""
        base = np.mod(self.w * (np.asarray(phi) - self.phi_start), 2 * np.pi)
        return base + 2 * np.pi * k

    def phi_of_theta(self, theta):
        return self.phi_start + self.w * np.asarray(theta)


@dataclass
class GlyphScene:
    vol: np.ndarray
    spiral: Spiral
    glyph: np.ndarray
    pix: float
    k_glyph: int
    phi_g: float
    z_c: float


def spiral_glyph_volume(shape_zyx, spiral: Spiral, glyph: np.ndarray, *, k_glyph: int, phi_g: float, z_c: float,
                        pix: float, sigma: float = 1.0, ink_offset: float = 1.2, ink_sigma: float = 0.9,
                        ink_amp: float = 1.0, base: float = 0.3, envelope: float = 0.02) -> GlyphScene:
    """Density = sheets (Gaussian across the sheet) + an ink bump on the inner
    face of winding ``k_glyph`` carrying ``glyph`` (reading: col -> +phi,
    row -> +z), + a faint envelope inside the outer winding so the 'masked'
    volume is non-zero inside the scroll (as organizer masked volumes are)."""
    Z, Y, X = shape_zyx
    zz, yy, xx = np.meshgrid(np.arange(Z, dtype=np.float64), np.arange(Y, dtype=np.float64),
                             np.arange(X, dtype=np.float64), indexing="ij")
    dx, dy = xx - spiral.centre[0], yy - spiral.centre[1]
    r = np.hypot(dx, dy)
    phi = np.arctan2(dy, dx)
    nk = int(math.ceil(spiral.turns)) + 1
    dens = np.zeros_like(r)
    theta_max = 2 * np.pi * spiral.turns
    for k in range(nk):
        th = spiral.theta_at(phi, k)
        rk = spiral.radius(th)
        valid = th <= theta_max
        dens += np.where(valid, np.exp(-0.5 * ((r - rk) / sigma) ** 2), 0.0)
    vol = base * dens
    # ink on the recto (inner face, r = r_k - ink_offset) of winding k_glyph
    th = spiral.theta_at(phi, k_glyph)
    rk = spiral.radius(th)
    dphi = (phi - phi_g + np.pi) % (2 * np.pi) - np.pi
    r_g = float(spiral.radius(spiral.theta_at(np.array(phi_g), k_glyph)))
    col = dphi * r_g / pix + (glyph.shape[1] - 1) / 2
    row = (zz - z_c) / pix + (glyph.shape[0] - 1) / 2
    ink = map_coordinates(glyph, np.stack([row.ravel(), col.ravel()]), order=1, mode="constant",
                          cval=0.0).reshape(r.shape)
    vol += ink_amp * ink * np.exp(-0.5 * ((r - (rk - ink_offset)) / ink_sigma) ** 2)
    vol += envelope * (r < spiral.r_outer + 2)
    return GlyphScene(vol=vol.astype(np.float32), spiral=spiral, glyph=glyph, pix=pix, k_glyph=k_glyph,
                      phi_g=phi_g, z_c=z_c)


def spiral_sheet_grid(spiral: Spiral, k: int, shape_hw, step: float, angle_deg: float, *, s0: float, z0: float,
                      phi_ref: float) -> np.ndarray:
    """(H, W, 3) grid on winding k: arc s (along +phi) and z rotated by
    ``angle_deg``: s = step (u cos a - v sin a) + s0, z = step (u sin a +
    v cos a) + z0; phi = phi_ref + s / r(phi_ref). du x dv points outward for
    |a| < 90 deg (u ~ +phi, v ~ +z)."""
    H, W = shape_hw
    a = np.radians(angle_deg)
    vv, uu = np.mgrid[0:H, 0:W].astype(np.float64)
    s = step * (uu * np.cos(a) - vv * np.sin(a)) + s0
    z = step * (uu * np.sin(a) + vv * np.cos(a)) + z0
    r_ref = float(spiral.radius(spiral.theta_at(np.array(phi_ref), k)))
    phi = phi_ref + s / r_ref
    # follow winding k continuously (no wrap of theta_at across phi_start)
    th0 = spiral.theta_at(np.array(phi_ref), k)
    th = th0 + spiral.w * (phi - phi_ref)
    r = spiral.radius(th)
    return np.stack([spiral.centre[0] + r * np.cos(phi), spiral.centre[1] + r * np.sin(phi), z], axis=-1).astype(np.float32)


def spiral_row_grid(spiral: Spiral, n_u: int, n_v: int, z0: float, dz: float, theta0: float, dtheta: float) -> np.ndarray:
    """Grid following the spiral itself along u (theta0 + i dtheta, possibly
    several turns) and z along v; used for winding-sense tests."""
    th = theta0 + dtheta * np.arange(n_u)
    phi = spiral.phi_of_theta(th)
    r = spiral.radius(th)
    x = spiral.centre[0] + r * np.cos(phi)
    y = spiral.centre[1] + r * np.sin(phi)
    z = z0 + dz * np.arange(n_v)
    P = np.empty((n_v, n_u, 3), np.float32)
    P[..., 0] = x[None]
    P[..., 1] = y[None]
    P[..., 2] = z[:, None]
    return P


# --- frame transforms of a whole scene (volume + mesh + metadata) ----------

def transform_scene(vol: np.ndarray, pts: np.ndarray, kind: str):
    """Return (vol', pts', props') for:
    'ascending'  : as built; text top at low z; right-handed.
    'rot180x'    : scroll mounted upside down = proper 180 deg rotation about
                   x (z -> Z-1-z, y -> Y-1-y); text top at HIGH z; right-handed.
    'reflect_z'  : slice order reversed only (z -> Z-1-z): a reflection; text
                   top at high z; the metadata must declare left-handed.
    'reflect_x'  : x -> X-1-x only (a reflection in-plane); text top at low z;
                   left-handed.
    """
    Z, Y, X = vol.shape
    p = np.array(pts, np.float32, copy=True)
    if kind == "ascending":
        return vol, p, dict(left_handed_coordinates=False, z_direction_is_top_to_bottom=True)
    if kind == "rot180x":
        p[..., 2] = (Z - 1) - p[..., 2]
        p[..., 1] = (Y - 1) - p[..., 1]
        return np.ascontiguousarray(vol[::-1, ::-1]), p, dict(left_handed_coordinates=False,
                                                              z_direction_is_top_to_bottom=False)
    if kind == "reflect_z":
        p[..., 2] = (Z - 1) - p[..., 2]
        return np.ascontiguousarray(vol[::-1]), p, dict(left_handed_coordinates=True,
                                                        z_direction_is_top_to_bottom=False)
    if kind == "reflect_x":
        p[..., 0] = (X - 1) - p[..., 0]
        return np.ascontiguousarray(vol[:, :, ::-1]), p, dict(left_handed_coordinates=True,
                                                              z_direction_is_top_to_bottom=True)
    raise ValueError(kind)


ZATTRS_ZYX = {"multiscales": [{"axes": [{"name": "z"}, {"name": "y"}, {"name": "x"}]}]}


def centroid_curve_from_volume(vol: np.ndarray, thresh: float = 0.01, z_stride: int = 2, sigma: float = 8.0):
    """CT-mask centroid per slice of an in-memory synthetic volume (z-smoothed),
    the synthetic stand-in for the level-4 centroid of an organizer volume."""
    from .level_mesh import CentreCurve

    zs, xs, ys = [], [], []
    for z in range(0, vol.shape[0], z_stride):
        yy, xx = np.nonzero(vol[z] > thresh)
        if yy.size:
            zs.append(float(z))
            xs.append(float(xx.mean()))
            ys.append(float(yy.mean()))
    z = np.array(zs)
    w = np.exp(-0.5 * ((z[:, None] - z[None, :]) / sigma) ** 2)
    w /= w.sum(1, keepdims=True)
    return CentreCurve(z=z, x=w @ np.array(xs), y=w @ np.array(ys))
