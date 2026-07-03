"""
rf_shadow.py
------------
Computes which parts of the terrain are in line-of-sight (LOS) of a radio
transmitter (the EW/jammer source) versus which parts are shielded by the
terrain itself ("radio shadow" / non-line-of-sight, NLOS).

Core idea: for every grid cell, we sample the terrain profile along the
straight line connecting the transmitter and that cell. If the direct
sight line stays above the ground everywhere along that profile, the cell
is exposed (LOS). If the ground rises above the sight line anywhere along
the way, the cell is shielded (NLOS / radio shadow).

This is a simplified viewshed / intervisibility analysis. It ignores
atmospheric refraction and Earth curvature (both negligible at the
short ranges relevant to a small UAV evading a ground-based jammer,
typically < 5-10 km).

Fully vectorized with numpy: instead of looping over target cells, we
loop over sample points along the sight line (a small constant number)
and evaluate all target cells at once for each sample.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


def bilinear_sample(grid: np.ndarray, rs: np.ndarray, cs: np.ndarray) -> np.ndarray:
    """
    Bilinearly sample `grid` at fractional (row, col) coordinates `rs`, `cs`.
    `rs` and `cs` can be arrays of any (matching) shape.
    """
    rows, cols = grid.shape
    rs = np.clip(rs, 0, rows - 1 - 1e-6)
    cs = np.clip(cs, 0, cols - 1 - 1e-6)

    r0 = np.floor(rs).astype(int)
    c0 = np.floor(cs).astype(int)
    r1 = r0 + 1
    c1 = c0 + 1
    fr = rs - r0
    fc = cs - c0

    v00 = grid[r0, c0]
    v01 = grid[r0, c1]
    v10 = grid[r1, c0]
    v11 = grid[r1, c1]

    top = v00 * (1 - fc) + v01 * fc
    bot = v10 * (1 - fc) + v11 * fc
    return top * (1 - fr) + bot * fr


@dataclass
class ShadowResult:
    visible: np.ndarray       # bool grid, True = line-of-sight to source (exposed)
    clearance: np.ndarray     # meters of clearance above terrain (negative = blocked)

    @property
    def shadow(self) -> np.ndarray:
        """True where the cell is SHIELDED from the source (radio shadow)."""
        return ~self.visible


def compute_los_grid(
    elevation: np.ndarray,
    source_rc: tuple[int, int],
    source_height_agl: float,
    target_height_agl: float,
    n_samples: int = 150,
    clearance_tolerance: float = 1.0,
) -> ShadowResult:
    """
    Compute line-of-sight visibility from a single source point to every
    cell in the grid.

    Parameters
    ----------
    elevation : (rows, cols) terrain elevation in meters.
    source_rc : (row, col) grid index of the transmitter.
    source_height_agl : transmitter antenna height above ground, meters
        (e.g. a jammer mounted on a mast/vehicle: 2-10 m).
    target_height_agl : receiver height above ground, meters (drone
        flight altitude above local terrain).
    n_samples : number of points sampled along each sight line. Should be
        >= the grid diagonal in cells for good accuracy.
    clearance_tolerance : meters of slack allowed before a cell is
        considered blocked (accounts for grid discretization error).

    Returns
    -------
    ShadowResult with `visible` (LOS) and `clearance` (meters) grids.
    """
    rows, cols = elevation.shape
    r0, c0 = source_rc

    row_idx, col_idx = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")

    src_elev = elevation[r0, c0] + source_height_agl
    dst_elev = elevation + target_height_agl  # per-cell target elevation

    min_clearance = np.full((rows, cols), np.inf)

    # skip t=0 (the source itself - trivially has clearance == source_height_agl)
    for t in np.linspace(0.0, 1.0, n_samples)[1:]:
        rs = r0 + (row_idx - r0) * t
        cs = c0 + (col_idx - c0) * t

        terrain_t = bilinear_sample(elevation, rs, cs)
        sight_t = src_elev + (dst_elev - src_elev) * t
        clearance_t = sight_t - terrain_t

        min_clearance = np.minimum(min_clearance, clearance_t)

    visible = min_clearance >= -clearance_tolerance
    return ShadowResult(visible=visible, clearance=min_clearance)


def nearest_shadow_cell(
    shadow_mask: np.ndarray, from_rc: tuple[int, int]
) -> tuple[int, int] | None:
    """Find the closest shadow (NLOS) cell to a given grid position (Euclidean)."""
    ys, xs = np.where(shadow_mask)
    if len(ys) == 0:
        return None
    r0, c0 = from_rc
    d2 = (ys - r0) ** 2 + (xs - c0) ** 2
    k = np.argmin(d2)
    return int(ys[k]), int(xs[k])
