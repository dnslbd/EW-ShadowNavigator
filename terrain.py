"""
terrain.py
----------
Provides a Digital Elevation Model (DEM) for the navigation pipeline.

Two sources are supported:
1. `generate_synthetic_dem`  - procedurally generated terrain (diamond-square /
   midpoint displacement fractal algorithm). No external data or network
   access required. Ideal for rapid development, testing and demos.
2. `load_real_dem`           - loads a real GeoTIFF DEM (e.g. an SRTM tile)
   via `rasterio`, for when you want to run the pipeline over an actual
   location. `rasterio` is an optional dependency - the rest of the
   pipeline only ever consumes a plain numpy array, so switching between
   the two sources requires no other code changes.

Elevation is always returned in meters, on a square grid with a fixed
`cellsize` (meters per grid cell).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class DEM:
    """Container for a Digital Elevation Model."""

    elevation: np.ndarray   # shape (rows, cols), meters
    cellsize: float         # meters per grid cell (assumed square cells)
    source: str = "synthetic"  # "synthetic" or path to the source file

    @property
    def shape(self):
        return self.elevation.shape

    def extent_meters(self):
        rows, cols = self.shape
        return rows * self.cellsize, cols * self.cellsize


def _next_pow2_plus_one(n: int) -> int:
    """Smallest size (2**k + 1) >= n, required by the diamond-square algorithm."""
    k = 1
    while (2 ** k) + 1 < n:
        k += 1
    return (2 ** k) + 1


def generate_synthetic_dem(
    size: int = 129,
    seed: int = 42,
    roughness: float = 0.55,
    base_height: float = 180.0,
    height_scale: float = 70.0,
    cellsize: float = 20.0,
) -> DEM:
    """
    Generate a synthetic terrain using the diamond-square (midpoint
    displacement) fractal algorithm. Produces believable rolling
    hills/valleys - enough structure for line-of-sight / radio-shadow
    experiments without needing any real map data.

    Parameters
    ----------
    size : requested grid size (will be rounded up to 2**k + 1).
    seed : RNG seed, for reproducible terrain.
    roughness : 0..1, higher = rougher / more jagged terrain.
    base_height : mean elevation in meters.
    height_scale : controls total relief (peak-to-valley range), in meters.
    cellsize : meters represented by one grid cell (spatial resolution).
    """
    n = _next_pow2_plus_one(size)
    rng = np.random.default_rng(seed)

    grid = np.zeros((n, n), dtype=np.float64)

    # seed the four corners
    grid[0, 0] = rng.uniform(-1, 1)
    grid[0, -1] = rng.uniform(-1, 1)
    grid[-1, 0] = rng.uniform(-1, 1)
    grid[-1, -1] = rng.uniform(-1, 1)

    step = n - 1
    scale = 1.0

    while step > 1:
        half = step // 2

        # --- diamond step: center of each square = avg of 4 corners + noise
        for i in range(half, n, step):
            for j in range(half, n, step):
                avg = (
                    grid[i - half, j - half]
                    + grid[i - half, j + half]
                    + grid[i + half, j - half]
                    + grid[i + half, j + half]
                ) / 4.0
                grid[i, j] = avg + rng.uniform(-1, 1) * scale

        # --- square step: midpoints of each edge = avg of surrounding diamond pts
        for i in range(0, n, half):
            start = half if (i // half) % 2 == 0 else 0
            for j in range(start, n, step):
                total, count = 0.0, 0
                if i - half >= 0:
                    total += grid[i - half, j]
                    count += 1
                if i + half < n:
                    total += grid[i + half, j]
                    count += 1
                if j - half >= 0:
                    total += grid[i, j - half]
                    count += 1
                if j + half < n:
                    total += grid[i, j + half]
                    count += 1
                grid[i, j] = total / count + rng.uniform(-1, 1) * scale

        step = half
        scale *= 2 ** (-roughness)

    # normalize to [0, 1] then map to real-world elevation range
    grid -= grid.min()
    if grid.max() > 0:
        grid /= grid.max()
    elevation = base_height + grid * height_scale

    # crop back to the originally requested size if we rounded up
    elevation = elevation[:size, :size]

    return DEM(elevation=elevation, cellsize=cellsize, source="synthetic")


def load_real_dem(path: str, band: int = 1) -> DEM:
    """
    Load a real-world DEM from a GeoTIFF file (e.g. an SRTM/ASTER tile)
    using `rasterio`. Requires `pip install rasterio`.

    This is intentionally isolated in its own function: if rasterio is
    not installed, only this function fails - the rest of the app keeps
    working with `generate_synthetic_dem`.
    """
    try:
        import rasterio
    except ImportError as exc:
        raise ImportError(
            "rasterio is required to load real DEM files. "
            "Install it with: pip install rasterio"
        ) from exc

    with rasterio.open(path) as src:
        elevation = src.read(band).astype(np.float64)
        # assume square pixels; take the x resolution as cellsize
        cellsize = abs(src.transform.a)

    return DEM(elevation=elevation, cellsize=float(cellsize), source=path)
