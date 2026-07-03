"""
localization.py
----------------
Bayesian grid localization of the jamming source, using noisy RSSI
(received signal strength) measurements taken by the drone as it moves.

The drone does NOT know the jammer's true position. Instead, it
maintains a belief - a probability distribution over every grid cell,
representing "how likely is the jammer to be here?". Every time the
drone takes a new RSSI measurement, that belief is updated using
Bayes' rule:

    belief(cell) <- belief(cell) * P(measured_rssi | jammer_at cell)

`P(measured_rssi | jammer_at cell)` is computed from a standard
log-distance path-loss propagation model, plus an extra attenuation
term if the terrain would block line-of-sight between the drone's
current position and that candidate cell (reusing the same
`compute_los_grid` machinery as the shadow map - line-of-sight is
reciprocal, so "can the candidate cell see the drone" is the same
computation as "can the drone see the candidate cell").

This is the standard "RSSI fingerprinting" / grid-based Bayes filter
technique used in radio direction-finding, indoor positioning and
wildlife-tracking beacon localization - not something specific to any
weapon system.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from .rf_shadow import compute_los_grid


@dataclass
class PathLossModel:
    """Simple log-distance path-loss model with terrain-blockage penalty."""

    tx_power_dbm: float = 30.0      # effective transmit power at 1 m, dBm
    path_loss_exponent: float = 3.0  # 2 = free space, 3-4 = cluttered terrain
    reference_distance_m: float = 1.0
    nlos_extra_loss_db: float = 25.0  # extra attenuation if terrain-blocked
    noise_std_db: float = 4.0        # measurement + shadow-fading noise (std dev)

    def expected_rssi(self, distance_m: np.ndarray, blocked: np.ndarray) -> np.ndarray:
        distance_m = np.maximum(distance_m, self.reference_distance_m)
        path_loss = 10 * self.path_loss_exponent * np.log10(
            distance_m / self.reference_distance_m
        )
        rssi = self.tx_power_dbm - path_loss
        rssi = np.where(blocked, rssi - self.nlos_extra_loss_db, rssi)
        return rssi

    def simulate_measurement(
        self, distance_m: float, blocked: bool, rng: np.random.Generator
    ) -> float:
        mean = self.expected_rssi(np.array(distance_m), np.array(blocked))
        return float(mean + rng.normal(0, self.noise_std_db))


@dataclass
class BayesianLocalizer:
    """Grid-based Bayesian filter for jammer localization."""

    elevation: np.ndarray
    cellsize: float
    jammer_height_agl: float
    drone_height_agl: float
    model: PathLossModel = field(default_factory=PathLossModel)
    belief: np.ndarray = field(init=False)

    def __post_init__(self):
        rows, cols = self.elevation.shape
        self.belief = np.ones((rows, cols)) / (rows * cols)

    def update(self, drone_rc: tuple[int, int], measured_rssi_dbm: float) -> None:
        """Fold in one new RSSI measurement taken at the drone's current cell."""
        rows, cols = self.elevation.shape
        r0, c0 = drone_rc

        # distance (in meters) from the drone's position to every candidate cell
        row_idx, col_idx = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
        dist_m = (
            np.hypot(row_idx - r0, col_idx - c0) * self.cellsize
        )

        # LOS/NLOS from the drone's current position to every candidate cell
        # (line-of-sight is reciprocal, so this doubles as "would a jammer at
        # this candidate cell be able to see the drone right now")
        los = compute_los_grid(
            self.elevation,
            source_rc=drone_rc,
            source_height_agl=self.drone_height_agl,
            target_height_agl=self.jammer_height_agl,
            n_samples=100,
        )
        blocked = los.shadow

        expected = self.model.expected_rssi(dist_m, blocked)
        diff = measured_rssi_dbm - expected
        likelihood = np.exp(-0.5 * (diff / self.model.noise_std_db) ** 2)

        self.belief *= likelihood
        total = self.belief.sum()
        if total > 0:
            self.belief /= total
        else:
            # numerical safety net: reset to uniform if belief collapsed to 0
            self.belief[:] = 1.0 / self.belief.size

    def estimate(self) -> tuple[int, int]:
        """Return the current maximum-a-posteriori (MAP) jammer position estimate."""
        idx = np.argmax(self.belief)
        return np.unravel_index(idx, self.belief.shape)

    def confidence(self) -> float:
        """Peak belief value - a rough proxy for localization confidence (0..1)."""
        return float(self.belief.max())
