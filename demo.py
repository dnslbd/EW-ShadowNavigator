"""
demo.py
-------
Standalone, dependency-light demonstration of the full EW-ShadowNavigator
pipeline: terrain -> radio-shadow map -> Bayesian jammer localization ->
adaptive risk-aware path planning.

Unlike app.py (the interactive Streamlit/pydeck version), this script
only needs numpy/scipy/matplotlib, so it's a good first thing to run to
confirm your environment is set up correctly, and produces a static
PNG report you can drop straight into a README, a slide deck, or a
LinkedIn post.

Usage:
    python demo.py
Output:
    assets/mission_report.png
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from terrain import generate_synthetic_dem
from rf_shadow import compute_los_grid
from pathfinding import astar, build_risk_grid, path_exposure_fraction
from mission import run_adaptive_mission


# ----------------------------------------------------------------------
# Scenario configuration - tweak these to explore different terrains
# ----------------------------------------------------------------------
DEM_SIZE = 101
DEM_SEED = 11
DEM_ROUGHNESS = 0.62
DEM_HEIGHT_SCALE = 140.0
DEM_CELLSIZE = 25.0

JAMMER_RC = (28, 44)
JAMMER_HEIGHT_AGL = 8.0
DRONE_HEIGHT_AGL = 10.0

START = (5, 5)
GOAL = (95, 95)
REPLAN_EVERY = 12
MISSION_SEED = 1


def main() -> None:
    print("Generating synthetic terrain...")
    dem = generate_synthetic_dem(
        size=DEM_SIZE, seed=DEM_SEED, roughness=DEM_ROUGHNESS,
        height_scale=DEM_HEIGHT_SCALE, cellsize=DEM_CELLSIZE,
    )
    elev = dem.elevation
    print(f"  grid: {elev.shape}, cellsize: {dem.cellsize} m, "
          f"extent: {dem.extent_meters()[0]/1000:.2f} x {dem.extent_meters()[1]/1000:.2f} km, "
          f"relief: {elev.max()-elev.min():.0f} m")

    print("Computing ground-truth radio-shadow map...")
    true_shadow = compute_los_grid(
        elev, JAMMER_RC, JAMMER_HEIGHT_AGL, DRONE_HEIGHT_AGL, n_samples=150
    )
    print(f"  {true_shadow.shadow.mean()*100:.1f}% of the map is in radio shadow")

    print("Running adaptive mission (localization + replanning)...")
    result = run_adaptive_mission(
        elevation=elev, cellsize=dem.cellsize, start=START, goal=GOAL,
        jammer_true_rc=JAMMER_RC, jammer_height_agl=JAMMER_HEIGHT_AGL,
        drone_height_agl=DRONE_HEIGHT_AGL, replan_every=REPLAN_EVERY,
        seed=MISSION_SEED,
    )
    print(f"  reached goal: {result.reached_goal}")
    print(f"  localization error: {result.localization_error_m:.0f} m "
          f"(confidence {result.final_confidence:.2f})")
    print(f"  exposure along flown path: {result.exposure_fraction*100:.1f}%")

    # baseline: what if the drone ignored RF risk entirely and flew direct?
    naive_path = astar(np.ones_like(elev), START, GOAL)
    naive_exposure = path_exposure_fraction(naive_path, true_shadow.shadow)
    print(f"  [baseline] naive direct-path exposure: {naive_exposure*100:.1f}%")

    print("Rendering mission report...")
    render_report(dem, true_shadow, result, naive_path, naive_exposure)
    print("Saved to assets/mission_report.png")


def render_report(dem, true_shadow, result, naive_path, naive_exposure):
    elev = dem.elevation
    fig, axes = plt.subplots(2, 2, figsize=(13, 12))

    # --- panel 1: terrain + scenario setup
    ax = axes[0, 0]
    im = ax.imshow(elev, cmap="terrain")
    ax.scatter(*JAMMER_RC[::-1], c="red", marker="^", s=160, edgecolor="black", label="Jammer (true)", zorder=5)
    ax.scatter(START[1], START[0], c="black", marker="o", s=80, label="Start", zorder=5)
    ax.scatter(GOAL[1], GOAL[0], c="black", marker="*", s=160, label="Goal", zorder=5)
    ax.set_title(f"1. Terrain (relief {elev.max()-elev.min():.0f} m)")
    ax.legend(loc="upper right", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, label="m")

    # --- panel 2: ground-truth radio shadow
    ax = axes[0, 1]
    ax.imshow(elev, cmap="gray", alpha=0.5)
    ax.imshow(np.where(true_shadow.shadow, 1, np.nan), cmap="Greens", alpha=0.8, vmin=0, vmax=1)
    ax.scatter(*JAMMER_RC[::-1], c="red", marker="^", s=160, edgecolor="black", zorder=5)
    ax.set_title(f"2. Ground-truth radio shadow ({true_shadow.shadow.mean()*100:.0f}% of area)")

    # --- panel 3: Bayesian belief evolution (final state)
    ax = axes[1, 0]
    # reconstruct final belief isn't stored directly on result; approximate via
    # a fresh localizer replay is unnecessary for the report - show estimate vs truth instead
    ax.imshow(elev, cmap="gray", alpha=0.4)
    confidences = [s.confidence for s in result.steps]
    ax.plot(range(len(confidences)), confidences, marker="o", color="tab:blue")
    ax.set_xlabel("Replan step")
    ax.set_ylabel("Localization confidence (peak belief)")
    ax.set_title("3. Localization confidence over the mission")
    ax.grid(alpha=0.3)

    # --- panel 4: flown trajectory vs naive baseline
    ax = axes[1, 1]
    ax.imshow(elev, cmap="gray", alpha=0.5)
    ax.imshow(np.where(true_shadow.shadow, 1, np.nan), cmap="Greens", alpha=0.6, vmin=0, vmax=1)
    ys, xs = zip(*naive_path)
    ax.plot(xs, ys, c="orange", lw=2, ls="--", label=f"Naive direct ({naive_exposure*100:.0f}% exposed)")
    ys, xs = zip(*result.trajectory)
    ax.plot(xs, ys, c="blue", lw=2.5, label=f"Adaptive EW-ShadowNav ({result.exposure_fraction*100:.0f}% exposed)")
    ax.scatter(*JAMMER_RC[::-1], c="red", marker="^", s=160, edgecolor="black", zorder=5)
    ax.scatter(*result.final_estimate[::-1], c="cyan", marker="x", s=140, linewidths=3,
               label=f"Final jammer estimate (err {result.localization_error_m:.0f} m)", zorder=5)
    ax.scatter(START[1], START[0], c="black", marker="o", s=80, zorder=5)
    ax.scatter(GOAL[1], GOAL[0], c="black", marker="*", s=160, zorder=5)
    ax.legend(loc="upper left", fontsize=7.5)
    ax.set_title("4. Flown route vs. naive baseline")

    fig.suptitle("EW-ShadowNavigator - Mission Report (synthetic terrain demo)", fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig("assets/mission_report.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
