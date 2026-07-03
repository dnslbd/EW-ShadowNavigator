"""
mission.py
----------
Ties localization + pathfinding into a single adaptive mission loop, so
both the CLI demo (demo.py) and the interactive Streamlit app (app.py)
can drive the same simulation logic.

Mission loop (each iteration):
  1. The drone takes an RSSI measurement at its current position
     (simulated against the TRUE jammer position - this represents the
     real world "sensor reading").
  2. That measurement updates the Bayesian belief about the jammer's
     location (the drone's internal, imperfect estimate).
  3. A risk map is (re)built from the *current estimate*, and A* replans
     a route from the drone's current position to the goal.
  4. The drone flies the next `replan_every` steps of that route, then
     the loop repeats - so the route keeps adapting as the belief
     sharpens, exactly like the "dynamic replanning" behavior described
     in the project concept (D* Lite was the original inspiration; see
     pathfinding.py for why we use periodic A* here instead).

The result is a full realized trajectory that can be compared against
the ground-truth shadow map for a final "how well did we do" score.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from rf_shadow import compute_los_grid, ShadowResult
from localization import BayesianLocalizer, PathLossModel
from pathfinding import astar, build_risk_grid, path_exposure_fraction


@dataclass
class MissionStep:
    drone_pos: tuple[int, int]
    measured_rssi: float
    jammer_estimate: tuple[int, int]
    confidence: float
    planned_segment: list[tuple[int, int]]


@dataclass
class MissionResult:
    trajectory: list[tuple[int, int]]
    steps: list[MissionStep]
    true_shadow: ShadowResult
    final_estimate: tuple[int, int]
    final_confidence: float
    localization_error_m: float
    exposure_fraction: float
    reached_goal: bool


def run_adaptive_mission(
    elevation: np.ndarray,
    cellsize: float,
    start: tuple[int, int],
    goal: tuple[int, int],
    jammer_true_rc: tuple[int, int],
    jammer_height_agl: float = 8.0,
    drone_height_agl: float = 10.0,
    replan_every: int = 12,
    exposure_penalty: float = 6.0,
    max_iterations: int = 40,
    seed: int = 0,
    model: PathLossModel | None = None,
) -> MissionResult:
    """Run one full adaptive mission from `start` to `goal`."""
    rng = np.random.default_rng(seed)
    model = model or PathLossModel()

    # ground truth, used only to simulate realistic sensor readings and to
    # score the mission afterwards - the drone's planner never sees this directly
    true_shadow = compute_los_grid(
        elevation, jammer_true_rc, jammer_height_agl, drone_height_agl, n_samples=150
    )

    localizer = BayesianLocalizer(
        elevation=elevation,
        cellsize=cellsize,
        jammer_height_agl=jammer_height_agl,
        drone_height_agl=drone_height_agl,
        model=model,
    )

    trajectory = [start]
    steps: list[MissionStep] = []
    current = start
    reached_goal = False

    for _ in range(max_iterations):
        # 1. take a measurement at the current position (against ground truth)
        true_dist = np.hypot(current[0] - jammer_true_rc[0], current[1] - jammer_true_rc[1]) * cellsize
        blocked_here = bool(true_shadow.shadow[current])
        measured = model.simulate_measurement(true_dist, blocked_here, rng)
        localizer.update(current, measured)

        estimate = tuple(int(v) for v in localizer.estimate())
        confidence = localizer.confidence()

        # 2. build a risk map from the CURRENT estimate and replan from here
        est_shadow = compute_los_grid(
            elevation, estimate, jammer_height_agl, drone_height_agl, n_samples=100
        )
        risk = build_risk_grid(est_shadow.shadow, exposure_penalty=exposure_penalty)
        path = astar(risk, current, goal)

        if not path or len(path) < 2:
            reached_goal = (current == goal)
            break

        # 3. fly the next `replan_every` steps of that plan (or all of it, if shorter)
        segment = path[: replan_every + 1]
        steps.append(
            MissionStep(
                drone_pos=current,
                measured_rssi=measured,
                jammer_estimate=estimate,
                confidence=confidence,
                planned_segment=segment,
            )
        )
        trajectory.extend(segment[1:])
        current = segment[-1]

        if current == goal:
            reached_goal = True
            break

    final_estimate = tuple(int(v) for v in localizer.estimate())
    err_m = float(
        np.hypot(final_estimate[0] - jammer_true_rc[0], final_estimate[1] - jammer_true_rc[1])
        * cellsize
    )
    exposure = path_exposure_fraction(trajectory, true_shadow.shadow)

    return MissionResult(
        trajectory=trajectory,
        steps=steps,
        true_shadow=true_shadow,
        final_estimate=final_estimate,
        final_confidence=localizer.confidence(),
        localization_error_m=err_m,
        exposure_fraction=exposure,
        reached_goal=reached_goal,
    )
