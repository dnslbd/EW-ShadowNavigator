# EW-ShadowNavigator

**Inversive UAV navigation via jammer-signal terrain masking.**

Turns an electronic-warfare (EW) jamming source into a navigation beacon
instead of a threat: the UAV localizes the jammer purely from its own
emissions, then routes through terrain-masked "radio shadow" zones to
avoid it — no GPS required.

![Mission report](assets/mission_report.png)
*Full pipeline on a synthetic demo terrain: ground-truth radio shadow (top
right), localization confidence converging over the mission (bottom
left), and the actually-flown adaptive route vs. a naive direct path
(bottom right). See "Honest results" below for what these numbers mean.*

## The core idea

To jam GPS or radio, a station must broadcast a large amount of power —
it "lights up" the RF spectrum like a searchlight. That is also its
weakness: **the jammer cannot hide.**

1. **Localize the threat.** The drone takes RSSI (signal strength)
   readings as it moves. A grid-based Bayesian filter combines these
   noisy readings with a log-distance path-loss model to converge on the
   jammer's position — passively, with no GPS and no active radar.
2. **Find the radio shadow.** Given that position estimate, a
   line-of-sight analysis over a Digital Elevation Model (DEM) finds
   every point hills and ridges shield from the jammer — radio waves are
   blocked by terrain just like light is.
3. **Route through it.** An A* planner threads a path from start to
   goal that prefers those shadow cells, re-planning periodically as the
   localization estimate sharpens — the route gets safer the more the
   drone learns.

This is a defensive, passive navigation technique: it does not target,
track, or interfere with anything — it only helps a UAV find its own way
home when GPS is denied. The same technique applies directly to civilian
autonomous logistics in increasingly congested RF environments.

## Architecture

```
src/
  terrain.py        DEM source: synthetic (diamond-square fractal) or
                     real GeoTIFF via rasterio
  rf_shadow.py       Line-of-sight / radio-shadow analysis (vectorized)
  localization.py    Bayesian grid filter for RSSI-based localization
  pathfinding.py     Risk-weighted A* search
  mission.py         Ties it together: the adaptive measure -> localize
                     -> replan -> fly loop
app.py               Interactive Streamlit + pydeck 3D front-end
demo.py              Dependency-light CLI demo -> assets/mission_report.png
```

Each module in `src/` only depends on numpy/scipy and has no UI code, so
the same logic drives both the CLI demo and the interactive app.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt

# Fast, dependency-light sanity check (numpy/scipy/matplotlib only):
python demo.py            # -> assets/mission_report.png

# Full interactive 3D app:
streamlit run app.py
```

## Deploying for free (Streamlit Community Cloud)

1. Push this repo to GitHub (public).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** →
   select the repo and `app.py`.
3. You get a public URL — anyone can open it in a browser and interact
   with the simulation with no local install.

## Using a real DEM instead of synthetic terrain

`src/terrain.py` also exposes `load_real_dem(path)`, which reads any
GeoTIFF elevation file via `rasterio` (e.g. an SRTM tile from
[OpenTopography](https://opentopography.org) or USGS EarthExplorer). The
Streamlit app's sidebar already has an "Upload real GeoTIFF" option wired
up to it — the rest of the pipeline is agnostic to where the elevation
grid came from. Uncomment `rasterio` in `requirements.txt` to enable it.

## Honest results (and current scope)

Numbers from the bundled demo scenario (synthetic terrain, 2.5×2.5 km,
122 m relief) — reproduced by `python demo.py`:

| Metric | Value |
|---|---|
| Ground-truth radio shadow coverage | 13.9% of the map |
| Final localization error | 0 m (this seed) — typically converges to 1-2 grid cells |
| Final localization confidence | 55% |
| Exposure, naive direct path | 89.0% |
| Exposure, adaptive route (realistic, learns while flying) | 70.4% |
| Exposure, route if jammer position were known from the start | 49.3% |

The gap between the "realistic" and "if known from the start" rows is
expected and, we think, an honest and interesting result: early in a
mission the drone hasn't yet localized the threat, so it can't route
around it optimally yet — it improves its route as its confidence grows.
That explore-then-exploit curve is visible directly in the bottom-left
panel of the mission report above.

**What's simplified for this stage, on purpose:**
- The planner uses periodic A* replanning rather than incremental
  D\* Lite. Same adaptive behavior, far less implementation risk in the
  available time; `mission.py` isolates this so swapping in D\* Lite
  later only touches one function.
- Terrain defaults to procedurally generated synthetic DEMs so the
  project is instantly runnable and reproducible with no downloads;
  real-DEM support is implemented and wired into the UI, ready for a
  real SRTM tile.
- ONNX export for on-board (Jetson/RPi-class) deployment is a planned
  next step, not yet implemented.

## Roadmap

- [ ] Swap periodic A* for incremental D\* Lite replanning
- [ ] Export the decision core via ONNX Runtime for on-board deployment
- [ ] Validate against a real DEM tile (Kyiv region / Carpathians)
- [ ] Package the core as a small FastAPI service for integration with
      Mission Planner / QGroundControl

## License

MIT — see `LICENSE`.
