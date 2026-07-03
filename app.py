"""
app.py
------
Interactive Streamlit + pydeck front-end for EW-ShadowNavigator.

Run locally with:
    pip install -r requirements.txt
    streamlit run app.py

Deploy for free on Streamlit Community Cloud:
    1. Push this folder to a public GitHub repo.
    2. Go to https://share.streamlit.io -> "New app" -> pick the repo -> app.py.
    3. Done - you get a public URL judges can open in their browser, no
       install required on their end.

NOTE: this file was written and carefully reviewed, but could not be
executed inside the development sandbox used to build this project
(streamlit/pydeck aren't installable there - no network access). All the
actual math (terrain, radio-shadow, localization, path planning) lives in
`src/` and WAS unit-tested there with numpy/scipy. If something in the
pydeck rendering below needs a small tweak, the underlying results are
still correct - see `demo.py` for a pure matplotlib version that is
guaranteed to run, as a fallback / cross-check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

from terrain import generate_synthetic_dem, load_real_dem
from rf_shadow import compute_los_grid
from mission import run_adaptive_mission


st.set_page_config(page_title="EW-ShadowNavigator", layout="wide")


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------

# Geographic anchor for the grid (Kyiv region). deck.gl layers work in
# lng/lat by default, so we map the meter-based grid onto a small patch of
# real coordinates around this anchor. This is the robust, well-supported
# path (unlike a cartesian OrbitView, which needs per-layer coordinate-system
# overrides and renders blank if you forget them).
ANCHOR_LAT = 50.4501
ANCHOR_LNG = 30.5234
_M_PER_DEG_LAT = 111_320.0


def _m_per_deg_lng(lat_deg: float) -> float:
    return 111_320.0 * np.cos(np.radians(lat_deg))


def grid_to_lnglat(rc: tuple[int, int], elevation: np.ndarray, cellsize: float):
    """Map a grid (row, col) to [lng, lat], centered on the anchor."""
    rows, cols = elevation.shape
    r, c = int(rc[0]), int(rc[1])
    east_m = (c - cols / 2) * cellsize
    north_m = (rows / 2 - r) * cellsize
    lat = ANCHOR_LAT + north_m / _M_PER_DEG_LAT
    lng = ANCHOR_LNG + east_m / _m_per_deg_lng(ANCHOR_LAT)
    return float(lng), float(lat)


def elevation_to_rgb(norm: np.ndarray) -> np.ndarray:
    """Map normalized elevation [0,1] to an RGB terrain-like gradient."""
    stops = np.array([
        [30, 90, 40],     # low ground - green
        [150, 130, 60],   # mid - brown
        [235, 235, 225],  # high - light gray/white
    ])
    idx = norm * (len(stops) - 1)
    lo = np.clip(idx.astype(int), 0, len(stops) - 2)
    frac = (idx - lo)[:, None]
    rgb = stops[lo] * (1 - frac) + stops[lo + 1] * frac
    return rgb.astype(int)


def build_grid_dataframe(elevation: np.ndarray, cellsize: float, shadow_mask: np.ndarray | None):
    rows, cols = elevation.shape
    r, c = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")

    east_m = (c - cols / 2) * cellsize
    north_m = (rows / 2 - r) * cellsize
    lat = ANCHOR_LAT + north_m / _M_PER_DEG_LAT
    lng = ANCHOR_LNG + east_m / _m_per_deg_lng(ANCHOR_LAT)

    norm = (elevation - elevation.min()) / max(elevation.max() - elevation.min(), 1e-6)
    elev_color = elevation_to_rgb(norm.ravel())

    df = pd.DataFrame({
        "lng": [float(v) for v in lng.ravel()],
        "lat": [float(v) for v in lat.ravel()],
        "elevation": [float(v) for v in elevation.ravel()],
    })
    # pydeck serializes to JSON, so color columns must be plain Python lists
    # of ints - NOT numpy arrays (which aren't JSON-serializable and cause
    # "vars() argument must have __dict__").
    df["color_elevation"] = [[int(v) for v in row] for row in elev_color]

    if shadow_mask is not None:
        flat = shadow_mask.ravel()
        df["color_shadow"] = [
            [60, 180, 75] if s else [210, 60, 60] for s in flat
        ]

    return df


def path_to_lnglatz(path, elevation, cellsize, ve, lift_m=12.0):
    """Path as [lng, lat, altitude_m], altitude scaled to match extruded terrain."""
    out = []
    for rc in path:
        lng, lat = grid_to_lnglat(rc, elevation, cellsize)
        z = float(elevation[int(rc[0]), int(rc[1])]) * ve + lift_m * ve
        out.append([lng, lat, z])
    return out


# ----------------------------------------------------------------------
# Sidebar - scenario controls
# ----------------------------------------------------------------------

st.sidebar.title("EW-ShadowNavigator")
st.sidebar.caption("Inversive UAV navigation via jammer-signal terrain masking")

st.sidebar.header("1. Terrain")
dem_source = st.sidebar.radio("DEM source", ["Synthetic (demo)", "Upload real GeoTIFF"])

if dem_source == "Synthetic (demo)":
    size = st.sidebar.slider("Grid size (cells)", 51, 151, 101, step=10)
    seed = st.sidebar.number_input("Random seed", 0, 9999, 21)
    roughness = st.sidebar.slider("Roughness", 0.3, 0.9, 0.62)
    height_scale = st.sidebar.slider("Relief (m)", 30, 300, 150)
    cellsize = st.sidebar.slider("Cell size (m)", 10, 50, 25)
else:
    uploaded = st.sidebar.file_uploader("GeoTIFF DEM (.tif)", type=["tif", "tiff"])

vertical_exaggeration = st.sidebar.slider(
    "Vertical exaggeration (visual only)", 1.0, 8.0, 3.0,
    help="Multiplies elevation for the 3D view only, so relief is easier to see. "
         "Does not affect any of the underlying physics/calculations.",
)

st.sidebar.header("2. Scenario")
if dem_source == "Synthetic (demo)":
    max_idx = size - 1
    default_jammer = (min(43, max_idx), min(44, max_idx))
else:
    max_idx = 100  # placeholder until a real DEM is loaded
    default_jammer = (43, 44)

jr = st.sidebar.slider("Jammer row", 0, max_idx, min(default_jammer[0], max_idx))
jc = st.sidebar.slider("Jammer col", 0, max_idx, min(default_jammer[1], max_idx))
jammer_height_agl = st.sidebar.slider("Jammer mast height AGL (m)", 2.0, 20.0, 8.0)
drone_height_agl = st.sidebar.slider("Drone flight height AGL (m)", 5.0, 60.0, 10.0)

sr = st.sidebar.slider("Start row", 0, max_idx, 5)
sc = st.sidebar.slider("Start col", 0, max_idx, 5)
gr = st.sidebar.slider("Goal row", 0, max_idx, min(95, max_idx))
gc = st.sidebar.slider("Goal col", 0, max_idx, min(95, max_idx))

st.sidebar.header("3. Mission parameters")
replan_every = st.sidebar.slider("Replan interval (steps)", 4, 30, 8)
exposure_penalty = st.sidebar.slider("Exposure penalty (risk weight)", 1.0, 15.0, 6.0)
mission_seed = st.sidebar.number_input("Mission RNG seed", 0, 9999, 1)

run_clicked = st.sidebar.button("Run mission", type="primary", use_container_width=True)


# ----------------------------------------------------------------------
# Main area
# ----------------------------------------------------------------------

st.title("EW-ShadowNavigator")
st.caption(
    "Turns the jammer's own emissions into a navigation beacon: the drone "
    "localizes the RF source from signal strength, then routes through "
    "terrain-masked \u2018radio shadow\u2019 zones to avoid it."
)

if dem_source == "Synthetic (demo)":
    dem = generate_synthetic_dem(size=size, seed=int(seed), roughness=roughness,
                                  height_scale=float(height_scale), cellsize=float(cellsize))
else:
    if uploaded is None:
        st.info("Upload a GeoTIFF DEM in the sidebar, or switch back to 'Synthetic (demo)' to try it instantly.")
        st.stop()
    with open("/tmp/_uploaded_dem.tif", "wb") as f:
        f.write(uploaded.read())
    try:
        dem = load_real_dem("/tmp/_uploaded_dem.tif")
    except ImportError as e:
        st.error(str(e))
        st.stop()

jammer_rc = (int(jr), int(jc))
start = (int(sr), int(sc))
goal = (int(gr), int(gc))

if "mission_result" not in st.session_state:
    st.session_state.mission_result = None
    st.session_state.true_shadow = None

if run_clicked:
    with st.spinner("Computing radio-shadow map and running adaptive mission..."):
        true_shadow = compute_los_grid(
            dem.elevation, jammer_rc, jammer_height_agl, drone_height_agl, n_samples=150
        )
        result = run_adaptive_mission(
            elevation=dem.elevation, cellsize=dem.cellsize, start=start, goal=goal,
            jammer_true_rc=jammer_rc, jammer_height_agl=jammer_height_agl,
            drone_height_agl=drone_height_agl, replan_every=int(replan_every),
            exposure_penalty=float(exposure_penalty), seed=int(mission_seed),
        )
        st.session_state.true_shadow = true_shadow
        st.session_state.mission_result = result

result = st.session_state.mission_result
true_shadow = st.session_state.true_shadow

# --- metrics row
if result is not None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reached goal", "Yes" if result.reached_goal else "No")
    c2.metric("Localization error", f"{result.localization_error_m:.0f} m")
    c3.metric("Localization confidence", f"{result.final_confidence*100:.0f}%")
    c4.metric("Path exposure", f"{result.exposure_fraction*100:.0f}%")
else:
    st.info("Configure a scenario in the sidebar, then click **Run mission**.")

# --- 3D view
color_mode = st.radio(
    "Terrain color", ["Elevation", "Radio shadow (ground truth)"], horizontal=True,
    disabled=true_shadow is None,
)

shadow_for_color = true_shadow.shadow if (true_shadow is not None and color_mode.startswith("Radio")) else None
grid_df = build_grid_dataframe(dem.elevation, dem.cellsize, shadow_for_color)
ve = float(vertical_exaggeration)
grid_df["elevation_scaled"] = grid_df["elevation"] * ve
color_col = "color_shadow" if shadow_for_color is not None else "color_elevation"

layers = [
    pdk.Layer(
        "GridCellLayer",
        data=grid_df,
        get_position=["lng", "lat"],
        get_elevation="elevation_scaled",
        elevation_scale=1,
        cell_size=float(dem.cellsize),
        extruded=True,
        get_fill_color=color_col,
        pickable=False,
        opacity=0.85,
    )
]

if result is not None:
    # PathLayer with 3D [lng, lat, altitude_m] points. Altitude uses the same
    # vertical exaggeration as the extruded terrain so the route floats above it.
    traj = path_to_lnglatz(result.trajectory, dem.elevation, dem.cellsize, ve, lift_m=14.0)
    path_df = pd.DataFrame({"path": [traj], "name": ["UAV route"]})
    layers.append(
        pdk.Layer(
            "PathLayer",
            data=path_df,
            get_path="path",
            get_color=[30, 120, 255],
            width_min_pixels=4,
            get_width=6,
        )
    )

    # Markers as vertical columns (clearly visible above the terrain surface).
    def marker_row(rc, label, color, base_lift, height):
        lng, lat = grid_to_lnglat(rc, dem.elevation, dem.cellsize)
        base = float(dem.elevation[int(rc[0]), int(rc[1])]) * ve + base_lift
        return {
            "lng": lng, "lat": lat, "base": base,
            "elev": height, "label": label, "color": color,
        }

    marker_df = pd.DataFrame([
        marker_row(jammer_rc, "Jammer (true)", [220, 30, 30], 0.0, 220.0),
        marker_row(start, "Start", [20, 20, 20], 0.0, 140.0),
        marker_row(goal, "Goal", [20, 20, 20], 0.0, 140.0),
        marker_row(result.final_estimate, "Jammer estimate", [0, 200, 210], 0.0, 180.0),
    ])
    layers.append(
        pdk.Layer(
            "ColumnLayer",
            data=marker_df,
            get_position=["lng", "lat"],
            get_elevation="elev",
            elevation_scale=1,
            radius=float(dem.cellsize) * 1.2,
            get_fill_color="color",
            pickable=True,
            auto_highlight=True,
        )
    )

extent = float(max(dem.extent_meters()))
# zoom that fits the whole patch on screen (Web Mercator meters-per-pixel math)
zoom = float(np.log2(40_075_016.0 * np.cos(np.radians(ANCHOR_LAT)) * 800.0 / (512.0 * max(extent, 1.0))))
view_state = pdk.ViewState(
    latitude=ANCHOR_LAT,
    longitude=ANCHOR_LNG,
    zoom=zoom,
    pitch=50.0,
    bearing=20.0,
)

deck = pdk.Deck(
    layers=layers,
    initial_view_state=view_state,
    map_provider=None,
    map_style=None,
    tooltip={"text": "{label}"} if result is not None else None,
)
st.pydeck_chart(deck, use_container_width=True)
st.caption(
    "Drag to rotate/pan, scroll to zoom. Green/red = radio shadow vs. exposed "
    "(toggle above). Blue line = the drone's actually-flown adaptive route. "
    "Red column = true jammer, cyan column = the drone's final estimate."
)

# --- confidence chart + downloadable log
if result is not None:
    st.subheader("Localization confidence over the mission")
    conf_df = pd.DataFrame({
        "replan_step": range(len(result.steps)),
        "confidence": [s.confidence for s in result.steps],
    }).set_index("replan_step")
    st.line_chart(conf_df)

    log_df = pd.DataFrame([
        {
            "step": i,
            "drone_row": s.drone_pos[0], "drone_col": s.drone_pos[1],
            "measured_rssi_dbm": round(s.measured_rssi, 2),
            "jammer_estimate_row": s.jammer_estimate[0], "jammer_estimate_col": s.jammer_estimate[1],
            "confidence": round(s.confidence, 4),
        }
        for i, s in enumerate(result.steps)
    ])
    st.download_button(
        "Download mission log (CSV)",
        log_df.to_csv(index=False).encode("utf-8"),
        file_name="ews_shadownavigator_mission_log.csv",
        mime="text/csv",
    )

with st.expander("How this works"):
    st.markdown(
        """
**1. The paradox this exploits:** to jam GPS/radio, a source has to transmit a lot
of power - it "lights up" the RF spectrum. That makes it findable.

**2. Localization:** the drone takes noisy RSSI (signal strength) readings from
different positions as it moves. A grid-based Bayesian filter combines these
readings with a log-distance path-loss model (and terrain line-of-sight checks)
to converge on the jammer's likely position - without ever needing GPS.

**3. Radio shadow:** given a jammer position estimate, a line-of-sight analysis
over the terrain (DEM) finds cells that the jammer's signal geometrically
cannot reach - hills and ridges block radio waves just like they block light.

**4. Adaptive routing:** an A* planner threads a route from start to goal that
prefers those shadow cells, and re-plans periodically as the localization
estimate sharpens - so the route gets safer the more the drone learns.

*(Roadmap: swap the periodic A* replanner for incremental D\\* Lite, and export
the decision core via ONNX Runtime for on-board deployment on Jetson-class
hardware - see README.md.)*
        """
    )
