import numpy as np
import open3d as o3d
import argparse, json, os, time, sys, math, threading, select
import shutil
from contextlib import contextmanager
from pathlib import Path


# =========================================================
# MANUAL
# =========================================================

MANUAL="""
============================================================
OUSTER ARCHITECTURAL ANALYSIS — FULL SLAM EDITION
============================================================

USAGE
-----

Single file
    python ouster_analysis.py --input room.pcd
    python ouster_analysis.py --input room.pcd --output ./out

Multiple files stitched
    python ouster_analysis.py --input r1.pcd --multi r2.pcd r3.pcd

Footprint + separate height bag
    python ouster_analysis.py --input footprint_session --height-bag height_session --height-bag-mode height-static
    python ouster_analysis.py --input /data/ouster_bags/footprint_session_1r++ --height-bag /data/ouster_bags/height_session_1r+ --height-bag-mode height-static

Live capture (guided single perimeter walk, 45 s max)
    python ouster_analysis.py --input live
    python ouster_analysis.py --input live --seconds 40

ROS 2 walking bag recorded by the ROS container
    python ouster_analysis.py --input /data/ouster_bags/os1_session

ROS bag (single static scan saved to bag)
    python ouster_analysis.py --input room.bag
    python ouster_analysis.py --input os1_session
    python ouster_analysis.py --input os1_session --bag-mode static

ROS bag (dedicated static height bag)
    python ouster_analysis.py --input height_session --bag-mode height-static
    python ouster_analysis.py --input height_session --bag-mode height-static --footprint-area 10.954

ROS bag (walking scan — split by position first)
    python ouster_analysis.py --input pos1.bag --multi pos2.bag pos3.bag
    python ouster_analysis.py --input pos1 --multi pos2 pos3

    IMPORTANT: A single ROS 2 bag directory from record.launch.xml is
    now treated as a walking scan by default. The script samples
    PointCloud2 frames from the bag, gravity-aligns them individually,
    and runs the same registration path used for live walking capture.

    Use --bag-mode static only for bags recorded from one fixed sensor
    position, where stacking all frames into one cloud is desired.

    Bags are expected to be mounted into the analysis container at
    /data/ouster_bags.

    Output files are written to ./out by default. Use --output to choose
    a different export directory.


OUTPUT
------
All generated files are written under the configured output directory
(default: ./out).

floorplan_<timestamp>.dxf
    2-D floor plan polygon in CAD format with edge-length labels and
    a small summary block for area, perimeter, and height.

roomcloud_<timestamp>.ply
    canonical 3-D export containing the most complete cleaned scene cloud
    with semantic coloring for walls, floor, ceiling, furniture, and
    structural diagnostics context.

report.json
report_latest.json
report_<timestamp>.json
    floor_area_m2     gross floor area (m²)
    floor_perimeter_m polygon perimeter (m)
    room_height_m     estimated room height (m)
    volume_m3         area × height  (m³)
    walls             number of fitted room sides in the final footprint
    direct_wall_faces number of fitted room sides supported by direct wall evidence
    wall_observations number of wall observations retained after recovery
    corners           number of polygon vertices
    ceiling_detected  true if a ceiling plane was found (height is exact)
    wall_source       source of the final wall geometry
    height_confidence confidence label for room_height_m
    measurement_warnings list of caveats for fallback-derived results
    scan_time_sec     total processing time

two-bag report additions
    footprint_input   source used for footprint area/perimeter
    height_input      optional separate source used for room height
    footprint_room_height_m original footprint-bag-only height estimate
    footprint_volume_m3 original footprint-bag-only volume estimate

height-static report
    room_height_m     estimated room height from a dedicated static height bag
    height_source     ceiling-slice or top-envelope heuristic source
    height_confidence confidence label for room_height_m
    height_diagnostics support counts near the top band and floor/ceiling slices


CAPTURE BEST PRACTICE
---------------------
Sensor placement
    Place sensor on a tripod at 1.0–1.5 m height and 0.8-1.2 m from walls.
    You can handhold the sensor, but a stable mount usually produces better results.
    Avoid placing it directly against a wall — the adjacent wall
    will have almost no coverage, reducing detected corner count.

Walk slowly
Pause at corners
Scan doorways
Keep sensor level
Overlap rooms

For one-lap room + furniture capture
    Keep the sensor mostly level and aimed diagonally inward so the scan keeps
    seeing three bands at once whenever possible: a thin floor strip, the
    furniture-height middle band, and the lower wall/ceiling transition.
    Height support from the same lap is opportunistic, not the primary target.
    Do not keep the sensor pitched upward for long wall segments, or low
    furniture and floor coverage will degrade. Use only brief upward glances
    where the floor still remains visible.

Never run or spin.


MINIMUM REQUIREMENTS
--------------------
At least 3 distinct walls visible in the scan.
Floor must be partially visible.
Use --multi for room-scale coverage.

ACCURACY
--------
Floor area
    Single scan:           ±2–5 %
    Multi-scan + ICP:      ±1–3 %
    Best case (slow walk): ±1 %

    Area is derived from the intersection-polygon corners
    (structurally computed, not raw point measurements).
    This is more accurate than a raw convex hull for
    rectangular rooms.

    NOTE: Concave rooms (L-shape, alcoves) will be
    overestimated because the polygon is convex.
    For those, capture multiple scans from different
    positions and use --multi.

Room height
    If ceiling plane detected (ceiling_detected=true):
        Exact plane-to-plane distance.  Error ±2–5 cm.
    If no ceiling detected (ceiling_detected=false):
        Mean of per-wall top-Z minus floor centroid Z.  Error ±5–12 cm.
        height_confidence will be medium or low, not high.
    Scan toward the ceiling to maximise detection rate.

Volume
    floor_area × room_height.
    With ceiling detected:    ±3–6 %.
    Without ceiling detected: ±5–12 %.

Linear dimensions (wall lengths)
    ±2–5 cm per wall (single scan).
    ±1–2 cm with multi-scan registration.

Ouster OS1 hardware range accuracy: ±1–3 cm.
This software does not improve on underlying sensor limits;
it redistributes and reduces accumulated drift via ICP and
pose-graph optimisation, but cannot exceed sensor precision.

TROUBLESHOOTING
---------------
No floor detected       → include floor in scan
No walls detected       → scan must cover ≥3 walls
Bad geometry            → move slower, add more scans
Warped map              → capture more overlap
Few corners detected    → room needs ≥3 visible wall intersections
Volume seems wrong      → scan higher (toward ceiling) for better height estimate
Furniture counted as wall
    Furniture faces are filtered by a distance-adaptive Z-span rule:
    a wall at distance d must span at least max(0.4, d×0.65) metres
    vertically (≈ the sensor FOV window at that range).
Height too low          → sensor's vertical FOV cannot see the ceiling
    from a desk — move to the room centre or walk the perimeter more slowly.
    During the perimeter lap, keep the sensor facing inward toward the room
    centre but mostly level so a thin floor strip stays visible. Only add a
    slight upward tilt briefly where the floor remains visible, typically near
    corners or more open wall segments. Do not chase height with continuous
    upward pitch; that usually damages the footprint and furniture evidence
    more than it helps the ceiling estimate.
    Guided live now relies on the perimeter lap alone, so ceiling evidence must
    be captured during that lap rather than in a separate height step.


============================================================
TECHNOLOGY STACK OVERVIEW
============================================================

This software achieves architectural-scale reconstruction
accuracy by combining multiple advanced computational
geometry and robotics techniques. Each library serves a
specific role in the pipeline.

CORE LIBRARIES USED
-------------------

Open3D
    Provides point cloud processing, filtering,
    registration (ICP), pose graph optimization,
    RANSAC plane segmentation, and geometric
    transformations. This is the core plane-
    extraction engine used by this software.

NumPy
    Handles high-performance numerical operations
    such as covariance analysis, eigen decomposition,
    and matrix math used for alignment and geometry.

Shapely
    Computes 2D convex-hull areas for each
    detected plane to filter small/noise planes.

ROSbags Reader
    Enables loading recorded ROS2 scans with full
    precision point cloud data.

Ouster SDK (optional live mode)
    Streams real-time LiDAR data directly from sensor.


ALGORITHMIC TECHNIQUES
----------------------

World Alignment
    Floor alignment is driven primarily by RANSAC on
    the lower-Z slice of the cloud; PCA is only a
    fallback when a clean floor patch is not found.
    In walking mode each kept frame is gravity-aligned
    before registration.

RANSAC Plane Segmentation
    Iterative random-sample-consensus segmentation
    detects planar surfaces in any orientation.
    Planes are extracted one at a time; each set of
    inliers is removed before the next iteration.
    Static scans use a full-cloud pass. Walking scans
    use stratified Z-bands and wall-direction recovery
    because a merged walking cloud otherwise overweights
    horizontal surfaces.

Plane Classification
    Each detected plane is classified as floor or
    wall based on the angle between its normal and
    the vertical (Z) axis:
        |n_z| > 0.85  →  horizontal (floor/ceiling)
        |n_z| < 0.3   →  vertical   (wall)

ICP Registration
    Iterative Closest Point aligns multiple scans.
    Typical alignment error ±1–3 cm per pair.

Pose Graph Optimization
    Global optimization distributes alignment error
    across all scans instead of accumulating drift.

Walking Registration
    Walking scans use planar ICP (XY translation +
    Z-axis rotation only) after per-frame gravity
    alignment. This prevents tilt drift from building
    up across the capture.

Loop Closure Detection
    When scanning multiple rooms, small alignment
    errors accumulate. Loop closure detects when
    you return to a previously scanned area and
    globally corrects all scan positions.

    This prevents:
        warped floorplans
        slanted walls
        stretched geometry

Plane Intersection Geometry
    Room corners are found by solving the analytical
    3-plane intersection (two walls + floor), which
    is more accurate than using raw measured points.

Statistical Outlier Filtering
    Removes sensor noise and floating artifacts
    before plane segmentation.


WHY THIS APPROACH IS ACCURATE
------------------------------

Naive scan processing:
    uses raw points → noisy → distorted

This system:
    extracts structure → solves geometry → outputs CAD

By reconstructing structural primitives instead of raw
points, the system achieves architectural reliability.


PROFESSIONAL COMPARISON
-----------------------

This pipeline follows the same fundamental principles
used internally by:

    indoor mapping robots
    digital twin capture systems
    scan-to-BIM software
    surveying LiDAR rigs


ACCURACY LIMITATIONS
--------------------

Accuracy depends mainly on:

    scan speed
    overlap between scans
    surface reflectivity
    environment clutter

Hardware precision is rarely the limiting factor.

END
============================================================
"""


# =========================================================
# SAFE CONFIG SYSTEM
# =========================================================

CONFIG = dict(

    # preprocessing
    voxel_size = 0.04,
    outlier_neighbors = 40,
    outlier_std = 1.2,

    # ICP
    icp_max_dist = 0.6,
    icp_downsample = True,

    # loop closure
    loop_fitness_threshold = 0.30,

    # plane extraction (RANSAC)
    random_seed       = 1337,    # deterministic Open3D RANSAC seed
    ransac_dist       = 0.025,   # 2.5 cm inlier threshold
    min_plane_points  = 100,     # minimum inliers for a valid plane
    max_planes        = 20,      # maximum planes to extract
    min_plane_area    = 0.3,     # m² minimum valid plane area
    max_range_m       = 3.5,     # m — clip cloud XY radius; removes outdoor
                                 #     returns through windows/doors that would
                                 #     bias wall-synthesis centroid estimates.
                                 #     Set to None to disable.

    # wall qualification
    # Real walls span floor-to-ceiling (≥ 0.8 m).  Furniture faces
    # (shelves, table sides, sofa backs) span << 1 m.  Any vertical
    # plane whose inliers cover less than this height is rejected as
    # furniture/clutter and not counted as a wall.
    min_wall_abs_z_span  = 0.4,   # m — hard floor on Z-span (any wall)
    min_wall_fov_fraction = 0.65,  # Z-span must be ≥ dist_to_wall × this factor
                                   # ≈ 2·tan(18°); catches all > ~36° FOV sensors

    # floor detection
    floor_percentile        = 2,
    wall_height_percentile  = 99,
    floor_offset            = 0.15,

    # classification thresholds
    # |normal_z| > floor_z_comp_threshold  →  FLOOR
    # |normal_z| < wall_z_comp_threshold   →  candidate WALL (then Z-span check)
    # in between                           →  dead-zone, rejected
    floor_z_comp_threshold = 0.85,   # cos(32°) — tolerates residual tilt after ICP merge
    wall_z_comp_threshold  = 0.25,   # cos(76°) — accepts slightly-tilted wall planes

    # logging
    log_level = "info"
)

OUTPUT_DIR = Path("./out")

LATEST_ARTIFACT_NAMES = (
    "topdown_preview_latest.png",
    "scene_debug_latest.json",
    "floorplan_latest.dxf",
    "roomcloud_latest.ply",
    "roomcloud_raw_latest.ply",
    "roomcloud_clean_latest.ply",
    "roomcloud_scene_latest.ply",
    "roomcloud_view_latest.ply",
    "roommesh_latest.ply",
)

LAST_WALKING_QUALITY = None
LAST_WALKING_FIT_INFO = None
LAST_WALKING_FRAME_DATA = None


def set_output_dir(path):
    global OUTPUT_DIR

    OUTPUT_DIR = Path(path).expanduser()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def output_path(name):
    return OUTPUT_DIR / name


def _capture_latest_artifact_backups():
    backups = {}
    for name in LATEST_ARTIFACT_NAMES:
        latest_path = output_path(name)
        if not latest_path.exists():
            continue
        backup_path = output_path(f".backup_{name}")
        try:
            shutil.copyfile(latest_path, backup_path)
            backups[latest_path] = backup_path
        except Exception:
            pass
    return backups


def _restore_latest_artifact_backups(backups):
    restored = 0
    for latest_path, backup_path in (backups or {}).items():
        if not backup_path.exists():
            continue
        try:
            shutil.copyfile(backup_path, latest_path)
            restored += 1
        except Exception:
            pass
    for backup_path in (backups or {}).values():
        try:
            backup_path.unlink(missing_ok=True)
        except Exception:
            pass
    return restored


def _cleanup_latest_artifact_backups(backups):
    for backup_path in (backups or {}).values():
        try:
            backup_path.unlink(missing_ok=True)
        except Exception:
            pass


def save_report_json(report):
    export_tag = report.pop("_export_tag", None)
    if not export_tag:
        export_tag = time.strftime("%Y%m%d_%H%M%S")

    report_path = output_path("report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    latest_path = output_path("report_latest.json")
    _latest_copy(report_path, latest_path)

    tagged_path = output_path(f"report_{export_tag}.json")
    with open(tagged_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    log(f"Saved {report_path}")
    log(f"Saved {tagged_path}")


def save_live_guided_rejection_debug(debug_info):
    export_tag = time.strftime("%Y%m%d_%H%M%S")
    payload = dict(debug_info or {})
    payload["debug_export_tag"] = export_tag

    latest_path = output_path("live_guided_rejection_latest.json")
    tagged_path = output_path(f"live_guided_rejection_{export_tag}.json")

    with open(tagged_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    shutil.copyfile(tagged_path, latest_path)

    log(f"Saved {tagged_path}")
    log(f"Saved {latest_path}")


# =========================================================
# LOGGER
# =========================================================

def log(msg,level="info", force=False):

    levels=["silent","info","debug"]

    if force or levels.index(CONFIG["log_level"]) >= levels.index(level):
        print(msg)


def operator_log(msg):
    log(msg, "info", force=True)


@contextmanager
def log_level_scope(level):
    prev = CONFIG.get("log_level", "info")
    CONFIG["log_level"] = level
    try:
        yield
    finally:
        CONFIG["log_level"] = prev


def _operator_ready_countdown(prompt, seconds):
    seconds = max(0, int(seconds))
    if seconds <= 0:
        return

    operator_log(f"[GUIDE] {prompt}")
    operator_log(f"[GUIDE] Press Enter when ready, or wait {seconds}s for automatic start.")

    deadline = time.time() + seconds
    announced = None
    while True:
        remaining = int(math.ceil(deadline - time.time()))
        if remaining <= 0:
            break

        if remaining != announced and remaining <= min(seconds, 5):
            announced = remaining
            operator_log(f"[GUIDE] Starting in {remaining}s.")

        try:
            r, _, _ = select.select([sys.stdin], [], [], 0.25)
            if r:
                sys.stdin.readline()
                operator_log("[GUIDE] Starting now.")
                return
        except Exception:
            time.sleep(0.25)

    operator_log("[GUIDE] Starting now.")


def _guided_height_retry_reason(height_report, footprint_height_hint=None):
    room_height = height_report.get("room_height_m")
    if room_height is None:
        return "height estimate is unavailable"

    diag = height_report.get("height_diagnostics", {}) or {}
    if diag.get("height_hint_applied") and not _guided_height_has_independent_support(height_report):
        return (
            "height still depends on the walking hint; the static height hold has not produced independent "
            "ceiling/top-band evidence yet"
        )

    floor_slice_points = int(diag.get("floor_slice_points", 0))
    top_band_5cm = int(diag.get("top_band_points_5cm", 0))
    ceiling_slice_points = int(diag.get("ceiling_slice_points", 0))
    height_confidence = height_report.get("height_confidence")
    height_source = height_report.get("height_source")
    align_method = height_report.get("align_method")

    if align_method != "ransac" and floor_slice_points < 80:
        return (
            f"floor alignment is too weak "
            f"(align={align_method}, floor slice: {floor_slice_points} pts, ceiling slice: {ceiling_slice_points} pts)"
        )

    if height_confidence == "low" and top_band_5cm < 10:
        return (
            f"top-band support is still too weak "
            f"(5 cm band: {top_band_5cm} pts, ceiling slice: {ceiling_slice_points} pts)"
        )

    if (
        footprint_height_hint is not None
        and float(room_height) + 0.10 < float(footprint_height_hint)
        and (
            height_source == "static_ceiling_slice"
            or align_method != "ransac"
            or top_band_5cm < 40
        )
    ):
        return (
            f"height {float(room_height):.3f} m from {height_source} is implausibly below the walking estimate "
            f"{float(footprint_height_hint):.3f} m"
        )

    return None


GUIDED_LIVE_HEIGHT_POINT_LIMIT = 180000
GUIDED_LIVE_HEIGHT_FRAME_LIMIT = 60
GUIDED_LIVE_HEIGHT_RECENT_FRAME_LIMIT = 72
GUIDED_LIVE_HEIGHT_WINDOW_SCANS = 36
GUIDED_LIVE_HEIGHT_WINDOW_STEP = 12


def _guided_height_has_independent_support(height_report):
    diag = height_report.get("height_diagnostics", {}) or {}
    align_method = height_report.get("align_method")
    align_tilt_deg = float(height_report.get("align_tilt_deg", 0.0) or 0.0)
    floor_slice_points = int(diag.get("floor_slice_points", 0))
    ceiling_slice_points = int(diag.get("ceiling_slice_points", 0))
    top_band_5cm = int(diag.get("top_band_points_5cm", 0))
    height_source = height_report.get("height_source") or ""
    height_hint_applied = bool(diag.get("height_hint_applied"))

    if height_report.get("ceiling_detected"):
        if align_method == "ransac":
            return floor_slice_points >= 120 and ceiling_slice_points >= 25
        if height_hint_applied:
            return (
                align_tilt_deg <= 8.0
                and floor_slice_points >= 180
                and ceiling_slice_points >= 15
                and top_band_5cm >= 30
            )
        return False

    if align_method == "ransac":
        if height_source not in {"static_top_envelope_p99_99", "static_top_envelope_p99_9"}:
            return False
        return floor_slice_points >= 180 and top_band_5cm >= 25

    if height_hint_applied:
        # For guided live height, the static hold can independently confirm a
        # footprint-derived hint even when PCA alignment is used, as long as
        # the hold itself shows strong floor support and a non-trivial upper band.
        return (
            align_tilt_deg <= 8.0
            and floor_slice_points >= 150
            and (
                (ceiling_slice_points >= 20 and top_band_5cm >= 25)
                or (ceiling_slice_points >= 15 and top_band_5cm >= 30)
            )
        )

    if height_source not in {"static_top_envelope_p99_99", "static_top_envelope_p99_9"}:
        return False

    return (
        align_tilt_deg <= 8.0
        and floor_slice_points >= 180
        and ceiling_slice_points >= 12
        and top_band_5cm >= 30
    )


def _fuse_guided_height_with_footprint_hint(height_report, footprint_height_hint=None,
                                            hint_source=None):
    if height_report is None or footprint_height_hint is None:
        return height_report

    room_height = height_report.get("room_height_m")
    if room_height is None or height_report.get("ceiling_detected"):
        return height_report

    diag = dict(height_report.get("height_diagnostics", {}) or {})
    floor_slice_points = int(diag.get("floor_slice_points", 0))
    top_z_max = float(diag.get("top_z_max", 0.0) or 0.0)
    align_method = height_report.get("align_method")
    hint_height = float(footprint_height_hint)

    if hint_height <= float(room_height) + 0.05:
        return height_report

    if hint_height < 1.8 or hint_height > 6.0:
        return height_report

    if align_method != "ransac" and floor_slice_points < 120:
        return height_report

    if top_z_max > 0.0 and hint_height > top_z_max + 1.0:
        return height_report

    fused = dict(height_report)
    fused["room_height_m"] = round(hint_height, 3)
    fused["height_source"] = hint_source or "guided_footprint_ceiling_hint"
    fused["height_confidence"] = "medium"
    diag["footprint_height_hint_m"] = round(hint_height, 3)
    diag["height_hint_applied"] = True
    fused["height_diagnostics"] = diag

    warning = (
        "No ceiling plane was detected in the static height hold; height was fused "
        "from the dedicated static floor alignment and the footprint ceiling hint."
    )
    warnings = list(fused.get("measurement_warnings") or [])
    if warning not in warnings:
        warnings.append(warning)
    fused["measurement_warnings"] = warnings
    return fused


def _extract_guided_footprint_height_hint(footprint_clouds, footprint_report):
    hint_height = None
    hint_source = None

    def _estimate_supported_span_hint(clouds):
        spans = []
        supported_frames = 0
        for pcd in clouds or []:
            if pcd is None or len(pcd.points) < 120:
                continue
            if not pcd.has_normals():
                pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=30))

            pts = np.asarray(pcd.points)
            nrms = np.asarray(pcd.normals)
            if len(pts) == 0 or len(nrms) != len(pts):
                continue

            floor_z = float(np.percentile(pts[:, 2], 1))
            z_cut = float(np.percentile(pts[:, 2], 80))
            top_mask = (pts[:, 2] >= z_cut) & (np.abs(nrms[:, 2]) >= 0.55)
            if int(top_mask.sum()) < 20:
                continue

            top_z = float(np.percentile(pts[top_mask, 2], 98))
            span = top_z - floor_z
            if span >= 1.8:
                spans.append(span)
                supported_frames += 1

        if len(spans) < 3:
            return None, supported_frames

        return float(np.percentile(np.array(spans), 90)), supported_frames

    try:
        ceiling_hints = _extract_walking_ceiling_hints(footprint_clouds)
    except Exception:
        ceiling_hints = []

    if ceiling_hints:
        hint_height = max(float(get_plane_centroid(plane)[2]) for plane in ceiling_hints)
        hint_source = "guided_footprint_ceiling_patch"

    try:
        z_hint = _estimate_walking_ceiling_z(footprint_clouds)
    except Exception:
        z_hint = None

    if z_hint is not None and (hint_height is None or float(z_hint) > hint_height + 0.03):
        hint_height = float(z_hint)
        hint_source = "guided_footprint_ceiling_z_hint"

    supported_span_hint, supported_span_frames = _estimate_supported_span_hint(footprint_clouds)
    if (
        supported_span_hint is not None
        and supported_span_frames >= 3
        and (hint_height is None or float(supported_span_hint) > hint_height + 0.03)
    ):
        hint_height = float(supported_span_hint)
        hint_source = "guided_footprint_supported_span_hint"

    span_hints = []
    for pcd in footprint_clouds or []:
        if pcd is None or len(pcd.points) < 100:
            continue
        pts = np.asarray(pcd.points)
        floor_z = float(np.percentile(pts[:, 2], 1))
        top_z = float(np.percentile(pts[:, 2], 99.9))
        span = top_z - floor_z
        if span >= 1.8:
            span_hints.append(span)

    if len(span_hints) >= 3:
        span_hint = float(np.percentile(np.array(span_hints), 95))
        if hint_height is None or span_hint > hint_height + 0.03:
            hint_height = span_hint
            hint_source = "guided_footprint_span_hint"

    report_height = footprint_report.get("room_height_m")
    report_confidence = footprint_report.get("height_confidence")
    report_source = footprint_report.get("height_source")
    report_direct_wall_faces = int(footprint_report.get("direct_wall_faces") or 0)
    report_wall_observations = int(footprint_report.get("wall_observations") or 0)
    if (
        report_height is not None
        and float(report_height) >= 1.8
        and (
            report_confidence in {"high", "medium"}
            or (
                float(report_height) > float(hint_height or 0.0)
                and report_direct_wall_faces >= 4
                and report_wall_observations >= 8
            )
            or hint_height is None
            or float(report_height) > hint_height + 0.05
        )
    ):
        hint_height = max(float(report_height), float(hint_height or 0.0))
        hint_source = report_source or "guided_footprint_report"

    return hint_height, hint_source


def _build_guided_height_analysis_cloud(scans, recent_only=True):
    if not scans:
        return _build_live_point_cloud(scans, point_limit=GUIDED_LIVE_HEIGHT_POINT_LIMIT)

    if recent_only:
        selected_scans = scans[-GUIDED_LIVE_HEIGHT_RECENT_FRAME_LIMIT:]
    else:
        selected_scans = scans

    if len(selected_scans) > GUIDED_LIVE_HEIGHT_FRAME_LIMIT:
        step = max(1, int(math.ceil(len(selected_scans) / GUIDED_LIVE_HEIGHT_FRAME_LIMIT)))
        selected_scans = selected_scans[::step]

    return _build_aligned_live_point_cloud(
        selected_scans,
        point_limit=GUIDED_LIVE_HEIGHT_POINT_LIMIT,
        max_range_m=CONFIG.get("max_range_m", 3.5),
    )


def _guided_height_cloud_variants(scans, recent_only=True):
    if not scans:
        return []

    if recent_only:
        selected_scans = scans[-GUIDED_LIVE_HEIGHT_RECENT_FRAME_LIMIT:]
    else:
        selected_scans = scans

    if len(selected_scans) > GUIDED_LIVE_HEIGHT_FRAME_LIMIT:
        step = max(1, int(math.ceil(len(selected_scans) / GUIDED_LIVE_HEIGHT_FRAME_LIMIT)))
        selected_scans = selected_scans[::step]

    variants = []
    try:
        variants.append((
            _build_aligned_live_point_cloud(
                selected_scans,
                point_limit=GUIDED_LIVE_HEIGHT_POINT_LIMIT,
                max_range_m=CONFIG.get("max_range_m", 3.5),
            ),
            "aligned",
        ))
    except Exception:
        pass

    try:
        variants.append((
            _build_live_point_cloud(
                selected_scans,
                point_limit=GUIDED_LIVE_HEIGHT_POINT_LIMIT,
            ),
            "raw",
        ))
    except Exception:
        pass

    return variants


def _select_guided_height_cloud_variant(scans, footprint_area=None,
                                        footprint_height_hint=None,
                                        footprint_height_hint_source=None,
                                        source_label="live_height_preview",
                                        recent_only=True):
    best_cloud = None
    best_report = None
    best_score = None

    for candidate_cloud, variant_name in _guided_height_cloud_variants(scans, recent_only=recent_only):
        try:
            candidate_report = analyze_static_height_cloud(
                candidate_cloud,
                footprint_area=footprint_area,
                source_label=source_label,
                allow_relaxed_floor=True,
            )
            candidate_report = _fuse_guided_height_with_footprint_hint(
                candidate_report,
                footprint_height_hint=footprint_height_hint,
                hint_source=footprint_height_hint_source,
            )
            candidate_report["height_cloud_variant"] = variant_name
        except Exception:
            continue

        candidate_score = _guided_height_capture_score(
            candidate_report,
            footprint_height_hint=footprint_height_hint,
        )
        if best_score is None or candidate_score > best_score:
            best_score = candidate_score
            best_cloud = candidate_cloud
            best_report = candidate_report

    return best_cloud, best_report, best_score


def _guided_height_capture_score(height_report, footprint_height_hint=None):
    diag = height_report.get("height_diagnostics", {}) or {}
    room_height = height_report.get("room_height_m")
    if room_height is None:
        room_height = 0.0

    ceiling_detected = bool(height_report.get("ceiling_detected"))
    independent_support = _guided_height_has_independent_support(height_report)
    align_method = height_report.get("align_method")
    height_source = height_report.get("height_source") or ""
    floor_slice_points = int(diag.get("floor_slice_points", 0))
    ceiling_slice_points = int(diag.get("ceiling_slice_points", 0))
    top_band_5cm = int(diag.get("top_band_points_5cm", 0))
    top_z_max = float(diag.get("top_z_max", 0.0) or 0.0)

    plausible_vs_hint = 0
    if footprint_height_hint is None:
        plausible_vs_hint = 1
    else:
        if float(room_height) + 0.05 >= float(footprint_height_hint):
            plausible_vs_hint = 2
        elif float(room_height) + 0.20 >= float(footprint_height_hint):
            plausible_vs_hint = 1
    if diag.get("height_hint_applied") and not independent_support:
        plausible_vs_hint = 0

    return (
        int(ceiling_detected),
        int(independent_support),
        int(plausible_vs_hint),
        int(align_method == "ransac"),
        int(height_source == "static_top_envelope_max"),
        round(top_z_max, 3),
        int(ceiling_slice_points),
        int(floor_slice_points),
        int(top_band_5cm),
        round(float(room_height), 3),
    )


def _guided_height_scan_windows(scans):
    if not scans:
        return []

    recent_scans = scans[-GUIDED_LIVE_HEIGHT_RECENT_FRAME_LIMIT:]
    window = min(GUIDED_LIVE_HEIGHT_WINDOW_SCANS, len(recent_scans))
    if window <= 0:
        return []

    if len(recent_scans) <= window:
        return [recent_scans]

    windows = [recent_scans]
    for start in range(0, len(recent_scans) - window + 1, GUIDED_LIVE_HEIGHT_WINDOW_STEP):
        windows.append(recent_scans[start:start + window])
    tail = recent_scans[-window:]
    if windows[-1] is not tail:
        windows.append(tail)
    return windows


def _select_best_guided_height_candidate(scans, footprint_area=None,
                                         footprint_height_hint=None,
                                         footprint_height_hint_source=None,
                                         source_label="live_height_preview"):
    best_cloud = None
    best_report = None
    best_score = None

    for window_scans in _guided_height_scan_windows(scans):
        candidate_cloud, candidate_report, candidate_score = _select_guided_height_cloud_variant(
            window_scans,
            footprint_area=footprint_area,
            footprint_height_hint=footprint_height_hint,
            footprint_height_hint_source=footprint_height_hint_source,
            source_label=source_label,
            recent_only=False,
        )
        if candidate_cloud is None or candidate_report is None:
            continue
        if best_score is None or candidate_score > best_score:
            best_score = candidate_score
            best_cloud = candidate_cloud
            best_report = candidate_report

    return best_cloud, best_report, best_score


def _format_guided_height_status(height_report):
    diag = height_report.get("height_diagnostics", {}) or {}
    room_height = height_report.get("room_height_m")
    height_source = height_report.get("height_source") or "unknown"
    height_confidence = height_report.get("height_confidence") or "unknown"
    ceiling_detected = "yes" if height_report.get("ceiling_detected") else "no"
    align_method = height_report.get("align_method") or "unknown"
    floor_slice_points = int(diag.get("floor_slice_points", 0))
    ceiling_slice_points = int(diag.get("ceiling_slice_points", 0))
    top_band_5cm = int(diag.get("top_band_points_5cm", 0))
    top_z_p9999 = diag.get("top_z_p99_99")
    top_z_max = diag.get("top_z_max")

    parts = [
        f"{room_height} m",
        f"via {height_source}",
        f"confidence {height_confidence}",
        f"ceiling {ceiling_detected}",
        f"align {align_method}",
        f"floor slice {floor_slice_points}",
        f"ceiling slice {ceiling_slice_points}",
        f"top support {top_band_5cm}",
    ]
    if top_z_p9999 is not None:
        parts.append(f"z99.99 {float(top_z_p9999):.3f}")
    if top_z_max is not None:
        parts.append(f"zmax {float(top_z_max):.3f}")
    return ", ".join(parts) + "."


def _guided_height_operator_adjustment(height_report, footprint_height_hint=None):
    diag = height_report.get("height_diagnostics", {}) or {}
    floor_slice_points = int(diag.get("floor_slice_points", 0))
    ceiling_slice_points = int(diag.get("ceiling_slice_points", 0))
    top_band_5cm = int(diag.get("top_band_points_5cm", 0))
    align_method = height_report.get("align_method")
    room_height = height_report.get("room_height_m")

    if align_method != "ransac" or floor_slice_points < 80:
        return (
            "[GUIDE] Floor alignment is weak. Keep the sensor upright but tilt slightly less upward so more floor stays visible at the bottom of the scan before trying again."
        )

    if ceiling_slice_points < 25 and top_band_5cm >= 20:
        return (
            "[GUIDE] The scan sees a stable upper band, but not a usable ceiling slice yet. Stay near the room centre and tilt a bit more upward without losing the floor edge."
        )

    if (
        footprint_height_hint is not None
        and room_height is not None
        and float(room_height) + 0.08 < float(footprint_height_hint)
    ):
        return (
            "[GUIDE] Height is still below the walking estimate. Re-center the sensor, keep it level, and raise the aim only slightly while preserving a thin floor strip."
        )

    if top_band_5cm < 10:
        return (
            "[GUIDE] Top support is still weak. Return slightly toward the centre, tilt a bit more upward, and keep the floor edge visible."
        )

    return None


def _guided_height_fallback_accept_reason(height_report, footprint_height_hint=None):
    room_height = height_report.get("room_height_m")
    if room_height is None:
        return None

    diag = height_report.get("height_diagnostics", {}) or {}
    height_source = height_report.get("height_source")
    top_band_5cm = int(diag.get("top_band_points_5cm", 0))
    floor_slice_points = int(diag.get("floor_slice_points", 0))
    ceiling_slice_points = int(diag.get("ceiling_slice_points", 0))
    top_z_max = diag.get("top_z_max")
    height_hint_applied = bool(diag.get("height_hint_applied"))
    align_tilt_deg = float(height_report.get("align_tilt_deg", 0.0) or 0.0)

    if height_hint_applied and footprint_height_hint is not None:
        if (
            float(room_height) + 0.02 >= float(footprint_height_hint)
            and floor_slice_points >= 250
            and ceiling_slice_points >= 3
            and top_band_5cm >= 30
            and align_tilt_deg <= 8.0
        ):
            return (
                f"accepting stable guided height {float(room_height):.3f} m from the footprint hint "
                f"after strong static floor confirmation (floor slice {floor_slice_points}, "
                f"ceiling slice {ceiling_slice_points}, top support {top_band_5cm}, tilt {align_tilt_deg:.1f}°)"
            )

    if height_source not in {
        "static_top_envelope_max",
        "static_top_envelope_p99_99",
        "static_top_envelope_p99_9",
    }:
        return None

    if top_band_5cm < 6 or floor_slice_points < 180:
        return None

    if top_z_max is None or float(top_z_max) < 2.0:
        return None

    if (
        footprint_height_hint is not None
        and float(room_height) + 0.05 < float(footprint_height_hint)
    ):
        return None

    return (
        f"accepting sparse but plausible top-envelope height {float(room_height):.3f} m "
        f"({height_source}, floor slice {floor_slice_points}, top support {top_band_5cm})"
    )


def _format_guided_footprint_status(report):
    parts = [
        f"area {report.get('floor_area_m2')} m²",
        f"perimeter {report.get('floor_perimeter_m')} m",
    ]
    footprint_confidence = report.get("footprint_confidence")
    if footprint_confidence:
        parts.append(f"footprint confidence {footprint_confidence}")
    direct_faces = report.get("direct_wall_faces")
    if direct_faces is not None:
        parts.append(f"direct faces {int(direct_faces)}/4")
    wall_observations = report.get("wall_observations")
    if wall_observations is not None:
        parts.append(f"wall observations {int(wall_observations)}")
    wall_source = report.get("wall_source")
    if wall_source:
        parts.append(f"source {wall_source}")
    return ", ".join(parts) + "."


def _guided_footprint_has_ignorable_narrow_gap_ambiguity(report):
    if int(report.get("scene_opening_candidates") or 0) > 0:
        return False

    max_status = str(report.get("scene_max_evidence_edge_status") or "")
    if max_status != "rejected":
        return False

    rejections = {
        str(item)
        for item in (report.get("scene_max_evidence_edge_rejections") or [])
        if item is not None
    }
    if "too_narrow" not in rejections:
        return False
    if rejections - {"too_narrow", "weak_left_jamb", "weak_right_jamb", "weak_jamb"}:
        return False

    gap_width = float(report.get("scene_max_evidence_edge_gap_width") or 0.0)
    continuity = float(report.get("scene_max_evidence_edge_continuity") or 0.0)
    max_opening_evidence = float(report.get("scene_max_opening_evidence_score") or 0.0)
    if gap_width <= 0.0 or gap_width > 0.35:
        return False
    if continuity < 0.9:
        return False
    if max_opening_evidence < 0.35:
        return False

    return True


def _guided_footprint_effective_edge_metrics(report):
    max_opening_evidence = float(report.get("scene_max_opening_evidence_score") or 0.0)
    min_edge_continuity = float(report.get("scene_min_edge_continuity") or 0.0)
    min_edge_low_fill = float(report.get("scene_min_edge_low_fill_ratio") or 0.0)
    min_edge_top_fill = float(report.get("scene_min_edge_top_fill_ratio") or 0.0)

    if _guided_footprint_has_ignorable_narrow_gap_ambiguity(report):
        max_opening_evidence = min(max_opening_evidence, 0.14)
        min_edge_low_fill = max(min_edge_low_fill, 0.55)
        min_edge_top_fill = max(min_edge_top_fill, 0.35)

    return (
        max_opening_evidence,
        min_edge_continuity,
        min_edge_low_fill,
        min_edge_top_fill,
    )


def _guided_footprint_floor_support_ratio(report):
    scene_input_points = int(report.get("scene_input_points") or 0)
    if scene_input_points <= 0:
        return 0.0
    return float(report.get("scene_floor_points") or 0.0) / float(scene_input_points)


def _guided_footprint_capture_score(report):
    confidence_issues = list(report.get("confidence_issues") or [])
    max_direct_tail = float(report.get("walking_fit_max_direct_tail") or 0.0)
    floor_support_ratio = _guided_footprint_floor_support_ratio(report)
    floor_points = int(report.get("scene_floor_points") or 0)
    preview_worth_saving = _guided_footprint_preview_is_worth_saving(report)
    preview_usable = _guided_footprint_preview_is_usable(report)
    (
        max_opening_evidence,
        min_edge_continuity,
        min_edge_low_fill,
        min_edge_top_fill,
    ) = _guided_footprint_effective_edge_metrics(report)
    return (
        int(not bool(confidence_issues)),
        int(preview_worth_saving),
        int(preview_usable),
        round(floor_support_ratio, 4),
        int(floor_points),
        round(min_edge_top_fill, 3),
        round(min_edge_low_fill, 3),
        round(min_edge_continuity, 3),
        int(not bool(report.get("rectangular_fallback"))),
        -round(max_opening_evidence, 3),
        -round(max_direct_tail, 3),
        int(report.get("direct_wall_faces") or 0),
        int(report.get("wall_observations") or 0),
    )


def _guided_footprint_rescue_is_collapsed(candidate_report, baseline_report):
    if not candidate_report or not baseline_report:
        return False

    candidate_area = float(candidate_report.get("floor_area_m2") or 0.0)
    baseline_area = float(baseline_report.get("floor_area_m2") or 0.0)
    candidate_perimeter = float(candidate_report.get("floor_perimeter_m") or 0.0)
    baseline_perimeter = float(baseline_report.get("floor_perimeter_m") or 0.0)
    if candidate_area <= 0.0 or baseline_area <= 0.0 or candidate_perimeter <= 0.0 or baseline_perimeter <= 0.0:
        return False

    area_ratio = candidate_area / baseline_area
    perimeter_ratio = candidate_perimeter / baseline_perimeter
    if area_ratio >= 0.65 or perimeter_ratio >= 0.80:
        return False

    candidate_direct_faces = int(candidate_report.get("direct_wall_faces") or 0)
    baseline_direct_faces = int(baseline_report.get("direct_wall_faces") or 0)
    candidate_rect = bool(candidate_report.get("rectangular_fallback"))
    baseline_rect = bool(baseline_report.get("rectangular_fallback"))

    return (
        candidate_rect
        or candidate_direct_faces < baseline_direct_faces
        or not baseline_rect
    )


def _guided_footprint_should_replace_preview_candidate(current_report, current_frames_raw,
                                                       best_report, best_frames_raw):
    if best_report is None or best_frames_raw is None:
        return True

    current_score = _guided_footprint_capture_score(current_report)
    best_score = _guided_footprint_capture_score(best_report)

    # Preserve a shorter provisional preview unless the later candidate
    # materially improves the geometry, not just the trailing observation count.
    if current_score[:-1] > best_score[:-1]:
        return True
    if current_score[:-1] < best_score[:-1]:
        return False

    if len(current_frames_raw) < len(best_frames_raw):
        return True

    return len(current_frames_raw) <= len(best_frames_raw) + 2 and current_score[-1] > best_score[-1]


def _guided_footprint_fallback_accept_reason(report):
    confidence_issues = list(report.get("confidence_issues") or [])
    if not confidence_issues:
        return None

    direct_wall_faces = int(report.get("direct_wall_faces") or 0)
    wall_observations = int(report.get("wall_observations") or 0)
    wall_source = report.get("wall_source") or ""

    if direct_wall_faces < 3 or wall_observations < 3:
        return None

    if wall_source not in {"frame_hints_recovery", "global_plus_frame_hints"}:
        return None

    (
        max_opening_evidence,
        min_edge_continuity,
        min_edge_low_fill,
        min_edge_top_fill,
    ) = _guided_footprint_effective_edge_metrics(report)
    max_direct_tail = float(report.get("walking_fit_max_direct_tail") or 0.0)
    floor_support_ratio = _guided_footprint_floor_support_ratio(report)
    floor_points = int(report.get("scene_floor_points") or 0)
    if float(min_edge_continuity) < 0.9:
        return None
    if float(min_edge_low_fill) < 0.5:
        return None
    if float(min_edge_top_fill) < 0.18:
        return None
    if max_opening_evidence >= 0.15:
        return None
    # ICP drift in a walking scan naturally spreads wall support by 0.20-0.35 m;
    # the absolute values and edge fills are the primary geometry signal.
    if max_direct_tail > 0.35:
        return None
    if floor_support_ratio < 0.004 or floor_points < 800:
        return None

    allowed_issue_prefixes = (
        "global wall extraction found no walls and walking recovery could not directly support all four room faces",
        "footprint needed a synthesised wall face (3/4 direct faces)",
        "walking recovery remained too sparse after merging",
    )
    if any(not any(issue.startswith(prefix) for prefix in allowed_issue_prefixes) for issue in confidence_issues):
        return None

    return (
        f"accepting best recovered footprint candidate with {direct_wall_faces}/4 direct faces, "
        f"{wall_observations} wall observations, source {wall_source}, "
        f"min continuity {float(min_edge_continuity):.3f}, min low fill {float(min_edge_low_fill):.3f}, "
        f"min top fill {float(min_edge_top_fill):.3f}, max opening evidence {max_opening_evidence:.3f}, "
        f"direct tail {max_direct_tail:.3f}"
    )


def _estimate_footprint_confidence(report):
    confidence_issues = list(report.get("confidence_issues") or [])
    direct_wall_faces = int(report.get("direct_wall_faces") or 0)
    wall_observations = int(report.get("wall_observations") or 0)
    if direct_wall_faces < 3 or wall_observations < 3:
        return "low"

    floor_support_ratio = _guided_footprint_floor_support_ratio(report)
    floor_points = int(report.get("scene_floor_points") or 0)
    if confidence_issues or floor_support_ratio < 0.004 or floor_points < 800:
        return "low"

    if bool(report.get("rectangular_fallback")):
        return "low"

    max_direct_tail = float(report.get("walking_fit_max_direct_tail") or 0.0)
    (
        max_opening_evidence,
        min_edge_continuity,
        min_edge_low_fill,
        min_edge_top_fill,
    ) = _guided_footprint_effective_edge_metrics(report)

    if (
        max_opening_evidence >= 0.35
        or min_edge_continuity < 0.75
        or min_edge_low_fill < 0.40
        or min_edge_top_fill < 0.25
        or max_direct_tail > 0.28
    ):
        return "low"

    if (
        max_opening_evidence < 0.10
        and min_edge_continuity >= 0.95
        and min_edge_low_fill >= 0.65
        and min_edge_top_fill >= 0.50
        and max_direct_tail <= 0.08
    ):
        return "high"

    return "medium"


def _guided_footprint_preview_is_usable(report):
    confidence_issues = list(report.get("confidence_issues") or [])
    if confidence_issues:
        return False

    direct_wall_faces = int(report.get("direct_wall_faces") or 0)
    if direct_wall_faces < 3:
        return False

    (
        max_opening_evidence,
        _,
        min_edge_low_fill,
        min_edge_top_fill,
    ) = _guided_footprint_effective_edge_metrics(report)
    floor_support_ratio = _guided_footprint_floor_support_ratio(report)
    if (
        max_opening_evidence >= 0.35
        and (min_edge_low_fill < 0.55 or min_edge_top_fill < 0.35)
    ):
        return False

    if floor_support_ratio < 0.004 and (min_edge_low_fill < 0.55 or min_edge_top_fill < 0.45):
        return False

    max_direct_tail = float(report.get("walking_fit_max_direct_tail") or 0.0)
    if direct_wall_faces >= 4 and max_direct_tail > 0.28:
        return False

    return True


def _guided_footprint_preview_is_worth_saving(report):
    if _guided_footprint_preview_is_usable(report):
        return True

    if _guided_footprint_fallback_accept_reason(report) is not None:
        return True

    direct_wall_faces = int(report.get("direct_wall_faces") or 0)
    wall_observations = int(report.get("wall_observations") or 0)
    if direct_wall_faces < 3 or wall_observations < 3:
        return False

    max_opening_evidence = float(report.get("scene_max_opening_evidence_score") or 0.0)
    min_edge_low_fill = float(report.get("scene_min_edge_low_fill_ratio") or 0.0)
    min_edge_top_fill = float(report.get("scene_min_edge_top_fill_ratio") or 0.0)
    max_direct_tail = float(report.get("walking_fit_max_direct_tail") or 0.0)
    floor_support_ratio = _guided_footprint_floor_support_ratio(report)

    if _guided_footprint_has_ignorable_narrow_gap_ambiguity(report):
        max_opening_evidence = min(max_opening_evidence, 0.14)
        min_edge_low_fill = max(min_edge_low_fill, 0.55)
        min_edge_top_fill = max(min_edge_top_fill, 0.35)

    if (
        max_opening_evidence >= 0.45
        and min_edge_low_fill < 0.35
        and min_edge_top_fill < 0.25
    ):
        return False

    if max_direct_tail > 0.35 and min_edge_low_fill < 0.40:
        return False

    if floor_support_ratio < 0.004 and min_edge_low_fill < 0.55:
        return False

    return True


def _guided_footprint_preview_can_autostop(report, *, elapsed, seconds):
    direct_wall_faces = int(report.get("direct_wall_faces") or 0)
    wall_observations = int(report.get("wall_observations") or 0)
    if _guided_footprint_preview_is_usable(report):
        return direct_wall_faces >= 4 and wall_observations >= 4

    fallback_reason = _guided_footprint_fallback_accept_reason(report)
    if fallback_reason is None:
        return False

    full_lap_elapsed = max(32.0, min(float(seconds) * 0.8, 50.0))
    return elapsed >= full_lap_elapsed


def _guided_footprint_preview_schedule(seconds, frame_interval_sec):
    expected_frames = max(1.0, float(seconds) / max(float(frame_interval_sec), 0.1))
    preview_min_frames = max(8, min(16, int(math.ceil(expected_frames * 0.35))))
    preview_min_elapsed = max(
        8.0,
        min(float(seconds) * 0.35, preview_min_frames * float(frame_interval_sec)),
    )
    preview_check_interval = 2.0
    return preview_min_frames, preview_min_elapsed, preview_check_interval


def _guided_footprint_continuation_hint(report):
    weak_edge_idx = report.get("scene_weak_edge_index")
    vertices = list(report.get("polygon_vertices_xy") or [])
    if weak_edge_idx is None or len(vertices) < 4:
        return (
            "If one wall looked less complete than the others, revisit it slowly. "
            "Otherwise, do one more short slow lap and pause again at each corner."
        )

    edge_idx = int(weak_edge_idx)
    corner_a = edge_idx + 1
    corner_b = ((edge_idx + 1) % len(vertices)) + 1
    metrics = []
    continuity = report.get("scene_weak_edge_continuity")
    low_fill = report.get("scene_weak_edge_low_fill_ratio")
    top_fill = report.get("scene_weak_edge_top_fill_ratio")
    opening = report.get("scene_weak_edge_opening_evidence_score")
    if (
        continuity is not None and float(continuity) >= 0.95
        and low_fill is not None and float(low_fill) >= 0.85
        and top_fill is not None and float(top_fill) >= 0.50
        and (opening is None or float(opening) < 0.15)
    ):
        return (
            "No single weak wall stood out in the current preview. "
            "If this result is still rejected, do one more short slow lap and pause again at each corner."
        )

    if continuity is not None:
        metrics.append(f"continuity {float(continuity):.2f}")
    if low_fill is not None:
        metrics.append(f"low fill {float(low_fill):.2f}")
    if top_fill is not None:
        metrics.append(f"top fill {float(top_fill):.2f}")
    if opening is not None and float(opening) >= 0.35:
        metrics.append(f"opening ambiguity {float(opening):.2f}")

    metric_text = ""
    if metrics:
        metric_text = " (" + ", ".join(metrics) + ")"

    return (
        f"The current preview is weakest along the wall between corners {corner_a} and {corner_b}{metric_text}. "
        "If that wall is not obvious in the room, do one more short slow lap and pause again at each corner."
    )


def _validate_guided_footprint_frames(frames_raw):
    with log_level_scope("silent"):
        footprint_clouds = _prepare_walking_frames(
            frames_raw,
            source_name="Guided live footprint preview",
        )
        report = analyze_loaded_clouds(
            footprint_clouds,
            input_path="live_guided_footprint_preview",
            walking_mode=True,
            allow_guided_live_borderline=True,
            return_low_confidence_report=True,
            suppress_exports=True,
        )
    return report


def _evaluate_guided_footprint_frames(frames_raw, *, input_path="live_guided_footprint_window"):
    with log_level_scope("silent"):
        footprint_clouds = _prepare_walking_frames(
            frames_raw,
            source_name="Guided live footprint window",
        )
        report = analyze_loaded_clouds(
            footprint_clouds,
            input_path=input_path,
            walking_mode=True,
            allow_guided_live_borderline=True,
            return_low_confidence_report=True,
            suppress_exports=True,
        )
    return report, footprint_clouds


def _guided_footprint_window_slices(frame_count, *, include_full_window=True):
    if frame_count <= 0:
        return []

    min_frames = max(10, int(math.ceil(frame_count * 0.65)))
    if frame_count <= 32:
        min_frames = min(min_frames, 16)
    min_frames = min(min_frames, frame_count)
    if frame_count <= min_frames + 2:
        return [(0, frame_count)]

    medium_span = max(min_frames, int(math.ceil(frame_count * 0.8)))
    ordered_windows = []
    seen = set()

    def add_window(start, end):
        start = max(0, int(start))
        end = min(frame_count, int(end))
        if end - start < min_frames:
            return
        window = (start, end)
        if window in seen:
            return
        seen.add(window)
        ordered_windows.append(window)

    if include_full_window:
        add_window(0, frame_count)

    # Short guided-live captures need interior recovery slices, but a fully
    # exhaustive sweep causes long end-of-lap stalls because each candidate
    # triggers a full reconstruction. Use a bounded prioritized set instead.
    if frame_count <= 32:
        candidate_spans = sorted({
            medium_span,
            max(min_frames, int(math.ceil((medium_span + min_frames) * 0.5))),
            min_frames,
        }, reverse=True)

        max_windows = 8
        for span in candidate_spans:
            max_start = frame_count - span
            if max_start <= 0:
                add_window(0, frame_count)
                continue

            if max_start <= 3:
                start_candidates = list(range(0, max_start + 1))
            else:
                start_candidates = [
                    0,
                    max_start,
                    max_start // 2,
                    int(math.ceil(max_start / 2.0)),
                ]

            for start in start_candidates:
                add_window(start, start + span)
                if len(ordered_windows) >= max_windows:
                    return ordered_windows
        return ordered_windows

    candidate_spans = sorted({
        frame_count,
        medium_span,
        max(min_frames, int(math.ceil((frame_count + medium_span) * 0.5))),
        min_frames,
    }, reverse=True)

    for span in candidate_spans:
        max_start = frame_count - span
        if max_start <= 0:
            add_window(0, frame_count)
            continue

        step = max(1, int(math.ceil(max_start / 4.0)))
        for start in range(0, max_start + 1, step):
            add_window(start, start + span)

        add_window(max_start // 2, (max_start // 2) + span)
        add_window(max_start, frame_count)

    return ordered_windows


def _select_guided_footprint_subwindow(frames_raw):
    best_candidate = None
    best_backup_candidate = None
    for start, end in _guided_footprint_window_slices(len(frames_raw), include_full_window=False):
        window_frames = frames_raw[start:end]
        if len(window_frames) < 3:
            continue

        try:
            report, footprint_clouds = _evaluate_guided_footprint_frames(
                window_frames,
                input_path=f"live_guided_footprint_window_{start}_{end}",
            )
        except RuntimeError:
            continue

        fallback_reason = _guided_footprint_fallback_accept_reason(report)
        confidence_issues = list(report.get("confidence_issues") or [])
        candidate_score = _guided_footprint_capture_score(report) + (end - start,)
        candidate = {
            "frames_raw": [frame.copy() for frame in window_frames],
            "report": dict(report),
            "footprint_clouds": [o3d.geometry.PointCloud(cloud) for cloud in footprint_clouds],
            "score": candidate_score,
            "start": start,
            "end": end,
        }

        if confidence_issues and fallback_reason is None:
            if not _guided_footprint_preview_is_worth_saving(report):
                continue
            if best_backup_candidate is None or candidate_score > best_backup_candidate["score"]:
                best_backup_candidate = candidate
            continue

        if best_candidate is None or candidate_score > best_candidate["score"]:
            best_candidate = candidate

        if not confidence_issues and _guided_footprint_preview_is_usable(report):
            return candidate

    return best_candidate or best_backup_candidate


def _make_deterministic_seed(*parts):
    """Return a stable 31-bit seed derived from config + call context."""
    seed = int(CONFIG.get("random_seed", 1337)) & 0x7FFFFFFF
    for part in parts:
        if isinstance(part, float):
            value = int(round(part * 1000.0))
        else:
            value = int(part)
        seed = (seed * 1103515245 + 12345 + value) & 0x7FFFFFFF
    return seed or 1


def _set_open3d_random_seed(seed):
    """Best-effort seeding for libraries used by Open3D RANSAC."""
    seed = int(seed) & 0x7FFFFFFF
    np.random.seed(seed)
    o3d_random = getattr(getattr(o3d, "utility", None), "random", None)
    if o3d_random is not None and hasattr(o3d_random, "seed"):
        try:
            o3d_random.seed(seed)
        except Exception:
            pass
    return seed


def _segment_plane_seeded(pcd, distance_threshold, ransac_n, num_iterations, *seed_parts):
    """Wrapper around Open3D segment_plane with deterministic seeding."""
    _set_open3d_random_seed(_make_deterministic_seed(len(pcd.points), *seed_parts))
    return pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )


# =========================================================
# WORLD ALIGN
# =========================================================

def world_align(pcd, center_xy=True, warn_on_tilt=True, return_meta=False):
    """
    Gravity-align the cloud so the dominant floor plane becomes Z = 0.

    Parameters
    ----------
    pcd       : open3d.geometry.PointCloud
    center_xy : bool, default True
        Subtract the XY centroid after tilt correction.  Pass False when
        calling per-frame in walking mode: each frame must keep its real-world
        XY position so that pairwise ICP can measure actual lateral movement.
        (If every frame is XY-centered they all land at (0,0) and ICP returns
        near-identity transforms, collapsing all frames onto each other.)
    warn_on_tilt : bool, default True
        Emit user-facing large-tilt warnings.  Walking mode disables this per
        frame and instead reports a summary after outlier-frame rejection.
    return_meta : bool, default False
        When True, return ``(aligned_pcd, meta_dict)`` where ``meta_dict``
        contains tilt/support diagnostics for walking-mode frame filtering.

    Algorithm
    ---------
    1. Run RANSAC on the bottom 35 % of the raw Z range to find the floor
       plane normal (distance_threshold 0.06 m, 500 iterations).
    2. Build the shortest-arc rotation that maps that normal → [0, 0, 1].
    3. Apply rotation; optionally centre XY on the cloud centroid; shift Z so
       the floor-percentile lands at 0.
    4. Fall back to PCA eigenvector if RANSAC cannot find a clean floor.
    """
    pts = np.asarray(pcd.points)
    if len(pts) < 50:
        raise RuntimeError("Point cloud too small for alignment")

    # --- Step 1: find floor plane normal via RANSAC on lower Z slice ---
    # Retry with progressively larger Z fractions (35 % → 50 % → 65 %) so
    # that high sensor placements (≥ 1.5 m) still find the floor even though
    # only a small area is visible at extreme downward angles.
    # Enforce |n_z| > 0.85: if RANSAC returns a near-vertical plane the slice
    # contained more wall than floor — expand the slice and retry.
    floor_normal = None
    align_meta = {
        "method": "pca",
        "floor_slice_frac": None,
        "floor_inlier_ratio": 0.0,
        "tilt_deg": 0.0,
    }
    z_vals = pts[:, 2]
    z_lo, z_hi = float(z_vals.min()), float(z_vals.max())
    for _frac in (0.35, 0.50, 0.65):
        z_split   = z_lo + _frac * (z_hi - z_lo)
        floor_pts = pts[z_vals < z_split]
        if len(floor_pts) < 100:
            continue
        try:
            tmp        = o3d.geometry.PointCloud()
            tmp.points = o3d.utility.Vector3dVector(floor_pts)
            model, inliers = _segment_plane_seeded(
                tmp,
                distance_threshold=0.06,
                ransac_n=3,
                num_iterations=500,
                *(
                    101,
                    int(round(_frac * 1000.0)),
                    len(floor_pts),
                ),
            )
            n      = np.array(model[:3], dtype=float)
            z_comp = abs(float(n[2])) / (np.linalg.norm(n) + 1e-9)
            if len(inliers) >= 50 and z_comp > 0.85:
                floor_normal = n
                align_meta["method"] = "ransac"
                align_meta["floor_slice_frac"] = float(_frac)
                align_meta["floor_inlier_ratio"] = float(len(inliers)) / max(len(floor_pts), 1)
                log(
                    f"[DEBUG] world_align: floor RANSAC "
                    f"({len(inliers)} inliers from {len(floor_pts)} pts, "
                    f"z_frac={_frac:.0%}, |n_z|={z_comp:.3f})",
                    "debug",
                )
                break
            # Found a plane but it is nearly vertical (wall in slice) — expand
            log(
                f"[DEBUG] world_align: RANSAC z_frac={_frac:.0%} rejected "
                f"(|n_z|={z_comp:.2f} < 0.85, {len(inliers)} inliers) — retrying",
                "debug",
            )
        except Exception:
            pass

    if floor_normal is None:
        # PCA fallback: smallest-variance axis ≈ floor normal
        c0 = pts.mean(axis=0)
        cov = np.cov((pts - c0).T)
        try:
            _, vec = np.linalg.eigh(cov)
            floor_normal = vec[:, 0]
            log("[DEBUG] world_align: PCA fallback (no clean floor found)", "debug")
        except Exception:
            raise RuntimeError("Alignment failed (degenerate geometry)")

    # Canonical orientation: floor normal points up
    if floor_normal[2] < 0:
        floor_normal = -floor_normal
    floor_normal /= np.linalg.norm(floor_normal)

    # --- Step 2: shortest-arc rotation floor_normal → [0, 0, 1] ---
    z_axis = np.array([0.0, 0.0, 1.0])
    v = np.cross(floor_normal, z_axis)
    s = float(np.linalg.norm(v))
    c = float(np.dot(floor_normal, z_axis))
    tilt_deg = float(np.degrees(np.arccos(min(1.0, c))))
    align_meta["tilt_deg"] = tilt_deg

    if s < 1e-8:
        R = np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))

    log(f"[DEBUG] world_align: sensor tilt corrected by {tilt_deg:.2f}°", "debug")

    if tilt_deg > 45.0:
        # A tilt > 45° almost always means RANSAC found a wall or furniture
        # surface instead of the floor (sensor placed on its side, or no
        # usable floor patch in the lower-Z slice).  Applying this rotation
        # would swap floor and walls and corrupt all downstream geometry.
        raise RuntimeError(
            f"Sensor appears to be on its side or badly tilted ({tilt_deg:.1f}°). "
            "RANSAC floor detection found a near-vertical surface instead of the floor.\n"
            "Please:\n"
            "  • Place the sensor upright — connector/base at the bottom\n"
            "  • Ensure the floor is partially visible in the lower field of view\n"
            "  • The sensor should sit flat on a table or tripod, NOT on its side"
        )

    if warn_on_tilt and tilt_deg > 20.0:
        log(
            f"[INFO] Large sensor tilt detected ({tilt_deg:.1f}°). "
            "For best results keep the sensor level and stationary during capture.",
            "info",
        )

    # --- Step 3: rotate, centre XY (optional), shift Z so floor ≈ 0 ---
    aligned = pts @ R.T
    if center_xy:
        aligned[:, 0] -= aligned[:, 0].mean()
        aligned[:, 1] -= aligned[:, 1].mean()
    aligned[:, 2] -= float(
        np.percentile(aligned[:, 2], CONFIG.get("floor_percentile", 2))
    )

    o = o3d.geometry.PointCloud()
    o.points = o3d.utility.Vector3dVector(aligned)
    if return_meta:
        return o, align_meta
    return o


# =========================================================
# LIVE HELPERS
# =========================================================
def _iter_live_scans(scan_batch):
    if hasattr(scan_batch, '__iter__') and not isinstance(scan_batch, np.ndarray):
        for scan in scan_batch:
            yield scan
    else:
        yield scan_batch


def _build_live_point_cloud(scans, point_limit=None):
    if not scans:
        raise RuntimeError("No data received from sensor. Check connection and try again.")

    pts = np.vstack([s.reshape(-1, 3) for s in scans])
    pts = pts[np.any(pts != 0, axis=1)]
    if len(pts) == 0:
        raise RuntimeError("Live capture only contained zero-return pixels.")
    if point_limit is not None and len(pts) > point_limit:
        stride = max(1, int(len(pts) // point_limit))
        pts = pts[::stride]

    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(pts)
    return p


def _build_aligned_live_point_cloud(scans, point_limit=None, max_range_m=None):
    if not scans:
        raise RuntimeError("No data received from sensor. Check connection and try again.")

    aligned_frames = []
    for scan_xyz in scans:
        xyz = np.asarray(scan_xyz).reshape(-1, 3)
        xyz = xyz[np.any(xyz != 0, axis=1)]
        if len(xyz) < 100:
            continue

        frame_pcd = o3d.geometry.PointCloud()
        frame_pcd.points = o3d.utility.Vector3dVector(xyz)
        try:
            frame_pcd, _ = world_align(
                frame_pcd,
                center_xy=False,
                warn_on_tilt=False,
                return_meta=True,
            )
        except Exception:
            continue

        frame_pts = np.asarray(frame_pcd.points)
        if len(frame_pts) == 0:
            continue
        if max_range_m is not None:
            keep = (frame_pts[:, 0] ** 2 + frame_pts[:, 1] ** 2) <= max_range_m ** 2
            if keep.sum() >= 50:
                frame_pts = frame_pts[keep]
        if len(frame_pts) >= 50:
            aligned_frames.append(frame_pts)

    if not aligned_frames:
        return _build_live_point_cloud(scans, point_limit=point_limit)

    pts = np.vstack(aligned_frames)
    if point_limit is not None and len(pts) > point_limit:
        stride = max(1, int(len(pts) // point_limit))
        pts = pts[::stride]

    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(pts)
    return p


def _set_sensor_operating_mode(host, mode_name):
    try:
        from ouster import sdk
    except ImportError:
        return False

    try:
        current_cfg = sdk.sensor.get_config(host, active=True)
        current_cfg.operating_mode = getattr(sdk.core.OperatingMode, mode_name)
        sdk.sensor.set_config(host, current_cfg, persist=False, udp_dest_auto=False)
        log(f"[INFO] Sensor operating mode -> {mode_name}", "info")
        return True
    except Exception as exc:
        log(f"[DEBUG] Sensor operating mode change to {mode_name} skipped: {exc}", "debug")
        return False


def live_capture_walking_guided(host, port, seconds=45, frame_interval_sec=1.0, segment_label="footprint",
                               continuation_hint=None):
    try:
        from ouster import sdk
    except ImportError:
        raise RuntimeError(
            "Ouster SDK not installed.\n"
            "Install with: pip install ouster-sdk"
        )

    if segment_label == "footprint":
        operator_log(
            f"[GUIDE] Footprint step: walk the perimeter for up to {seconds}s. "
            f"The script keeps one position about every {frame_interval_sec:.1f}s."
        )
        operator_log(
            "[GUIDE] Keep the sensor facing inward toward the room centre but mostly level so a thin floor strip and the furniture-height middle band stay visible. "
            "Height support from this lap is only a bonus, so do not chase it with continuous upward tilt. Only add a slight upward tilt briefly where the floor remains visible, usually near corners or more open wall segments, then return to mostly level. "
            "Keep a steady pace along the walls and pause briefly at corners. "
            "Press Enter if you clearly finish the loop sooner."
        )
    else:
        operator_log(
            f"[GUIDE] Footprint continuation: continue for up to {seconds}s."
        )
        operator_log(
            f"[GUIDE] {continuation_hint or 'Do one more short slow lap and pause again at each corner.'}"
        )

    frames_raw = []
    last_kept = -frame_interval_sec
    start = None
    _stop = threading.Event()
    _last_cd_step = -1
    last_preview_check = -999.0
    auto_stop_elapsed = None
    preview_pass_report = None
    stable_preview_frames_raw = None
    usable_preview_frames_raw = None
    usable_preview_score = None
    usable_preview_report = None
    (
        preview_min_frames,
        preview_min_elapsed,
        preview_check_interval,
    ) = _guided_footprint_preview_schedule(
        seconds,
        frame_interval_sec,
    )

    def _enter_watcher():
        try:
            while not _stop.is_set():
                r, _, _ = select.select([sys.stdin], [], [], 0.5)
                if r:
                    sys.stdin.readline()
                    _stop.set()
                    break
        except Exception:
            pass

    _watcher = threading.Thread(target=_enter_watcher, daemon=True)
    _watcher.start()

    try:
        _set_sensor_operating_mode(host, "NORMAL")
        source = sdk.open_source(host, collate=False)
        metadata = source.sensor_info
        if isinstance(metadata, list):
            metadata = metadata[0]
        xyz_lut = sdk.core.XYZLut(metadata)

        start = time.time()
        operator_log("[GUIDE] Sensor connected. Start walking now.")

        for scan_batch in source:
            elapsed = time.time() - start
            remaining = int(seconds - elapsed)

            cd_step = remaining - (remaining % 5)
            if cd_step != _last_cd_step and remaining > 0:
                _last_cd_step = cd_step
                operator_log(
                    f"[GUIDE] Footprint in progress: {len(frames_raw)} kept positions, about {remaining}s left."
                )

            scan = next(iter(_iter_live_scans(scan_batch)))

            if elapsed - last_kept >= frame_interval_sec:
                xyz = xyz_lut(scan).reshape(-1, 3)
                xyz = xyz[np.any(xyz != 0, axis=1)]
                if len(xyz) >= 500:
                    frames_raw.append(xyz)
                    last_kept = elapsed

                    log(
                        f"[DEBUG] Guided footprint: kept frame {len(frames_raw)} at t={elapsed:.1f}s "
                        f"({len(xyz):,} pts)",
                        "debug",
                    )

                    preview_ready = (
                        auto_stop_elapsed is None
                        and len(frames_raw) >= preview_min_frames
                        and elapsed >= preview_min_elapsed
                        and (elapsed - last_preview_check) >= preview_check_interval
                    )
                    if preview_ready:
                        last_preview_check = elapsed
                        try:
                            preview_report = _validate_guided_footprint_frames(frames_raw)
                            preview_fallback_reason = _guided_footprint_fallback_accept_reason(preview_report)
                            preview_usable = _guided_footprint_preview_is_usable(preview_report)
                            preview_worth_saving = _guided_footprint_preview_is_worth_saving(preview_report)
                            if not preview_worth_saving:
                                operator_log(
                                    "[GUIDE] Footprint preview is still too broad for auto-stop, so the script will keep scanning."
                                )
                            elif preview_pass_report is None:
                                usable_preview_score = _guided_footprint_capture_score(preview_report)
                                usable_preview_frames_raw = [frame.copy() for frame in frames_raw]
                                usable_preview_report = dict(preview_report)
                                preview_pass_report = dict(preview_report)
                                if preview_usable and preview_fallback_reason is None:
                                    operator_log(
                                        "[GUIDE] Footprint preview found a provisional candidate: "
                                        f"{_format_guided_footprint_status(preview_report)} "
                                        "The script will save this candidate but keep scanning for one more confirmation before locking the perimeter."
                                    )
                                elif preview_fallback_reason is not None:
                                    operator_log(
                                        "[GUIDE] Footprint preview found a provisional recovered candidate: "
                                        f"{_format_guided_footprint_status(preview_report)} "
                                            "The script will save this candidate as provisional only. It is not accepted yet, and the current lap must still finish before final validation decides whether to stop."
                                    )
                                else:
                                    operator_log(
                                        "[GUIDE] Footprint preview found a promising first-lap candidate: "
                                        f"{_format_guided_footprint_status(preview_report)} "
                                            "The script will save it as provisional only, finish the current lap, and run final validation before deciding whether another pass is needed."
                                    )
                            else:
                                prev_area = float(preview_pass_report.get("floor_area_m2", 0.0) or 0.0)
                                prev_perimeter = float(preview_pass_report.get("floor_perimeter_m", 0.0) or 0.0)
                                curr_area = float(preview_report.get("floor_area_m2", 0.0) or 0.0)
                                curr_perimeter = float(preview_report.get("floor_perimeter_m", 0.0) or 0.0)
                                area_delta = abs(curr_area - prev_area)
                                perimeter_delta = abs(curr_perimeter - prev_perimeter)
                                preview_score = _guided_footprint_capture_score(preview_report)
                                if usable_preview_score is None or _guided_footprint_should_replace_preview_candidate(
                                    preview_report,
                                    frames_raw,
                                    usable_preview_report,
                                    usable_preview_frames_raw,
                                ):
                                    usable_preview_score = preview_score
                                    usable_preview_frames_raw = [frame.copy() for frame in frames_raw]
                                    usable_preview_report = dict(preview_report)
                                if area_delta <= 0.18 and perimeter_delta <= 0.12:
                                    stable_preview_frames_raw = [
                                        frame.copy()
                                        for frame in (usable_preview_frames_raw or frames_raw)
                                    ]
                                    if _guided_footprint_preview_can_autostop(preview_report, elapsed=elapsed, seconds=seconds):
                                        if preview_fallback_reason is None:
                                            operator_log(
                                                "[GUIDE] Footprint preview is stable: "
                                                f"{_format_guided_footprint_status(preview_report)} "
                                                "Finish the current wall segment; the scan will stop automatically in a few seconds and then run final validation."
                                            )
                                        else:
                                            operator_log(
                                                "[GUIDE] Footprint preview is stable enough to finish from this lap: "
                                                f"{_format_guided_footprint_status(preview_report)} "
                                                "One wall is still recovered rather than direct, but the geometry is stable enough to stop and run final validation from this lap."
                                            )
                                        auto_stop_elapsed = elapsed + 3.0
                                    else:
                                        preview_pass_report = dict(preview_report)
                                        if preview_fallback_reason is None:
                                            operator_log(
                                                "[GUIDE] Footprint preview is stable so far, but the perimeter is not complete enough yet. "
                                                "Keep walking to the fourth corner before auto-stop can lock the result."
                                            )
                                        else:
                                            operator_log(
                                                "[GUIDE] Footprint preview is stable so far, but the script still needs the rest of the current lap before it can lock this recovered candidate."
                                            )
                                else:
                                    preview_pass_report = dict(preview_report)
                                    operator_log(
                                        "[GUIDE] Footprint preview is still moving, so the script will keep scanning a little longer before stopping."
                                    )
                        except RuntimeError:
                            pass

            if auto_stop_elapsed is not None and elapsed >= auto_stop_elapsed:
                operator_log("[GUIDE] Stopping the footprint scan for validation.")
                _stop.set()

            if elapsed >= seconds or _stop.is_set():
                if _stop.is_set() and elapsed < seconds:
                    operator_log(
                        f"[GUIDE] Footprint segment stopped at {elapsed:.0f}s with {len(frames_raw)} kept positions."
                    )
                break

    except Exception as e:
        err = str(e)
        if any(k in err for k in ("Timeout", "Connection refused", "Failed to create")):
            raise RuntimeError(
                f"Could not connect to Ouster sensor at {host}.\n"
                f"Check power, IP and network.  Technical detail: {err}"
            )
        raise RuntimeError(f"Guided walking capture failed: {err}")
    finally:
        _stop.set()
        _set_sensor_operating_mode(host, "STANDBY")

    if len(frames_raw) < 3:
        raise RuntimeError(
            f"Guided footprint captured only {len(frames_raw)} usable frames (need ≥ 3). "
            f"Check the sensor connection and repeat the perimeter step."
        )

    operator_log(
        "[GUIDE] Footprint capture complete. The script will validate this perimeter now."
    )

    return {
        "frames_raw": frames_raw,
        "selected_frames_raw": stable_preview_frames_raw or usable_preview_frames_raw or frames_raw,
        "frames_with_hints": 0,
        "frames_with_two_dirs": 0,
        "total_hint_count": 0,
        "total_wall_obs": 0,
        "guidance_announced": False,
    }


def live_capture_height_guided(host, port, seconds=20, footprint_area=None,
                              footprint_height_hint=None, footprint_height_hint_source=None,
                              ready_seconds=8, segment_label="height"):
    try:
        from ouster import sdk
    except ImportError:
        raise RuntimeError(
            "Ouster SDK not installed.\n"
            "Install with: pip install ouster-sdk"
        )

    if segment_label == "height":
        operator_log(
            f"[GUIDE] Height step: move to the room centre and hold still for up to {seconds}s."
        )
    else:
        operator_log(
            f"[GUIDE] Height retry: stay near the room centre, keep still, and try again for up to {seconds}s."
        )
    operator_log(
        "[GUIDE] Keep the sensor upright, leave a sliver of floor visible, and tilt slightly upward until the script stops the scan."
    )
    _operator_ready_countdown(
        "Move to the room centre and get ready for the height scan.",
        ready_seconds,
    )

    scans = []
    start = None
    _stop = threading.Event()
    _last_diag_time = -999.0
    _last_progress_step = -1
    stable_checks = 0
    last_height = None
    weak_support_warned = False
    low_height_warned = False
    floor_alignment_warned = False
    hint_only_warned = False
    best_capture_cloud = None
    best_capture_report = None
    best_capture_score = None

    def _enter_watcher():
        try:
            while not _stop.is_set():
                r, _, _ = select.select([sys.stdin], [], [], 0.5)
                if r:
                    sys.stdin.readline()
                    _stop.set()
                    break
        except Exception:
            pass

    _watcher = threading.Thread(target=_enter_watcher, daemon=True)
    _watcher.start()

    try:
        _set_sensor_operating_mode(host, "NORMAL")
        source = sdk.open_source(host, collate=False)
        metadata = source.sensor_info
        if isinstance(metadata, list):
            metadata = metadata[0]
        xyz_lut = sdk.core.XYZLut(metadata)

        start = time.time()
        operator_log("[GUIDE] Sensor connected. Start the static height hold now.")

        for scan_batch in source:
            elapsed = time.time() - start
            remaining = int(seconds - elapsed)
            for scan in _iter_live_scans(scan_batch):
                scans.append(xyz_lut(scan))

            progress_step = remaining - (remaining % 5)
            if progress_step != _last_progress_step and remaining > 0:
                _last_progress_step = progress_step
                operator_log(
                    f"[GUIDE] Height in progress: {len(scans)} frames collected, about {remaining}s left."
                )

            if elapsed - _last_diag_time >= 2.5 and len(scans) >= 3:
                _last_diag_time = elapsed
                try:
                    p_live, height_report, candidate_score = _select_best_guided_height_candidate(
                        scans,
                        footprint_area=footprint_area,
                        footprint_height_hint=footprint_height_hint,
                        footprint_height_hint_source=footprint_height_hint_source,
                        source_label="live_height_preview",
                    )
                    if p_live is None or height_report is None:
                        raise RuntimeError("No usable recent height window yet")
                    if best_capture_score is None or candidate_score > best_capture_score:
                        best_capture_score = candidate_score
                        best_capture_report = dict(height_report)
                        best_capture_cloud = o3d.geometry.PointCloud(p_live)
                    diag = height_report.get("height_diagnostics", {}) or {}
                    top_band_5cm = int(diag.get("top_band_points_5cm", 0))
                    room_height = height_report.get("room_height_m")
                    height_source = height_report.get("height_source")
                    height_confidence = height_report.get("height_confidence")

                    operator_log(f"[GUIDE] Height preview: {_format_guided_height_status(height_report)}")

                    if room_height is not None:
                        if last_height is not None and abs(float(room_height) - float(last_height)) <= 0.03:
                            stable_checks += 1
                        else:
                            stable_checks = 1
                        last_height = float(room_height)

                    height_ready = True
                    if (
                        footprint_height_hint is not None
                        and room_height is not None
                        and float(room_height) + 0.08 < float(footprint_height_hint)
                    ):
                        height_ready = False
                        if elapsed >= 8.0 and not low_height_warned:
                            low_height_warned = True
                            adjustment = _guided_height_operator_adjustment(
                                height_report,
                                footprint_height_hint=footprint_height_hint,
                            )
                            if adjustment is not None:
                                operator_log(adjustment)

                    diag_floor_slice_points = int(diag.get("floor_slice_points", 0))

                    if (
                        height_report.get("ceiling_detected")
                        and height_ready
                        and _guided_height_has_independent_support(height_report)
                        and elapsed >= 8.0
                    ):
                        operator_log("[GUIDE] Ceiling support looks strong. Stopping the height scan.")
                        _stop.set()
                    elif (
                        room_height is not None
                        and height_ready
                        and _guided_height_has_independent_support(height_report)
                        and stable_checks >= 3
                        and elapsed >= 12.0
                    ):
                        operator_log("[GUIDE] Height estimate is stable with independent static support. Stopping the height scan.")
                        _stop.set()
                    elif elapsed >= 8.0 and top_band_5cm < 10 and not weak_support_warned:
                        weak_support_warned = True
                        adjustment = _guided_height_operator_adjustment(
                            height_report,
                            footprint_height_hint=footprint_height_hint,
                        )
                        if adjustment is not None:
                            operator_log(adjustment)
                    elif (
                        elapsed >= 12.0
                        and not hint_only_warned
                        and diag.get("height_hint_applied")
                        and not _guided_height_has_independent_support(height_report)
                    ):
                        hint_only_warned = True
                        operator_log(
                            "[GUIDE] Height still matches the walking hint only. Keep holding steady with a visible floor sliver; the static step still needs its own floor/ceiling evidence."
                        )

                    if (
                        elapsed >= 6.0
                        and not floor_alignment_warned
                        and (height_report.get("align_method") != "ransac" or diag_floor_slice_points < 80)
                    ):
                        floor_alignment_warned = True
                        operator_log(
                            "[GUIDE] The scan still lacks a strong floor slice for alignment. Keep the sensor level and let a little more floor remain visible before increasing the upward tilt again."
                        )
                except Exception as exc:
                    log(f"[DEBUG] Live height preview skipped: {exc}", "debug")

            if elapsed >= seconds or _stop.is_set():
                if _stop.is_set() and elapsed < seconds:
                    operator_log(f"[GUIDE] Height segment stopped at {elapsed:.0f}s.")
                break

    except Exception as e:
        err = str(e)
        if any(k in err for k in ("Timeout", "Connection refused", "Failed to create")):
            raise RuntimeError(
                f"Could not connect to Ouster sensor at {host}.\n"
                f"Check power, IP and network.  Technical detail: {err}"
            )
        raise RuntimeError(f"Guided height capture failed: {err}")
    finally:
        _stop.set()
        _set_sensor_operating_mode(host, "STANDBY")

    p, full_capture_report, full_capture_score = _select_guided_height_cloud_variant(
        scans,
        footprint_area=footprint_area,
        footprint_height_hint=footprint_height_hint,
        footprint_height_hint_source=footprint_height_hint_source,
        source_label="live_guided_height_full_capture",
        recent_only=False,
    )
    if p is None:
        p = _build_guided_height_analysis_cloud(scans, recent_only=False)
        full_capture_report = None
        full_capture_score = None
    best_capture_ok = False
    if best_capture_report is not None:
        best_capture_ok = _guided_height_retry_reason(
            best_capture_report,
            footprint_height_hint=footprint_height_hint,
        ) is None
    if (
        best_capture_cloud is not None
        and best_capture_ok
        and (full_capture_score is None or best_capture_score is None or best_capture_score >= full_capture_score)
    ):
        p = best_capture_cloud
        operator_log(
            f"[GUIDE] Using the best recent height window from this capture: {_format_guided_height_status(best_capture_report)}"
        )
    elif best_capture_report is not None:
        operator_log(
            "[GUIDE] Ignoring the best recent height window because it still lacks independent static support; validating the full capture instead."
        )
    operator_log(f"[GUIDE] Height capture complete: {len(scans)} frames collected.")
    return p


def _prepare_walking_frames(frames_raw, source_name="Walking scan"):
    """Convert raw XYZ frames into gravity-aligned point clouds for ICP."""

    global LAST_WALKING_QUALITY
    global LAST_WALKING_FRAME_DATA

    # Convert to Open3D PCDs with preprocessing.
    # CRITICAL: gravity-align each frame INDIVIDUALLY before ICP.
    # While walking the sensor tilts slightly with every step (~1-3°).
    # If frames are merged in raw sensor coordinates those tilts accumulate
    # and world_align sees a single cloud whose floor is the mean of many
    # slightly-tilted floor patches — the resulting merged floor normal is
    # noisy enough that RANSAC classifies all planes as floor-like.
    # Aligning each frame to gravity FIRST means ICP only needs to find XY
    # translation + rotation (no Z-tilt component), which is much more
    # robust.  Range clipping is also done per frame to suppress doorway /
    # corridor returns before registration.
    frame_data = []
    skipped = 0
    vs     = CONFIG.get("voxel_size", 0.04)
    for i, xyz in enumerate(frames_raw):
        pcd        = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        try:
            pcd, _align_meta = world_align(
                pcd,
                center_xy=False,
                warn_on_tilt=False,
                return_meta=True,
            )   # gravity-align, keep real XY
        except Exception as e:
            log(f"[DEBUG] Frame {i+1}: world_align skipped ({e})", "debug")
            skipped += 1
            if skipped > len(frames_raw) // 2:
                raise RuntimeError(
                    f"More than half of walking frames failed gravity alignment "
                    f"({skipped}/{len(frames_raw)}).  Check that the floor is "
                    f"visible and the sensor is held upright."
                )
            continue
        # Clip to max_range_m from the sensor (origin).
        # After world_align(center_xy=False) the coordinate system is rotated
        # for gravity but NOT translated — the sensor remains at XY≈(0,0).
        # Clipping from the centroid of all returns is WRONG: when the sensor
        # faces an open door/window, 50% of returns are outdoor (corridor) and
        # 50% are indoor; the centroid is pulled halfway outside, so the 3.5 m
        # circle includes all outdoor returns.  Clipping from the origin (sensor
        # position) always excludes returns beyond max_range_m from the sensor.
        _r = CONFIG.get("max_range_m", 3.5)
        _pts_f = np.asarray(pcd.points)
        if len(_pts_f) and _r is not None:
            _r2 = _pts_f[:, 0] ** 2 + _pts_f[:, 1] ** 2   # distance from sensor
            _keep = _r2 <= _r ** 2
            if _keep.sum() >= 50:
                pcd = pcd.select_by_index(np.where(_keep)[0])
        semantic_pcd = pcd.voxel_down_sample(min(vs, 0.02)) if len(pcd.points) > 5000 else pcd
        pcd = pcd.voxel_down_sample(vs)
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=30))
        frame_data.append({"pcd": pcd, "semantic_pcd": semantic_pcd, "meta": _align_meta})
        log(
            f"[DEBUG] Frame {i+1}/{len(frames_raw)}: {len(pcd.points)} pts after align+clip+voxel "
            f"(tilt={_align_meta['tilt_deg']:.1f}°, method={_align_meta['method']})",
            "debug",
        )

    _med_tilt = _p90_tilt = _max_tilt = 0.0
    _dropped = 0
    if len(frame_data) >= 4:
        _tilts = np.array([float(fd["meta"].get("tilt_deg", 0.0)) for fd in frame_data])
        _med_tilt = float(np.median(_tilts))
        _p90_tilt = float(np.percentile(_tilts, 90))
        _max_tilt = float(_tilts.max())
        _tilt_keep_max = max(15.0, _med_tilt + 7.5)
        _keep = _tilts <= _tilt_keep_max
        if _keep.sum() >= max(3, len(frame_data) - 2) and _keep.sum() < len(frame_data):
            _dropped = int(len(frame_data) - _keep.sum())
            log(
                f"[INFO] Walking frame tilt summary: median={_med_tilt:.1f}°, "
                f"p90={_p90_tilt:.1f}°, max={_max_tilt:.1f}° — dropping {_dropped} tilt outlier frame(s)",
                "info",
            )
            frame_data = [fd for fd, keep in zip(frame_data, _keep) if keep]
            _tilts = _tilts[_keep]
            _med_tilt = float(np.median(_tilts))
            _p90_tilt = float(np.percentile(_tilts, 90))
            _max_tilt = float(_tilts.max())
        else:
            log(
                f"[INFO] Walking frame tilt summary: median={_med_tilt:.1f}°, "
                f"p90={_p90_tilt:.1f}°, max={_max_tilt:.1f}°",
                "info",
            )
        if _p90_tilt > 15.0:
            log(
                f"[INFO] Large walking-frame tilt persists after filtering "
                f"(p90={_p90_tilt:.1f}°, max={_max_tilt:.1f}°). "
                "Keep the sensor level and avoid pitching it toward furniture.",
                "info",
            )

    LAST_WALKING_QUALITY = {
        "source_name": source_name,
        "frames_input": len(frames_raw),
        "frames_aligned": len(frame_data),
        "tilt_median_deg": float(_med_tilt),
        "tilt_p90_deg": float(_p90_tilt),
        "tilt_max_deg": float(_max_tilt),
        "tilt_dropped_frames": int(_dropped),
    }
    LAST_WALKING_FRAME_DATA = frame_data

    pcds = [fd["pcd"] for fd in frame_data]

    if len(pcds) < 3:
        raise RuntimeError(
            f"Only {len(pcds)} frames survived gravity alignment (need ≥ 3). "
            "Ensure the floor is visible and the sensor is held upright."
        )
    return pcds


def _point_field_dtype(datatype, is_bigendian):
    prefix = ">" if is_bigendian else "<"
    types = {
        1: "i1",
        2: "u1",
        3: "i2",
        4: "u2",
        5: "i4",
        6: "u4",
        7: "f4",
        8: "f8",
    }
    if datatype not in types:
        return None
    return np.dtype(prefix + types[datatype])


def _pointcloud2_xyz(msg):
    count = int(msg.width) * int(msg.height)
    if count <= 0 or not msg.data:
        return np.empty((0, 3), dtype=np.float32)

    names = []
    formats = []
    offsets = []
    is_bigendian = bool(getattr(msg, "is_bigendian", False))

    for field in sorted(msg.fields, key=lambda item: item.offset):
        dtype = _point_field_dtype(int(field.datatype), is_bigendian)
        if dtype is None:
            continue
        names.append(field.name)
        count_i = max(1, int(getattr(field, "count", 1)))
        if count_i > 1:
            dtype = np.dtype((dtype, (count_i,)))
        formats.append(dtype)
        offsets.append(int(field.offset))

    dtype = np.dtype({
        "names": names,
        "formats": formats,
        "offsets": offsets,
        "itemsize": int(msg.point_step),
    })

    if not {"x", "y", "z"}.issubset(dtype.names):
        raise RuntimeError("PointCloud2 message does not contain x/y/z fields")

    buffer = msg.data if isinstance(msg.data, (bytes, bytearray, memoryview)) else bytes(msg.data)
    arr = np.frombuffer(buffer, dtype=dtype, count=count)
    xyz = np.column_stack([
        np.asarray(arr["x"], dtype=np.float32).reshape(-1),
        np.asarray(arr["y"], dtype=np.float32).reshape(-1),
        np.asarray(arr["z"], dtype=np.float32).reshape(-1),
    ])
    mask = np.isfinite(xyz).all(axis=1) & np.any(xyz != 0, axis=1)
    return xyz[mask]


def _bytes_from_ros_value(value):
    if value is None:
        return b""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    try:
        return bytes(value)
    except TypeError:
        return b""


def _copy_payload_into_packet(packet, payload, packet_size):
    packet_bytes = payload[:packet_size]

    try:
        buf = packet.buf
        np.copyto(np.asarray(buf, dtype=np.uint8), np.frombuffer(packet_bytes, dtype=np.uint8))
        return
    except Exception:
        pass

    try:
        packet.buf = packet_bytes
        return
    except Exception as exc:
        raise RuntimeError(
            "Failed to copy ROS bag packet bytes into an Ouster SDK LidarPacket. "
            "This usually means the installed ouster-sdk Python binding expects a different packet buffer API."
        ) from exc


def _extract_metadata_text(msg):
    for attr in ("data", "metadata", "json"):
        value = getattr(msg, attr, None)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _get_pointcloud_connections(reader):
    cons = [c for c in reader.connections if "PointCloud2" in c.msgtype]
    if not cons:
        return []

    for topic in ("/ouster/points", "/points"):
        topic_cons = [c for c in cons if c.topic == topic]
        if topic_cons:
            return topic_cons

    for suffix in ("/points", "/points1"):
        topic_cons = [c for c in cons if c.topic.endswith(suffix)]
        if topic_cons:
            return topic_cons

    topics = sorted({c.topic for c in cons})
    return [c for c in cons if c.topic == topics[0]]


def _get_lidar_packet_connections(reader):
    packet_cons = [
        c for c in reader.connections
        if "PacketMsg" in c.msgtype and "lidar" in c.topic.lower()
    ]
    if not packet_cons:
        return []

    preferred_topics = (
        "/ouster/lidar_packets",
        "/os_node/lidar_packets",
    )
    for topic in preferred_topics:
        topic_cons = [c for c in packet_cons if c.topic == topic]
        if topic_cons:
            return topic_cons

    topics = sorted({c.topic for c in packet_cons})
    return [c for c in packet_cons if c.topic == topics[0]]


def _get_metadata_connections(reader):
    metadata_cons = [c for c in reader.connections if "metadata" in c.topic.lower()]
    if not metadata_cons:
        return []

    for topic in ("/ouster/metadata", "/metadata"):
        topic_cons = [c for c in metadata_cons if c.topic == topic]
        if topic_cons:
            return topic_cons

    topics = sorted({c.topic for c in metadata_cons})
    return [c for c in metadata_cons if c.topic == topics[0]]


def _metadata_json_path(path):
    bag_path = Path(path)
    search_dir = bag_path if bag_path.is_dir() else bag_path.parent
    if not search_dir.exists():
        return None

    json_files = sorted(search_dir.glob("*.json"))
    if not json_files:
        return None

    preferred = [p for p in json_files if "metadata" in p.name.lower()]
    return preferred[0] if preferred else json_files[0]


def _load_rosbag_metadata_text(reader, path):
    metadata_cons = _get_metadata_connections(reader)
    if metadata_cons:
        for conn, _, rawdata in reader.messages(connections=metadata_cons):
            msg = reader.deserialize(rawdata, conn.msgtype)
            metadata_text = _extract_metadata_text(msg)
            if metadata_text:
                return metadata_text, f"topic {conn.topic}"

    metadata_path = _metadata_json_path(path)
    if metadata_path:
        return metadata_path.read_text(), f"file {metadata_path}"

    return None, None


def _scan_xyz_from_packet_bag(reader, path, frame_interval_sec=1.0):
    try:
        from ouster.sdk import core
    except Exception as exc:
        raise RuntimeError(
            "Packet-based ROS bag support requires the Ouster SDK Python package."
        ) from exc

    packet_cons = _get_lidar_packet_connections(reader)
    if not packet_cons:
        return []

    metadata_text, metadata_source = _load_rosbag_metadata_text(reader, path)
    if not metadata_text:
        available_topics = sorted({c.topic for c in reader.connections})
        raise RuntimeError(
            "ROS bag contains raw lidar packets but no usable metadata source was found.\n"
            "Expected /ouster/metadata or a metadata JSON file next to the bag.\n\n"
            f"Available topics: {', '.join(available_topics[:20])}"
        )

    info = core.SensorInfo(metadata_text)
    batcher = core.ScanBatcher(info)
    scan = core.LidarScan(info)
    xyz_lut = core.XYZLut(info)
    packet_format = core.PacketFormat.from_info(info)
    packet_size = int(packet_format.lidar_packet_size)
    frame_interval_ns = int(frame_interval_sec * 1e9)
    last_kept_ns = None
    frames_raw = []

    log(
        f"[INFO] ROS bag walking mode: reconstructing scans from {packet_cons[0].topic} "
        f"using metadata from {metadata_source}",
        "info",
    )

    for conn, _, rawdata in reader.messages(connections=packet_cons):
        msg = reader.deserialize(rawdata, conn.msgtype)
        payload = _bytes_from_ros_value(getattr(msg, "buf", None))
        if len(payload) < packet_size:
            continue

        packet = core.LidarPacket(packet_size)
        _copy_payload_into_packet(packet, payload, packet_size)

        if not batcher(packet, scan):
            continue

        frame_ts = int(scan.get_first_valid_packet_timestamp())
        if frame_ts <= 0:
            frame_ts = int(scan.get_first_valid_column_timestamp())
        if frame_ts <= 0:
            frame_ts = int(scan.frame_id)

        if last_kept_ns is not None and (frame_ts - last_kept_ns) < frame_interval_ns:
            continue

        xyz = np.asarray(xyz_lut(scan.field(core.ChanField.RANGE)), dtype=np.float32)
        xyz = xyz.reshape(-1, 3)
        xyz = xyz[np.isfinite(xyz).all(axis=1) & np.any(xyz != 0, axis=1)]
        if len(xyz) < 500:
            continue

        frames_raw.append(xyz)
        last_kept_ns = frame_ts

    return frames_raw


def _open_anyreader(path):
    try:
        from rosbags.highlevel import AnyReader
    except Exception:
        raise RuntimeError("Install rosbags to read ROS 2 bag directories")

    try:
        return AnyReader([Path(path)])
    except Exception as exc:
        msg = str(exc)
        if "Rosbag2 version" in msg and "not supported" in msg:
            raise RuntimeError(
                "The installed rosbags package is too old for this ROS 2 bag format.\n"
                "This bag was recorded by a newer rosbag2 writer.\n\n"
                "Fix:\n"
                "  python3 -m pip install --upgrade 'rosbags>=0.11.0'\n\n"
                "If you are using the analysis Docker image, rebuild it after pulling the latest Dockerfile."
            ) from exc
        raise


def load_rosbag_frames(path, frame_interval_sec=1.0):
    frames_raw = []
    last_kept_ns = None
    frame_interval_ns = int(frame_interval_sec * 1e9)

    with _open_anyreader(path) as reader:
        cons = _get_pointcloud_connections(reader)
        if cons:
            log(
                f"[INFO] ROS bag walking mode: reading topic {cons[0].topic} "
                f"from {path}",
                "info",
            )

            for conn, timestamp, rawdata in reader.messages(connections=cons):
                if last_kept_ns is not None and (timestamp - last_kept_ns) < frame_interval_ns:
                    continue

                msg = reader.deserialize(rawdata, conn.msgtype)
                xyz = _pointcloud2_xyz(msg)
                if len(xyz) < 500:
                    continue

                frames_raw.append(xyz)
                last_kept_ns = timestamp
        else:
            frames_raw = _scan_xyz_from_packet_bag(
                reader,
                path,
                frame_interval_sec=frame_interval_sec,
            )

            if not frames_raw:
                available_topics = sorted({c.topic for c in reader.connections})
                raise RuntimeError(
                    "No usable Ouster scan topics found in ROS bag.\n"
                    "Expected one of:\n"
                    "  - PointCloud2 on /ouster/points (or similar)\n"
                    "  - raw lidar packets on /ouster/lidar_packets plus metadata\n\n"
                    f"Available topics: {', '.join(available_topics[:20])}"
                )

    if len(frames_raw) < 3:
        raise RuntimeError(
            f"ROS bag walking mode found only {len(frames_raw)} usable frames "
            f"in {path}. Record longer or reduce --bag-frame-interval."
        )

    log(
        f"[INFO] ROS bag walking mode: kept {len(frames_raw)} frames at "
        f"{frame_interval_sec:.1f}s spacing. Running ICP registration …",
        "info",
    )
    return _prepare_walking_frames(frames_raw, source_name="ROS bag walking scan")


# =========================================================
# LOAD FILE
# =========================================================

def _is_rosbag_path(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".bag", ".db3", ".mcap"]:
        return True

    if os.path.isdir(path):
        entries = set(os.listdir(path))
        if "metadata.yaml" in entries:
            return any(name.endswith((".mcap", ".db3")) for name in entries)

    return False


def load_file(path):

    ext=os.path.splitext(path)[1].lower()

    if ext in [".pcd",".ply"]:
        return o3d.io.read_point_cloud(path)

    if _is_rosbag_path(path):
        pts=[]

        with _open_anyreader(path) as r:
            cons=_get_pointcloud_connections(r)
            if cons:
                for c,t,d in r.messages(cons):
                    msg=r.deserialize(d,c.msgtype)
                    xyz = _pointcloud2_xyz(msg)
                    if len(xyz):
                        pts.extend(xyz.tolist())
            else:
                for xyz in _scan_xyz_from_packet_bag(r, path, frame_interval_sec=0.0):
                    if len(xyz):
                        pts.extend(xyz.tolist())

        if not pts:
            raise RuntimeError("No points found in ROS bag")

        arr=np.array(pts)
        p=o3d.geometry.PointCloud()
        p.points=o3d.utility.Vector3dVector(arr)
        return p

    raise RuntimeError("Unsupported file format")


# =========================================================
# PREPROCESS
# =========================================================

def preprocess(p):

    if len(p.points)==0:
        raise RuntimeError("Empty point cloud")

    p=p.voxel_down_sample(CONFIG["voxel_size"])

    p,_=p.remove_statistical_outlier(
        CONFIG["outlier_neighbors"],
        CONFIG["outlier_std"]
    )

    if len(p.points)<50:
        raise RuntimeError("Cloud vanished after filtering")

    p.estimate_normals(
        o3d.geometry.KDTreeSearchParamKNN(knn=50)
    )

    return p


# =========================================================
# ICP
# =========================================================

def _planar_trans(T):
    """Constrain a 6-DOF ICP 4×4 transform to SE(2): XY translation + Z rotation.

    Per-frame world_align ensures each frame has Z=0 at the floor.  Adjacent-
    frame ICP should therefore only need XY translation + rotation about the
    vertical axis.  However point-to-plane ICP has 6 DOF and in practice
    accumulates small Z-tilt components across many steps (~0.5°/step), which
    amounts to a 20°+ net floor tilt after 43 frames.  Stripping the tilt
    components after each step prevents this drift entirely.

    The planar projection:
      1. Extract Z-rotation angle from the XY block of R.
      2. Rebuild a pure Z-rotation matrix from that angle.
      3. Keep only T[0,3] and T[1,3] (XY translation); zero Z translation.
    """
    # Z-rotation angle from the XY block  (atan2 of the in-plane rotation)
    theta = float(np.arctan2(T[1, 0], T[0, 0]))
    c, s  = np.cos(theta), np.sin(theta)
    T_planar = np.eye(4)
    T_planar[0, 0] =  c;  T_planar[0, 1] = -s
    T_planar[1, 0] =  s;  T_planar[1, 1] =  c
    T_planar[0, 3] = T[0, 3]   # XY translation kept
    T_planar[1, 3] = T[1, 3]
    # T_planar[2, 3] = 0        # Z translation zeroed (already 0 from np.eye)
    return T_planar


def pairwise_icp(source, target, init_trans=None, planar=False):

    if init_trans is None:
        init_trans = np.eye(4)

    if CONFIG["icp_downsample"]:
        source=source.voxel_down_sample(CONFIG["voxel_size"])
        target=target.voxel_down_sample(CONFIG["voxel_size"])

    reg=o3d.pipelines.registration.registration_icp(
        source,target,
        CONFIG["icp_max_dist"],
        init_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

    trans = reg.transformation
    if planar:
        trans = _planar_trans(trans)
    return trans, reg.fitness


# =========================================================
# POSE GRAPH WITH LOOP CLOSURE
# =========================================================

def full_registration(pcds, planar=False):
    """Build pose graph using sequential odometry edges + start/end loop closure.

    planar=True  (walking mode): each ICP result is projected to SE(2)
    (XY translation + Z rotation only) before accumulation.  This prevents the
    ~0.5°/step Z-tilt drift that would otherwise accumulate to ~20° over 43
    frames and corrupt wall-plane classification.

    planar=False (static / file mode): standard 6-DOF ICP.
    """
    n = len(pcds)
    lc_threshold = CONFIG["loop_fitness_threshold"]

    pose_graph = o3d.pipelines.registration.PoseGraph()

    # accumulated[i] = cumulative transform from frame-0 to frame-i
    accumulated = [np.identity(4)]
    pose_graph.nodes.append(
        o3d.pipelines.registration.PoseGraphNode(np.identity(4))
    )

    # ── Step 1: sequential odometry (adjacent frames only) ──────────────────
    for s in range(n - 1):
        t = s + 1
        trans, fit = pairwise_icp(pcds[s], pcds[t], planar=planar)
        acc_t = trans @ accumulated[s]
        accumulated.append(acc_t)
        pose_graph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(acc_t))
        )
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(s, t, trans, uncertain=False)
        )
        log(f"[DEBUG] Odometry {s}→{t}: fitness={fit:.3f}", "debug")

    # ── Step 2: loop closure (start ↔ end of walk) ──────────────────────────
    # Use 3 frames from each end; skip if not enough frames for a real loop.
    window = min(3, n // 4)  # require at least 4x the window to call it a loop
    start_nodes = list(range(window))
    end_nodes   = list(range(max(window, n - window), n))

    for s in start_nodes:
        for t in end_nodes:
            if t <= s + 1:
                continue
            # Use accumulated odometry as the initial transform for ICP
            init_t = accumulated[t] @ np.linalg.inv(accumulated[s])
            trans, fit = pairwise_icp(pcds[s], pcds[t], init_trans=np.linalg.inv(init_t), planar=planar)
            log(f"[DEBUG] Loop closure {s}→{t}: fitness={fit:.3f}", "debug")
            if fit > lc_threshold:
                log(f"[INFO] Loop closure {s}→{t} added (fitness={fit:.3f})", "info")
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        s, t, trans, uncertain=True
                    )
                )

    # ── Step 3: global optimisation ─────────────────────────────────────────
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=0.6,
        edge_prune_threshold=0.25,
        reference_node=0
    )
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option
    )

    return pose_graph


# =========================================================
# MERGE CLOUDS
# =========================================================

def merge_clouds(pcds,pose_graph):

    p=o3d.geometry.PointCloud()

    for i,pcd in enumerate(pcds):
        pcd_t = o3d.geometry.PointCloud(pcd)
        pcd_t.transform(pose_graph.nodes[i].pose)
        p+=pcd_t

    if len(p.points)==0:
        raise RuntimeError("Merged cloud empty")

    return p


def _transform_point_clouds(pcds, pose_graph):
    transformed = []
    for i, pcd in enumerate(pcds):
        if i >= len(pose_graph.nodes):
            break
        pcd_t = o3d.geometry.PointCloud(pcd)
        pcd_t.transform(pose_graph.nodes[i].pose)
        transformed.append(pcd_t)
    return transformed


def _walking_hint_metrics(hints):
    if not hints:
        return {
            "count": 0,
            "dirs": 0,
            "area": 0.0,
        }

    return {
        "count": int(len(hints)),
        "dirs": int(_count_distinct_wall_dirs(hints)),
        "area": float(sum(max(float(get_plane_area(h)), 0.0) for h in hints)),
    }


def _walking_support_metrics(hints):
    return {
        "hint_count": int(len(hints)) if hints else 0,
        "wall_observations": int(len(_merge_wall_faces(hints))) if hints else 0,
    }


def _select_walking_hint_source(pre_hints, post_hints):
    pre = _walking_hint_metrics(pre_hints)
    post = _walking_hint_metrics(post_hints)

    use_pre = (
        pre["count"] >= 12
        and pre["dirs"] >= 2
        and post["count"] <= 2
    )

    if use_pre:
        log(
            f"[INFO] Walking hint selection: using pre-registration hints "
            f"({pre['count']} candidates, {pre['dirs']} dirs) over transformed hints "
            f"({post['count']} candidates, {post['dirs']} dirs)",
            "info",
        )
        return "pre"

    log(
        f"[INFO] Walking hint selection: using transformed hints "
        f"({post['count']} candidates, {post['dirs']} dirs); "
        f"pre-registration hints had {pre['count']} candidates",
        "info",
    )
    return "post"


def _transform_walking_frame_data(frame_data, pose_graph, key="semantic_pcd", z_shift=0.0):
    transformed = []
    if not frame_data:
        return transformed

    for i, frame in enumerate(frame_data):
        if i >= len(pose_graph.nodes):
            break
        pcd = frame.get(key)
        if pcd is None or len(pcd.points) == 0:
            continue
        pts = np.asarray(pcd.points)
        T = pose_graph.nodes[i].pose
        pts_h = np.column_stack([pts, np.ones(len(pts), dtype=float)])
        pts_t = (T @ pts_h.T).T[:, :3]
        if z_shift:
            pts_t = pts_t.copy()
            pts_t[:, 2] -= float(z_shift)
        transformed.append(pts_t)

    return transformed


def _merge_walking_frame_data_cloud(frame_data, pose_graph, key="semantic_pcd", voxel_size=0.02, z_shift=0.0):
    if not frame_data:
        return None

    merged_pts = _transform_walking_frame_data(
        frame_data,
        pose_graph,
        key=key,
        z_shift=z_shift,
    )

    if not merged_pts:
        return None

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(np.vstack(merged_pts))
    if voxel_size is not None and len(out.points) > 0:
        out = out.voxel_down_sample(voxel_size)
    return out


# =========================================================
# REMOVE CLUTTER
# =========================================================

def remove_clutter(p):

    pts=np.asarray(p.points)
    z=pts[:,2]

    floor=np.percentile(z,CONFIG["floor_percentile"])
    mask=np.abs(z-floor) < CONFIG["floor_offset"]

    clean=pts[mask]

    if len(clean)<50:
        raise RuntimeError("Floor detection failed")

    o=o3d.geometry.PointCloud()
    o.points=o3d.utility.Vector3dVector(clean)
    return o


# =========================================================
# PLANES
# =========================================================

def _plane_area(inlier_pts, normal):
    """Compute plane area by projecting inliers onto the plane's 2D coordinate system."""
    from shapely.geometry import MultiPoint
    z_comp = abs(normal[2])
    if z_comp > 0.7:
        # Horizontal plane — project to XY
        proj = inlier_pts[:, :2]
    else:
        # Vertical plane — project onto plane-local axes
        up = np.array([0.0, 0.0, 1.0])
        ax1 = np.cross(normal, up)
        n1 = np.linalg.norm(ax1)
        if n1 < 1e-6:
            ax1 = np.cross(normal, np.array([1.0, 0.0, 0.0]))
            n1 = np.linalg.norm(ax1)
        ax1 /= n1
        ax2 = np.cross(normal, ax1)
        ax2 /= np.linalg.norm(ax2)
        v = inlier_pts - inlier_pts.mean(axis=0)
        proj = np.column_stack([v @ ax1, v @ ax2])
    try:
        return MultiPoint(proj).convex_hull.area
    except Exception:
        return 0.0


def extract_planes(p, plane_log_level="debug"):
    """
    Extract floor and wall planes using iterative RANSAC plane segmentation.
    Returns list of plane dicts: {normal, centroid, area, shell_coords}.
    """
    log(f"[DEBUG] extract_planes input: {len(p.points)} points", "debug")
    pts_arr = np.asarray(p.points)
    log(
        f"[DEBUG] Bounds: X=[{pts_arr[:,0].min():.2f},{pts_arr[:,0].max():.2f}] "
        f"Y=[{pts_arr[:,1].min():.2f},{pts_arr[:,1].max():.2f}] "
        f"Z=[{pts_arr[:,2].min():.2f},{pts_arr[:,2].max():.2f}]",
        "debug"
    )

    # Work on a moderately downsampled copy for speed
    working = p.voxel_down_sample(max(CONFIG["voxel_size"], 0.05)) if len(p.points) > 10000 else p
    if not working.has_normals():
        working.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=30))

    planes = []
    remaining = working

    def _plane_sort_key(plane):
        normal = np.array(plane["normal"], dtype=float)
        normal /= np.linalg.norm(normal) + 1e-9
        centroid = np.array(plane["centroid"], dtype=float)
        area = float(plane["area"])
        return (
            -round(area, 4),
            round(float(normal[0]), 4),
            round(float(normal[1]), 4),
            round(float(normal[2]), 4),
            round(float(centroid[0]), 3),
            round(float(centroid[1]), 3),
            round(float(centroid[2]), 3),
        )

    for iteration in range(CONFIG["max_planes"]):
        if len(remaining.points) < CONFIG["min_plane_points"]:
            log(f"[DEBUG] Only {len(remaining.points)} points left — stopping", "debug")
            break

        try:
            plane_model, inliers = _segment_plane_seeded(
                remaining,
                distance_threshold=CONFIG["ransac_dist"],
                ransac_n=3,
                num_iterations=1000,
                *(
                    202,
                    iteration,
                    len(remaining.points),
                ),
            )
        except Exception as e:
            log(f"[DEBUG] RANSAC failed at iteration {iteration}: {e}", "debug")
            break

        if len(inliers) < CONFIG["min_plane_points"]:
            log(f"[DEBUG] Plane {iteration}: only {len(inliers)} inliers — stopping", "debug")
            break

        normal = np.array(plane_model[:3], dtype=float)
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal /= norm
        # Canonical orientation: Z component non-negative
        if normal[2] < 0:
            normal = -normal

        inlier_cloud = remaining.select_by_index(inliers)
        inlier_pts   = np.asarray(inlier_cloud.points)
        centroid     = inlier_pts.mean(axis=0)
        area         = _plane_area(inlier_pts, normal)

        remaining = remaining.select_by_index(inliers, invert=True)

        log(
            f"[DEBUG] RANSAC plane {iteration}: z_comp={abs(normal[2]):.2f} "
            f"area={area:.2f} m²  inliers={len(inliers)}",
            plane_log_level
        )

        if area < CONFIG["min_plane_area"]:
            log(f"[DEBUG]  → skipped (area < {CONFIG['min_plane_area']} m²)", "debug")
            continue

        planes.append({
            'shell_coords': inlier_pts,
            'normal'      : normal,
            'centroid'    : centroid,
            'area'        : area,
        })

    log(f"[DEBUG] extract_planes found {len(planes)} valid planes (before merge)", "debug")
    planes = sorted(planes, key=_plane_sort_key)
    planes = _merge_coplanar(planes)
    log(f"[DEBUG] After merging coplanar planes: {len(planes)}", "debug")
    return planes


def _make_horizontal_plane_from_slice(pcd, z_percentile=5, min_points=None):
    """Build a synthetic horizontal plane from a filtered floor/ceiling slice.

    Used as a fallback when the slice clearly contains a strong horizontal band
    but RANSAC fails to return a stable plane because the slice is sparse or
    fragmented after walking-mode filtering.
    """
    if pcd is None or len(pcd.points) == 0:
        return None

    coords = np.asarray(pcd.points)
    required_points = int(min_points or CONFIG["min_plane_points"])
    if len(coords) < required_points:
        return None

    z_ref = float(np.percentile(coords[:, 2], z_percentile))
    band = np.abs(coords[:, 2] - z_ref) <= max(CONFIG["ransac_dist"] * 2.0, 0.06)
    band_coords = coords[band] if int(band.sum()) >= required_points else coords
    area = float(_plane_area(band_coords, np.array([0.0, 0.0, 1.0])))
    if area < max(0.20, 0.6 * CONFIG["min_plane_area"]):
        return None

    centroid = band_coords.mean(axis=0)
    centroid[2] = z_ref
    return {
        "normal": np.array([0.0, 0.0, 1.0]),
        "centroid": centroid,
        "area": area,
        "shell_coords": band_coords,
        "synthetic": True,
    }


def _estimate_static_height_report(p_aligned, footprint_area=None, allow_relaxed_floor=False):
    """Estimate room height from a dedicated static upward-looking scan.

    This path is intentionally independent from full-room reconstruction.
    A height bag does not need enough wall support to fit a footprint, so it
    should not be forced through the generic full-room plane pipeline.
    """
    if p_aligned is None or len(p_aligned.points) == 0:
        raise RuntimeError("Height bag is empty after preprocessing.")

    pts = np.asarray(p_aligned.points)
    voxel_size = CONFIG.get("voxel_size", 0.04) * 3.0
    p_vox = p_aligned.voxel_down_sample(voxel_size)
    if not p_vox.has_normals():
        p_vox.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=30))

    def _slice_z(pcd, z_lo, z_hi):
        arr = np.asarray(pcd.points)
        idx = np.where((arr[:, 2] >= z_lo) & (arr[:, 2] <= z_hi))[0]
        if len(idx) == 0:
            return None
        out = pcd.select_by_index(idx.tolist())
        if not out.has_normals() and len(out.points):
            out.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=30))
        return out

    def _filter_local_normals(pcd, min_abs_nz=None, max_abs_nz=None):
        if pcd is None or len(pcd.points) == 0:
            return None
        if not pcd.has_normals():
            pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=30))
        nrms = np.asarray(pcd.normals)
        keep = np.ones(len(nrms), dtype=bool)
        abs_nz = np.abs(nrms[:, 2])
        if min_abs_nz is not None:
            keep &= abs_nz >= min_abs_nz
        if max_abs_nz is not None:
            keep &= abs_nz <= max_abs_nz
        idx = np.where(keep)[0]
        if len(idx) == 0:
            return None
        out = pcd.select_by_index(idx.tolist())
        if len(out.points):
            out.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=30))
        return out

    zmax = float(np.max(np.asarray(p_vox.points)[:, 2]))
    ceil_est = max(1.5, zmax - 0.35)

    floor_slice = _filter_local_normals(_slice_z(p_vox, -0.30, 0.25), min_abs_nz=0.75)
    ceil_slice = _filter_local_normals(_slice_z(p_vox, ceil_est, zmax + 0.10), min_abs_nz=0.75)

    floor_plane = _make_horizontal_plane_from_slice(floor_slice) if floor_slice is not None else None
    if floor_plane is None and allow_relaxed_floor and floor_slice is not None:
        floor_plane = _make_horizontal_plane_from_slice(
            floor_slice,
            min_points=max(60, CONFIG["min_plane_points"] // 2),
        )
    if floor_plane is None:
        raise RuntimeError("Height bag does not contain a usable floor slice.")

    floor_z = float(get_plane_centroid(floor_plane)[2])
    ceiling_plane = None
    if ceil_slice is not None and len(ceil_slice.points) >= max(40, CONFIG["min_plane_points"] // 2):
        ceil_planes = [
            plane for plane in extract_planes(ceil_slice)
            if abs(float(get_plane_normal(plane)[2])) > 0.80
        ]
        if ceil_planes:
            ceiling_plane = max(ceil_planes, key=lambda plane: float(get_plane_area(plane)))
        else:
            fallback = _make_horizontal_plane_from_slice(
                ceil_slice,
                z_percentile=95,
                min_points=max(40, CONFIG["min_plane_points"] // 2),
            )
            if fallback is not None:
                ceiling_plane = fallback

    top_z_max = float(np.max(pts[:, 2]))
    top_z_p999 = float(np.percentile(pts[:, 2], 99.9))
    top_z_p9999 = float(np.percentile(pts[:, 2], 99.99))
    top_band_2cm = int(np.sum(pts[:, 2] >= top_z_max - 0.02))
    top_band_5cm = int(np.sum(pts[:, 2] >= top_z_max - 0.05))
    top_band_10cm = int(np.sum(pts[:, 2] >= top_z_max - 0.10))

    measurement_warnings = []
    if ceiling_plane is not None:
        room_height = float(get_plane_centroid(ceiling_plane)[2] - floor_z)
        height_source = "static_ceiling_slice"
        height_confidence = "high"
        ceiling_detected = True
    else:
        if top_band_5cm >= 25:
            top_ref = top_z_p9999
            height_source = "static_top_envelope_p99_99"
        elif top_band_10cm >= 100:
            top_ref = top_z_p999
            height_source = "static_top_envelope_p99_9"
        else:
            top_ref = top_z_max
            height_source = "static_top_envelope_max"
        room_height = float(top_ref - floor_z)
        height_confidence = "low"
        ceiling_detected = False
        measurement_warnings.append(
            "No ceiling plane was detected; height comes from a static top-envelope heuristic."
        )

    volume = None
    if footprint_area is not None:
        volume = float(footprint_area) * float(room_height)

    return {
        "room_height_m": round(float(room_height), 3),
        "height_source": height_source,
        "height_confidence": height_confidence,
        "ceiling_detected": bool(ceiling_detected),
        "floor_z": round(float(floor_z), 3),
        "ceiling_z": round(float(get_plane_centroid(ceiling_plane)[2]), 3) if ceiling_plane is not None else None,
        "volume_m3": round(volume, 3) if volume is not None else None,
        "measurement_warnings": measurement_warnings,
        "height_diagnostics": {
            "voxel_points": int(len(p_vox.points)),
            "floor_slice_points": int(len(floor_slice.points)) if floor_slice is not None else 0,
            "ceiling_slice_points": int(len(ceil_slice.points)) if ceil_slice is not None else 0,
            "top_z_max": round(top_z_max, 3),
            "top_z_p99_9": round(top_z_p999, 3),
            "top_z_p99_99": round(top_z_p9999, 3),
            "top_band_points_2cm": int(top_band_2cm),
            "top_band_points_5cm": int(top_band_5cm),
            "top_band_points_10cm": int(top_band_10cm),
        },
    }


def analyze_static_height_cloud(p, footprint_area=None, source_label=None, allow_relaxed_floor=False):
    p = preprocess(o3d.geometry.PointCloud(p))
    p_aligned, align_meta = world_align(p, return_meta=True)
    report = _estimate_static_height_report(
        p_aligned,
        footprint_area=footprint_area,
        allow_relaxed_floor=allow_relaxed_floor,
    )
    if source_label is not None:
        report["height_bag"] = source_label
    report["align_tilt_deg"] = round(float(align_meta.get("tilt_deg", 0.0)), 3)
    report["align_method"] = align_meta.get("method")
    return report


def analyze_static_height_bag(path, footprint_area=None):
    report = analyze_static_height_cloud(load_file(path), footprint_area=footprint_area, source_label=path)
    return report


def analyze_loaded_clouds(clouds, input_path=None, walking_mode=False, repeat_walk_height_hint=None,
                         allow_guided_live_borderline=False, return_low_confidence_report=False,
                         suppress_exports=False):
    _skip_align = walking_mode
    walking_wall_hints = None
    walking_ceiling_hints = None
    walking_ceiling_z_hint = None
    walking_support_metrics = None
    pre_wall_hints = None
    pre_ceiling_hints = None
    pre_ceiling_z_hint = None
    pre_support_metrics = None
    if _skip_align:
        pre_wall_hints = _extract_walking_wall_hints(clouds)
        pre_ceiling_hints = _extract_walking_ceiling_hints(clouds)
        pre_ceiling_z_hint = _estimate_walking_ceiling_z(clouds)
        pre_support_metrics = _walking_support_metrics(pre_wall_hints)
        if repeat_walk_height_hint is None and input_path and _is_rosbag_path(input_path):
            repeat_walk_height_hint = _estimate_repeat_walk_height_hint(input_path)
        walking_quality = LAST_WALKING_QUALITY
    else:
        walking_quality = None

    pose_graph = full_registration(clouds, planar=_skip_align)
    transformed_clouds = _transform_point_clouds(clouds, pose_graph) if _skip_align else None
    merged = merge_clouds(clouds, pose_graph)
    semantic_cloud = None
    semantic_frames = None
    if _skip_align and LAST_WALKING_FRAME_DATA:
        # Do NOT apply a z_shift here: the semantic cloud and frames must remain
        # in the same coordinate frame as the merged cloud so that the floor_z
        # derived from the merged cloud applies correctly in _analyze_scene_exports.
        # A z_shift that moves the floor to z=0 while floor_z stays negative causes
        # floor_band and the wall low/top fill histograms to look at the wrong heights.
        semantic_cloud = _merge_walking_frame_data_cloud(
            LAST_WALKING_FRAME_DATA,
            pose_graph,
            key="semantic_pcd",
            voxel_size=0.015,
        )
        semantic_frames = _transform_walking_frame_data(
            LAST_WALKING_FRAME_DATA,
            pose_graph,
            key="semantic_pcd",
        )
    merged.estimate_normals(
        o3d.geometry.KDTreeSearchParamKNN(knn=50)
    )

    if _skip_align:
        post_wall_hints = _extract_walking_wall_hints(transformed_clouds)
        post_ceiling_hints = _extract_walking_ceiling_hints(transformed_clouds)
        post_ceiling_z_hint = _estimate_walking_ceiling_z(transformed_clouds)
        post_support_metrics = _walking_support_metrics(post_wall_hints)
        hint_source = _select_walking_hint_source(pre_wall_hints, post_wall_hints)
        if hint_source == "pre":
            walking_wall_hints = pre_wall_hints
            walking_ceiling_hints = pre_ceiling_hints
            walking_ceiling_z_hint = pre_ceiling_z_hint
            walking_support_metrics = pre_support_metrics
        else:
            walking_wall_hints = post_wall_hints
            walking_ceiling_hints = post_ceiling_hints
            walking_ceiling_z_hint = post_ceiling_z_hint
            walking_support_metrics = post_support_metrics
            if (
                pre_support_metrics is not None
                and pre_support_metrics.get("hint_count", 0) >= 12
                and pre_support_metrics.get("wall_observations", 0) >= 10
                and post_support_metrics.get("hint_count", 0) < 6
            ):
                walking_support_metrics = {
                    "hint_count": max(
                        int(post_support_metrics.get("hint_count", 0)),
                        int(pre_support_metrics.get("hint_count", 0)),
                    ),
                    "wall_observations": max(
                        int(post_support_metrics.get("wall_observations", 0)),
                        int(pre_support_metrics.get("wall_observations", 0)),
                    ),
                }

    return reconstruct(
        merged,
        skip_world_align=_skip_align,
        walking_wall_hints=walking_wall_hints,
        walking_ceiling_hints=walking_ceiling_hints,
        walking_ceiling_z_hint=walking_ceiling_z_hint,
        walking_quality=walking_quality,
        walking_support_metrics=walking_support_metrics,
        repeat_walk_height_hint=repeat_walk_height_hint,
        semantic_cloud=semantic_cloud,
        semantic_frames=semantic_frames,
        allow_guided_live_borderline=allow_guided_live_borderline,
        return_low_confidence_report=return_low_confidence_report,
        suppress_exports=suppress_exports,
    )


def analyze_input_path(input_path, multi=None, bag_mode="auto", bag_frame_interval=1.0,
                       host="192.168.1.20", port=7502, seconds=45):
    walking_mode = False
    repeat_walk_height_hint = None

    if input_path == "live":
        raise RuntimeError(
            "The legacy non-guided live path was removed. "
            "Route live input through analyze_live_guided() instead."
        )
    elif _is_rosbag_path(input_path) and not multi:
        resolved_bag_mode = bag_mode
        if resolved_bag_mode == "auto":
            resolved_bag_mode = "walking"

        if resolved_bag_mode == "walking":
            clouds = load_rosbag_frames(input_path, frame_interval_sec=bag_frame_interval)
            walking_mode = True
            repeat_walk_height_hint = _estimate_repeat_walk_height_hint(input_path)
        else:
            clouds = [preprocess(load_file(input_path))]
    else:
        clouds = [load_file(input_path)]
        if multi:
            for f in multi:
                clouds.append(load_file(f))
        clouds = [preprocess(c) for c in clouds]

    return analyze_loaded_clouds(
        clouds,
        input_path=input_path,
        walking_mode=walking_mode,
        repeat_walk_height_hint=repeat_walk_height_hint,
    )


def analyze_live_guided(host="192.168.1.20", port=7502, seconds=45,
                       frame_interval_sec=1.0):
    latest_backups = _capture_latest_artifact_backups()
    continuation_seconds = max(8, min(12, int(seconds)))

    best_attempt_frames_raw = None
    best_attempt_report = None
    best_attempt_score = None
    best_attempt_clouds = None
    best_attempt_label = None

    rejection_debug = {
        "seconds": int(seconds),
        "frame_interval_sec": float(frame_interval_sec),
        "attempts": [],
    }

    def _analyze_guided_attempt(frames_raw, *, source_name, input_path, suppress_exports=True):
        with log_level_scope("silent"):
            clouds = _prepare_walking_frames(
                frames_raw,
                source_name=source_name,
            )
            report = analyze_loaded_clouds(
                clouds,
                input_path=input_path,
                walking_mode=True,
                allow_guided_live_borderline=True,
                return_low_confidence_report=True,
                suppress_exports=suppress_exports,
            )
        return clouds, report

    def _consider_attempt(frames_raw, clouds, report):
        nonlocal best_attempt_frames_raw
        nonlocal best_attempt_report
        nonlocal best_attempt_score
        nonlocal best_attempt_clouds
        nonlocal best_attempt_label

        if report is None:
            return

        fallback_reason = _guided_footprint_fallback_accept_reason(report)
        candidate_score = (
            int((not bool(report.get("confidence_issues") or [])) or fallback_reason is not None),
        ) + _guided_footprint_capture_score(report)

        if best_attempt_score is None or candidate_score > best_attempt_score:
            best_attempt_score = candidate_score
            best_attempt_frames_raw = [frame.copy() for frame in frames_raw]
            best_attempt_report = dict(report)
            best_attempt_clouds = clouds
            best_attempt_label = report.get("_debug_attempt_label")

    def _record_attempt(label, frames_raw, report=None, error=None):
        entry = {
            "label": label,
            "kept_positions": len(frames_raw or []),
        }
        if report is not None:
            report_copy = dict(report)
            report_copy.pop("_export_tag", None)
            report_copy.pop("_debug_attempt_label", None)
            entry["report"] = report_copy
            entry["fallback_reason"] = _guided_footprint_fallback_accept_reason(report)
            entry["preview_usable"] = _guided_footprint_preview_is_usable(report)
            entry["preview_worth_saving"] = _guided_footprint_preview_is_worth_saving(report)
        if error is not None:
            entry["error"] = error
        rejection_debug["attempts"].append(entry)

    def _save_rejection_debug(reason, last_error_text, confidence_issue_list=None, fallback=None):
        rejection_debug["reason"] = reason
        rejection_debug["last_error"] = last_error_text
        rejection_debug["used_preview_snapshot"] = bool(used_preview_snapshot)
        rejection_debug["segment_frame_count"] = len(segment_frames_raw or [])
        rejection_debug["selected_frame_count"] = len(selected_segment_frames_raw or [])
        rejection_debug["best_attempt_label"] = best_attempt_label
        rejection_debug["best_attempt_score"] = list(best_attempt_score) if best_attempt_score is not None else None
        rejection_debug["final_attempt_frame_count"] = len(attempt_frames_raw or [])
        if confidence_issue_list is not None:
            rejection_debug["final_confidence_issues"] = list(confidence_issue_list)
        if fallback is not None:
            rejection_debug["final_fallback_reason"] = fallback
        save_live_guided_rejection_debug(rejection_debug)

    footprint_report = None
    footprint_clouds = None
    last_error = None
    segment = live_capture_walking_guided(
        host,
        port,
        seconds=int(seconds),
        frame_interval_sec=frame_interval_sec,
        segment_label="footprint",
        continuation_hint=None,
    )
    segment_frames_raw = segment.get("frames_raw", [])
    selected_segment_frames_raw = segment.get("selected_frames_raw") or segment_frames_raw
    used_preview_snapshot = len(selected_segment_frames_raw) < len(segment_frames_raw)
    if used_preview_snapshot:
        operator_log(
            f"[GUIDE] Using the stable preview snapshot from this perimeter ({len(selected_segment_frames_raw)} positions) instead of the longer tail ({len(segment_frames_raw)} positions)."
        )

    attempt_frames_raw = [frame.copy() for frame in selected_segment_frames_raw]
    operator_log(f"[GUIDE] Validating footprint from {len(attempt_frames_raw)} kept positions.")

    try:
        footprint_clouds, footprint_report = _analyze_guided_attempt(
            attempt_frames_raw,
            source_name="Guided live footprint",
            input_path="live_guided_footprint",
        )
        if footprint_report is not None:
            footprint_report["_debug_attempt_label"] = "preview_snapshot"
        _consider_attempt(attempt_frames_raw, footprint_clouds, footprint_report)
        _record_attempt("preview_snapshot", attempt_frames_raw, footprint_report)
    except RuntimeError as exc:
        last_error = str(exc)
        footprint_report = None
        _record_attempt("preview_snapshot", attempt_frames_raw, error=last_error)

    rescue_attempted = False
    same_lap_recovery_attempted = False
    continuation_attempted = False
    initial_confidence_issues = list(footprint_report.get("confidence_issues") or []) if footprint_report is not None else []
    initial_fallback_reason = None
    if initial_confidence_issues:
        initial_fallback_reason = _guided_footprint_fallback_accept_reason(footprint_report)

    need_full_perimeter_retry = (
        used_preview_snapshot
        and len(segment_frames_raw) > len(attempt_frames_raw)
        and (footprint_report is None or (initial_confidence_issues and initial_fallback_reason is None))
    )
    if need_full_perimeter_retry:
        rescue_attempted = True
        operator_log(
            "[GUIDE] The saved preview snapshot is still provisional, so the script will validate the full captured perimeter from the same lap before asking for more walking."
        )
        try:
            footprint_clouds, footprint_report = _analyze_guided_attempt(
                [frame.copy() for frame in segment_frames_raw],
                source_name="Guided live footprint full perimeter",
                input_path="live_guided_footprint_full_perimeter",
            )
            if footprint_report is not None:
                footprint_report["_debug_attempt_label"] = "full_perimeter"
            _consider_attempt(segment_frames_raw, footprint_clouds, footprint_report)
            _record_attempt("full_perimeter", segment_frames_raw, footprint_report)
            attempt_frames_raw = [frame.copy() for frame in segment_frames_raw]
        except RuntimeError as exc:
            last_error = str(exc)
            footprint_report = None
            _record_attempt("full_perimeter", segment_frames_raw, error=last_error)

    if best_attempt_report is not None:
        footprint_report = dict(best_attempt_report)
        footprint_clouds = best_attempt_clouds
        attempt_frames_raw = [frame.copy() for frame in (best_attempt_frames_raw or attempt_frames_raw)]

    confidence_issues = list(footprint_report.get("confidence_issues") or []) if footprint_report is not None else []

    fallback_reason = None
    if confidence_issues:
        fallback_reason = _guided_footprint_fallback_accept_reason(footprint_report)

    need_same_lap_recovery = (
        not used_preview_snapshot
        and len(segment_frames_raw) >= 10
        and (footprint_report is None or (confidence_issues and fallback_reason is None))
    )
    if need_same_lap_recovery:
        same_lap_recovery_attempted = True
        operator_log(
            "[GUIDE] The first-lap perimeter is still weak, so the script will compare a few same-lap slices before asking for more walking."
        )
        recovery_candidate = _select_guided_footprint_subwindow(segment_frames_raw)
        if recovery_candidate is None:
            _record_attempt(
                "same_lap_recovery",
                segment_frames_raw,
                error="no_recoverable_same_lap_slice",
            )
        else:
            candidate_frames_raw = recovery_candidate.get("frames_raw") or []
            candidate_report = dict(recovery_candidate.get("report") or {})
            baseline_report = best_attempt_report or footprint_report
            if _guided_footprint_rescue_is_collapsed(candidate_report, baseline_report):
                operator_log(
                    "[GUIDE] Ignoring a collapsed same-lap recovery slice because it shrank too far inside the captured perimeter."
                )
                _record_attempt(
                    "same_lap_recovery",
                    candidate_frames_raw,
                    candidate_report,
                    error="collapsed_same_lap_slice",
                )
            else:
                candidate_report["_debug_attempt_label"] = "same_lap_recovery"
                footprint_clouds = recovery_candidate.get("footprint_clouds")
                footprint_report = candidate_report
                _consider_attempt(candidate_frames_raw, footprint_clouds, footprint_report)
                _record_attempt("same_lap_recovery", candidate_frames_raw, footprint_report)
                attempt_frames_raw = [frame.copy() for frame in candidate_frames_raw]

        if best_attempt_report is not None:
            footprint_report = dict(best_attempt_report)
            footprint_clouds = best_attempt_clouds
            attempt_frames_raw = [frame.copy() for frame in (best_attempt_frames_raw or attempt_frames_raw)]

        confidence_issues = list(footprint_report.get("confidence_issues") or []) if footprint_report is not None else []
        fallback_reason = None
        if confidence_issues:
            fallback_reason = _guided_footprint_fallback_accept_reason(footprint_report)

    need_continuation = footprint_report is None or (confidence_issues and fallback_reason is None)
    if need_continuation:
        continuation_attempted = True
        continuation_hint = (
            _guided_footprint_continuation_hint(footprint_report)
            if footprint_report is not None
            else "The first lap did not validate cleanly. Do one more short slow lap and pause again at each corner."
        )
        operator_log(
            "[GUIDE] The first lap is still weak, so the script will guide one short continuation before rejecting the run."
        )
        continuation_segment = live_capture_walking_guided(
            host,
            port,
            seconds=continuation_seconds,
            frame_interval_sec=frame_interval_sec,
            segment_label="continuation",
            continuation_hint=continuation_hint,
        )
        continuation_frames_raw = (
            continuation_segment.get("selected_frames_raw")
            or continuation_segment.get("frames_raw")
            or []
        )
        combined_frames_raw = [frame.copy() for frame in segment_frames_raw]
        combined_frames_raw.extend(frame.copy() for frame in continuation_frames_raw)
        operator_log(
            f"[GUIDE] Re-validating footprint from {len(combined_frames_raw)} kept positions after the short continuation."
        )
        try:
            footprint_clouds, footprint_report = _analyze_guided_attempt(
                combined_frames_raw,
                source_name="Guided live footprint with continuation",
                input_path="live_guided_footprint_with_continuation",
            )
            if footprint_report is not None:
                footprint_report["_debug_attempt_label"] = "short_continuation"
            _consider_attempt(combined_frames_raw, footprint_clouds, footprint_report)
            _record_attempt("short_continuation", combined_frames_raw, footprint_report)
            attempt_frames_raw = combined_frames_raw
        except RuntimeError as exc:
            last_error = str(exc)
            footprint_report = None
            _record_attempt("short_continuation", combined_frames_raw, error=last_error)

        if best_attempt_report is not None:
            footprint_report = dict(best_attempt_report)
            footprint_clouds = best_attempt_clouds
            attempt_frames_raw = [frame.copy() for frame in (best_attempt_frames_raw or attempt_frames_raw)]

        confidence_issues = list(footprint_report.get("confidence_issues") or []) if footprint_report is not None else []
        fallback_reason = None
        if confidence_issues:
            fallback_reason = _guided_footprint_fallback_accept_reason(footprint_report)

    if footprint_report is None:
        _save_rejection_debug("no_accepted_attempt", last_error)
        raise RuntimeError(
            "Live guidance could not confirm a stable footprint from this perimeter. "
            "Please do one more short slow lap and pause again at each corner.\n"
            f"{last_error}".rstrip()
        )

    if confidence_issues:
        last_error = (
            "Low-confidence walking measurement rejected:\n  • "
            + "\n  • ".join(confidence_issues)
            + "\nPlease repeat the scan or record one fixed bag for offline tuning."
        )
        if fallback_reason is None:
            _save_rejection_debug(
                "low_confidence_after_continuation",
                last_error,
                confidence_issue_list=confidence_issues,
                fallback=fallback_reason,
            )
            raise RuntimeError(
                "Live guidance still needs another lap before it can lock the footprint.\n"
                f"{last_error}\n"
                f"{_guided_footprint_continuation_hint(footprint_report)}"
            )

        operator_log(f"[GUIDE] Footprint fallback accepted: {fallback_reason}.")
    else:
        if rescue_attempted:
            operator_log("[GUIDE] Full-perimeter retry succeeded; continuing with the strongest candidate from this capture.")
        if same_lap_recovery_attempted:
            if len(attempt_frames_raw) < len(segment_frames_raw):
                operator_log("[GUIDE] Same-lap recovery succeeded; the script kept the strongest slice from this perimeter.")
            else:
                operator_log("[GUIDE] Same-lap recovery did not beat the full perimeter, so the script kept the stronger original candidate.")
        if continuation_attempted:
            if len(attempt_frames_raw) > len(segment_frames_raw):
                operator_log("[GUIDE] Short continuation succeeded; the added wall support was enough to lock the footprint.")
            else:
                operator_log("[GUIDE] Short continuation did not improve the saved footprint candidate, so the script kept the stronger earlier layout.")
        operator_log(
            f"[GUIDE] Footprint locked: area {footprint_report.get('floor_area_m2')} m², "
            f"perimeter {footprint_report.get('floor_perimeter_m')} m."
        )

    accepted_footprint_report = dict(footprint_report)

    # Internal live validation attempts should not write timestamped scene
    # artifacts. Export the layout only once, from the final accepted frame set.
    footprint_clouds, footprint_report = _analyze_guided_attempt(
        attempt_frames_raw,
        source_name="Guided live footprint final export",
        input_path="live_guided_footprint_final",
        suppress_exports=False,
    )
    _record_attempt("final_export", attempt_frames_raw, footprint_report)

    export_confidence_issues = list(footprint_report.get("confidence_issues") or []) if footprint_report is not None else []
    export_fallback_reason = None
    if export_confidence_issues:
        export_fallback_reason = _guided_footprint_fallback_accept_reason(footprint_report)

    export_regressed = footprint_report is None
    if not export_regressed and export_confidence_issues and export_fallback_reason is None:
        export_regressed = True
    if not export_regressed:
        _export_score = _guided_footprint_capture_score(footprint_report)
        _accepted_score = _guided_footprint_capture_score(accepted_footprint_report)
        # Quantize floor_points (position 4) to 100-pt buckets so that trivial
        # non-deterministic voxelisation noise (±5 pts) between the validation
        # pass and the final export pass does not trigger a false regression.
        _export_score = _export_score[:4] + (_export_score[4] // 100,) + _export_score[5:]
        _accepted_score = _accepted_score[:4] + (_accepted_score[4] // 100,) + _accepted_score[5:]
        if _export_score < _accepted_score:
            export_regressed = True

    if export_regressed:
        restored = _restore_latest_artifact_backups(latest_backups)
        _save_rejection_debug(
            "final_export_regressed",
            "Final export reconstruction regressed after the accepted perimeter selection.",
            confidence_issue_list=export_confidence_issues,
            fallback=export_fallback_reason,
        )
        raise RuntimeError(
            "Live guidance found an acceptable perimeter candidate, but the final export reconstruction regressed it.\n"
            "The script restored the previous *_latest artifacts instead of publishing a broken result.\n"
            + (
                "Low-confidence export details:\n  • " + "\n  • ".join(export_confidence_issues)
                if export_confidence_issues else
                "Final export did not produce a usable report."
            )
            + (f"\nRestored latest artifacts: {restored}." if restored else "")
        )

    footprint_height_hint, footprint_height_hint_source = _extract_guided_footprint_height_hint(
        footprint_clouds or [],
        footprint_report,
    )
    if footprint_height_hint is None:
        footprint_height_hint = footprint_report.get("room_height_m")
        footprint_height_hint_source = footprint_report.get("height_source") or "guided_footprint_report"
    else:
        operator_log(
            f"[GUIDE] Footprint ceiling hint: {float(footprint_height_hint):.3f} m via {footprint_height_hint_source}."
        )
    report = dict(footprint_report)
    report["footprint_input"] = "live_guided_footprint"
    report["height_input"] = "live_guided_footprint"
    report["height_input_mode"] = "live-guided-perimeter"
    report["footprint_room_height_m"] = report.get("room_height_m")
    report["footprint_volume_m3"] = report.get("volume_m3")

    if footprint_height_hint is not None and float(footprint_height_hint) >= 1.8:
        report["room_height_m"] = round(float(footprint_height_hint), 3)
        report["height_source"] = footprint_height_hint_source or report.get("height_source")

    area = report.get("floor_area_m2")
    height = report.get("room_height_m")
    if area is not None and height is not None:
        report["volume_m3"] = round(float(area) * float(height), 3)

    warnings = list(report.get("measurement_warnings") or [])
    perimeter_warning = (
        "Guided live completed from the perimeter walk only; room height comes from walking wall/cloud-top evidence."
    )
    if perimeter_warning not in warnings:
        warnings.append(perimeter_warning)
    report["measurement_warnings"] = warnings

    export_tag = report.get("_export_tag")
    if export_tag:
        update_dxf_summary(
            export_tag,
            floor_area=report.get("floor_area_m2"),
            perimeter=report.get("floor_perimeter_m"),
            room_height=report.get("room_height_m"),
        )

    report["live_mode"] = "guided"
    operator_log(
        f"[GUIDE] Live measurement complete: area {report.get('floor_area_m2')} m², "
        f"perimeter {report.get('floor_perimeter_m')} m, height {report.get('room_height_m')} m."
    )

    if (
        str(report.get("footprint_confidence") or "") == "low"
        and not _guided_footprint_preview_is_usable(report)
    ):
        restored = _restore_latest_artifact_backups(latest_backups)
        if restored > 0:
            operator_log(
                "[GUIDE] This run finished with low footprint confidence; preserving prior *_latest artifacts "
                "and keeping this run only in timestamped outputs."
            )
    else:
        _cleanup_latest_artifact_backups(latest_backups)

    return report


def _merge_footprint_and_height_reports(footprint_report, height_report, footprint_input, height_input, height_mode):
    report = dict(footprint_report)

    footprint_height = report.get("room_height_m")
    footprint_volume = report.get("volume_m3")
    report["footprint_input"] = footprint_input
    report["height_input"] = height_input
    report["height_input_mode"] = height_mode
    report["footprint_room_height_m"] = footprint_height
    report["footprint_volume_m3"] = footprint_volume
    report["room_height_m"] = height_report.get("room_height_m")
    report["height_source"] = height_report.get("height_source")
    report["height_confidence"] = height_report.get("height_confidence")
    report["ceiling_detected"] = height_report.get("ceiling_detected", report.get("ceiling_detected"))
    if height_report.get("floor_z") is not None:
        report["height_floor_z"] = height_report.get("floor_z")
    if height_report.get("ceiling_z") is not None:
        report["height_ceiling_z"] = height_report.get("ceiling_z")
    if height_report.get("height_diagnostics") is not None:
        report["height_diagnostics"] = height_report.get("height_diagnostics")
    if height_report.get("align_tilt_deg") is not None:
        report["height_align_tilt_deg"] = height_report.get("align_tilt_deg")
    if height_report.get("align_method") is not None:
        report["height_align_method"] = height_report.get("align_method")

    area = report.get("floor_area_m2")
    height = report.get("room_height_m")
    if area is not None and height is not None:
        report["volume_m3"] = round(float(area) * float(height), 3)
    else:
        report["volume_m3"] = height_report.get("volume_m3")

    warnings = []
    seen = set()
    for warning in (footprint_report.get("measurement_warnings") or []):
        if warning not in seen:
            warnings.append(warning)
            seen.add(warning)
    bridge_warning = "Room height and volume were taken from a separate height bag."
    warnings.append(bridge_warning)
    seen.add(bridge_warning)
    for warning in (height_report.get("measurement_warnings") or []):
        if warning not in seen:
            warnings.append(warning)
            seen.add(warning)
    report["measurement_warnings"] = warnings

    return report


def _merge_coplanar(planes, normal_thresh=0.93, offset_thresh=0.12):
    """
    Merge plane detections that represent the same physical surface.

    RANSAC extracts planes iteratively; a large flat surface (floor, long wall)
    is often split into multiple sub-patches across iterations.  Two planes are
    considered the same surface when:
      - their normals are nearly parallel  (|n1·n2| > normal_thresh), AND
      - their plane offsets d = -n·c agree within offset_thresh metres.

    offset_thresh is automatically extended for large planes: two sub-patches
    of the same long wall can have centroids > 1 m apart even when they are
    truly coplanar.  We scale the threshold by the mean plane half-diagonal.
    """
    if len(planes) <= 1:
        return planes

    def _plane_sort_key(plane):
        normal = np.array(plane["normal"], dtype=float)
        normal /= np.linalg.norm(normal) + 1e-9
        centroid = np.array(plane["centroid"], dtype=float)
        area = float(plane["area"])
        return (
            -round(area, 4),
            round(float(normal[0]), 4),
            round(float(normal[1]), 4),
            round(float(normal[2]), 4),
            round(float(centroid[0]), 3),
            round(float(centroid[1]), 3),
            round(float(centroid[2]), 3),
        )

    def _plane_extent(plane):
        """Half-diagonal of the inlier bounding box (metres)."""
        sc = plane.get('shell_coords')
        if sc is None or len(sc) < 2:
            return 0.0
        span = sc.max(axis=0) - sc.min(axis=0)
        return float(np.linalg.norm(span)) * 0.5

    used   = [False] * len(planes)
    merged = []

    for i, pi in enumerate(planes):
        if used[i]:
            continue
        ni = pi['normal'] / np.linalg.norm(pi['normal'])
        di = -float(np.dot(ni, pi['centroid']))
        ei = _plane_extent(pi)

        group = [pi]
        used[i] = True

        for j in range(i + 1, len(planes)):
            if used[j]:
                continue
            nj = planes[j]['normal'] / np.linalg.norm(planes[j]['normal'])
            dj = -float(np.dot(nj, planes[j]['centroid']))
            ej = _plane_extent(planes[j])

            # Scale offset tolerance by average plane size, minimum offset_thresh
            adaptive_thresh = max(offset_thresh, 0.04 * (ei + ej))

            dot_ij = float(np.dot(ni, nj))

            if dot_ij > normal_thresh:
                # Same-direction normals: same surface iff offsets match
                if abs(di - dj) < adaptive_thresh:
                    group.append(planes[j])
                    used[j] = True
            elif dot_ij < -normal_thresh:
                # Antiparallel normals: same surface iff d1 + d2 ≈ 0
                if abs(di + dj) < adaptive_thresh:
                    group.append(planes[j])
                    used[j] = True

        if len(group) == 1:
            merged.append(pi)
        else:
            all_pts = np.vstack([p['shell_coords'] for p in group])
            # Weighted-average normal (by inlier count)
            weights = np.array([len(p['shell_coords']) for p in group], dtype=float)
            normal  = np.average([p['normal'] for p in group], axis=0, weights=weights)
            normal /= np.linalg.norm(normal)
            if normal[2] < 0:
                normal = -normal
            centroid = all_pts.mean(axis=0)
            area     = _plane_area(all_pts, normal)
            log(
                f"[DEBUG]   Merged {len(group)} coplanar patches → "
                f"area={area:.2f} m²  inliers={len(all_pts)}",
                "debug",
            )
            merged.append({
                'shell_coords': all_pts,
                'normal'      : normal,
                'centroid'    : centroid,
                'area'        : area,
            })

    return sorted(merged, key=_plane_sort_key)


def _merge_wall_faces(walls, azimuth_thresh_deg=20.0, d_thresh=0.25):
    """
    Second-pass wall deduplication: merge walls that are sub-patches of the
    same physical face (e.g. a single wall split by residual sensor tilt).

    Two walls are treated as the same face when:
      1. Their XY-projected normals point in the same direction (within
         azimuth_thresh_deg), AND
      2. Their XY-only plane offsets  d_xy = -n_xy · c_xy  agree within
         d_thresh metres.

    Using the XY-projected offset (ignoring Z) makes the comparison
    height-independent: two sub-patches of the same wall recorded at
    different scan heights produce the same d_xy even when their 3-D
    d values differ due to residual tilt.

    Opposite walls have the same XY azimuth but very different d_xy values
    (separated by the room width), so they are kept separate.
    """
    if len(walls) <= 1:
        return walls

    def _wall_sort_key(w):
        n = get_plane_normal(w)
        n_xy = np.array(n[:2], dtype=float)
        n_xy_norm = float(np.linalg.norm(n_xy))
        if n_xy_norm < 1e-6:
            return (999.0, 999.0, 0.0)
        n_xy /= n_xy_norm
        az = float(np.arctan2(n_xy[1], n_xy[0])) % np.pi
        c_xy = get_plane_centroid(w)[:2]
        d_xy = -float(np.dot(n_xy, c_xy))
        area = float(get_plane_area(w))
        return (round(az, 4), round(d_xy, 3), -round(area, 3))

    walls = sorted(walls, key=_wall_sort_key)

    az_thresh = np.radians(azimuth_thresh_deg)
    used   = [False] * len(walls)
    result = []

    for i, wi in enumerate(walls):
        if used[i]:
            continue
        ni     = get_plane_normal(wi)
        ni_xy  = ni[:2].copy()
        nxy_n  = np.linalg.norm(ni_xy)
        if nxy_n < 1e-6:
            result.append(wi)
            used[i] = True
            continue
        ni_xy /= nxy_n
        ci_xy  = get_plane_centroid(wi)[:2]
        di     = -float(np.dot(ni_xy, ci_xy))   # XY-only offset

        group  = [wi]
        used[i] = True

        for j in range(i + 1, len(walls)):
            if used[j]:
                continue
            nj    = get_plane_normal(walls[j])
            nj_xy = nj[:2].copy()
            nxy_nj = np.linalg.norm(nj_xy)
            if nxy_nj < 1e-6:
                continue
            nj_xy /= nxy_nj
            cj_xy  = get_plane_centroid(walls[j])[:2]
            dj     = -float(np.dot(nj_xy, cj_xy))  # XY-only offset

            # Check XY-azimuth agreement (fold antiparallel → same half-circle)
            dot_xy = float(np.dot(ni_xy, nj_xy))
            if dot_xy < np.cos(az_thresh):
                continue  # different azimuth — different wall direction

            # Same azimuth AND same XY face distance → same physical face
            if abs(di - dj) < d_thresh:
                group.append(walls[j])
                used[j] = True

        if len(group) == 1:
            result.append(wi)
        else:
            all_pts = np.vstack([w['shell_coords'] for w in group])
            weights = np.array([len(w['shell_coords']) for w in group], dtype=float)
            normal  = np.average(
                [get_plane_normal(w) for w in group], axis=0, weights=weights
            )
            normal /= np.linalg.norm(normal)
            if normal[2] < 0:
                normal = -normal
            centroid = all_pts.mean(axis=0)
            area     = _plane_area(all_pts, normal)
            log(
                f"[DEBUG]   Wall-dedup: merged {len(group)} same-face planes "
                f"→ area={area:.2f} m²",
                "debug",
            )
            result.append({
                'shell_coords': all_pts,
                'normal'      : normal,
                'centroid'    : centroid,
                'area'        : area,
            })

    if len(result) < len(walls):
        log(
            f"[DEBUG] Wall-dedup: {len(walls)} walls → {len(result)} after "
            f"face-merge ({azimuth_thresh_deg:.0f}° az, {d_thresh:.2f} m d-tol)",
            "debug",
        )
    return sorted(result, key=_wall_sort_key)

def get_plane_normal(p):
    """Extract plane normal - works with dict or object with attributes"""
    if isinstance(p, dict) and 'normal' in p:
        return p['normal']
    elif hasattr(p, 'normal'):
        return np.array(p.normal)
    elif hasattr(p, 'exterior'):
        # For Shapely polygons, compute normal from exterior
        coords = np.array(p.exterior.coords[:-1])  # Remove duplicate end point
        if len(coords) >= 3:
            v1 = coords[1] - coords[0]
            v2 = coords[2] - coords[0]
            n = np.cross(v1, v2)
            return n / (np.linalg.norm(n) + 1e-8)
    return np.array([0, 0, 1])

def get_plane_centroid(p):
    """Extract plane centroid - works with dict or object with attributes"""
    if isinstance(p, dict) and 'centroid' in p:
        return p['centroid']
    elif hasattr(p, 'centroid'):
        return np.array(p.centroid)
    elif hasattr(p, 'exterior'):
        return np.array(p.exterior.coords).mean(axis=0)
    return np.array([0, 0, 0])

def get_plane_area(p):
    """Extract plane area - works with dict or object with attributes"""
    if isinstance(p, dict) and 'area' in p:
        return p['area']
    elif hasattr(p, 'area'):
        return p.area
    elif hasattr(p, 'exterior'):
        from shapely.geometry import Polygon as ShapelyPolygon
        coords = np.array(p.exterior.coords)
        if len(coords) > 2:
            return ShapelyPolygon(coords).area
    return 0


def get_plane_floor_z(p, percentile=5):
    """Robust lower-Z estimate for a plane using its inlier points when available."""
    if isinstance(p, dict) and 'shell_coords' in p:
        coords = p['shell_coords']
        if coords is not None and len(coords):
            return float(np.percentile(coords[:, 2], percentile))
    return float(get_plane_centroid(p)[2])


def _angdiff_pi(a, b):
    """Smallest angular difference on [0, π)."""
    d = abs(a - b)
    return min(d, np.pi - d)


def _count_distinct_wall_dirs(ws, tol_deg=15.0):
    """Count distinct wall azimuth groups, folding antiparallel directions."""
    if len(ws) == 0:
        return 0
    if len(ws) == 1:
        return 1
    azs = sorted(
        float(np.arctan2(get_plane_normal(w)[1], get_plane_normal(w)[0])) % np.pi
        for w in ws
    )
    tol = np.radians(tol_deg)
    n_dir = 1
    for i in range(1, len(azs)):
        if abs(azs[i] - azs[i - 1]) > tol:
            n_dir += 1
    return n_dir


def _extract_walking_wall_hints(pcds, summary_log_level="info"):
    """Extract wall observations per walking frame, then merge globally.

    The merged walking cloud has proven unstable for wall extraction because it
    overweights horizontal clutter.  This helper instead mines wall candidates
    from each already-gravity-aligned frame, where walls are simpler and less
    entangled, then merges those candidates in global coordinates.
    """
    hints = []
    min_pts = CONFIG.get("min_plane_points", 100)
    wall_thresh = CONFIG.get('wall_z_comp_threshold', 0.25)
    abs_floor = CONFIG.get('min_wall_abs_z_span', 0.4)
    fov_frac = CONFIG.get('min_wall_fov_fraction', 0.65)

    def _qualify_frame_wall(plane):
        n = get_plane_normal(plane)
        n = n / (np.linalg.norm(n) + 1e-9)
        z_component = abs(float(n[2]))
        if z_component > wall_thresh:
            return None

        coords = plane.get('shell_coords') if isinstance(plane, dict) else None
        if coords is None or len(coords) < min_pts:
            return None
        z_span = float(coords[:, 2].max() - coords[:, 2].min())
        c_xy = get_plane_centroid(plane)[:2]
        d_xy = float(np.linalg.norm(c_xy))
        min_span = max(abs_floor, d_xy * fov_frac)
        if z_span < min_span:
            return None
        return plane

    for i, pcd in enumerate(pcds):
        if len(pcd.points) < min_pts:
            continue
        if not pcd.has_normals():
            pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=30))

        pts = np.asarray(pcd.points)
        nrms = np.asarray(pcd.normals)
        z_hi = max(1.7, float(np.percentile(pts[:, 2], 98)) - 0.10)
        mask = (
            (pts[:, 2] >= 0.15) &
            (pts[:, 2] <= z_hi) &
            (np.abs(nrms[:, 2]) <= 0.35)
        )
        idx = np.where(mask)[0]
        if len(idx) < min_pts:
            continue

        wall_slice = pcd.select_by_index(idx.tolist())
        wall_slice.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=30))
        planes = extract_planes(wall_slice)

        frame_walls = []
        for p in planes:
            qualified = _qualify_frame_wall(p)
            if qualified is not None:
                frame_walls.append(qualified)

        if len(frame_walls) < 2 or _count_distinct_wall_dirs(frame_walls) < 2:
            cluster_walls = _walls_from_normal_clusters(
                wall_slice,
                min_face_points=max(60, min_pts // 2),
                az_tol_deg=18.0,
                log_level="debug",
            )
            cluster_walls = [
                wall for wall in (_qualify_frame_wall(p) for p in cluster_walls)
                if wall is not None
            ]
            if cluster_walls:
                frame_walls = _merge_wall_faces(frame_walls + cluster_walls)
                log(
                    f"[DEBUG] Walking wall hints: frame {i+1}/{len(pcds)} recovered "
                    f"{len(cluster_walls)} wall(s) from point-normal clusters",
                    "debug",
                )

        if frame_walls:
            frame_walls = _merge_wall_faces(frame_walls)
            hints.extend(frame_walls)
            log(
                f"[DEBUG] Walking wall hints: frame {i+1}/{len(pcds)} contributed "
                f"{len(frame_walls)} wall(s)",
                "debug",
            )

    if hints:
        hints = _merge_wall_faces(hints)
        log(f"[INFO] Walking wall hints: merged to {len(hints)} wall candidate(s)", summary_log_level)
    return hints


def _extract_walking_ceiling_hints(pcds):
    """Extract ceiling observations per walking frame, then merge globally."""
    hints = []
    min_pts = max(60, CONFIG.get("min_plane_points", 100) // 2)

    def _qualify_frame_ceiling(plane, z_top):
        n = get_plane_normal(plane)
        n = n / (np.linalg.norm(n) + 1e-9)
        if abs(float(n[2])) < 0.75:
            return None
        coords = plane.get('shell_coords') if isinstance(plane, dict) else None
        if coords is None or len(coords) < min_pts:
            return None
        z_med = float(np.median(coords[:, 2]))
        if z_med < 1.7:
            return None
        if z_med < z_top - 0.45:
            return None
        return plane

    for i, pcd in enumerate(pcds):
        if len(pcd.points) < min_pts:
            continue
        if not pcd.has_normals():
            pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=30))

        pts = np.asarray(pcd.points)
        nrms = np.asarray(pcd.normals)
        z_top = float(np.percentile(pts[:, 2], 98))
        z_lo = max(1.5, z_top - 0.85)
        mask = (
            (pts[:, 2] >= z_lo) &
            (np.abs(nrms[:, 2]) >= 0.70)
        )
        idx = np.where(mask)[0]
        if len(idx) < min_pts:
            continue

        ceil_slice = pcd.select_by_index(idx.tolist())
        ceil_slice.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=30))
        planes = extract_planes(ceil_slice)

        frame_ceils = []
        for p in planes:
            qualified = _qualify_frame_ceiling(p, z_top)
            if qualified is not None:
                frame_ceils.append(qualified)

        if not frame_ceils:
            fallback_ceil = _make_horizontal_plane_from_slice(
                ceil_slice,
                z_percentile=95,
                min_points=min_pts,
            )
            qualified = _qualify_frame_ceiling(fallback_ceil, z_top) if fallback_ceil is not None else None
            if qualified is not None:
                frame_ceils.append(qualified)
                log(
                    f"[DEBUG] Walking ceiling hints: frame {i+1}/{len(pcds)} synthesised "
                    f"a ceiling patch from the top horizontal band",
                    "debug",
                )

        if frame_ceils:
            frame_ceils = _merge_coplanar(frame_ceils, normal_thresh=0.97, offset_thresh=0.18)
            hints.extend(frame_ceils)
            log(
                f"[DEBUG] Walking ceiling hints: frame {i+1}/{len(pcds)} contributed "
                f"{len(frame_ceils)} ceiling patch(es)",
                "debug",
            )

    if hints:
        hints = _merge_coplanar(hints, normal_thresh=0.97, offset_thresh=0.22)
        log(f"[INFO] Walking ceiling hints: merged to {len(hints)} ceiling candidate(s)", "info")
    return hints


def _estimate_walking_ceiling_z(pcds):
    """Robust ceiling-Z hint from per-frame top horizontal returns."""
    z_hints = []
    for pcd in pcds:
        if len(pcd.points) < 100:
            continue
        if not pcd.has_normals():
            pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=30))
        pts = np.asarray(pcd.points)
        nrms = np.asarray(pcd.normals)

        # Ceiling evidence can be sparse in a walking scan. Be permissive here:
        # mine the top ~20 % of each frame and accept moderately horizontal
        # normals, then take a high percentile of the surviving Z values.
        z_cut = float(np.percentile(pts[:, 2], 80))
        mask = (pts[:, 2] >= z_cut) & (np.abs(nrms[:, 2]) >= 0.55)
        if mask.sum() < 20:
            continue
        z_hints.append(float(np.percentile(pts[mask, 2], 98)))

    if not z_hints:
        log("[INFO] Walking ceiling Z hint: unavailable (too few top horizontal returns)", "info")
        return None

    z_hint = float(np.percentile(np.array(z_hints), 90))
    log(f"[INFO] Walking ceiling Z hint: {z_hint:.2f} m from per-frame top returns", "info")
    return z_hint


def _estimate_repeat_walk_height_hint(bag_path):
    """Estimate room height from sibling repeated walking bags of the same room.

    This is a conservative repeat-set helper for cases where a single walk bag
    does not recover enough ceiling evidence on its own. It scans sibling bag
    directories matching the same ``*_walkN`` prefix and uses the strongest
    robust per-frame floor-to-top span found across that repeated set.
    """
    bag_dir = Path(bag_path)
    name = bag_dir.name
    marker = "_walk"
    if marker not in name:
        return None
    prefix, suffix = name.rsplit(marker, 1)
    if not suffix.isdigit():
        return None

    parent = bag_dir.parent
    sibling_dirs = sorted(
        p for p in parent.iterdir()
        if p.is_dir() and p.name.startswith(prefix + marker) and p.name[len(prefix + marker):].isdigit()
    )
    if len(sibling_dirs) < 2:
        return None

    bag_height_hints = []
    for sibling in sibling_dirs:
        try:
            clouds = load_rosbag_frames(str(sibling), frame_interval_sec=1.0)
        except Exception:
            continue

        spans = []
        for pcd in clouds:
            pts = np.asarray(pcd.points)
            if len(pts) < 100:
                continue
            floor_z = float(np.percentile(pts[:, 2], 1))
            top_z = float(np.percentile(pts[:, 2], 99.9))
            span = top_z - floor_z
            if span >= 1.8:
                spans.append(span)
        if len(spans) >= 3:
            bag_height_hints.append(float(np.percentile(np.array(spans), 95)))

    if not bag_height_hints:
        return None

    hint = float(max(bag_height_hints))
    log(
        f"[INFO] Repeat-walk height hint: {hint:.2f} m from {len(bag_height_hints)} sibling walk bag(s)",
        "info",
    )
    return hint


def _walls_from_normal_clusters(pcd, min_face_points=80, az_tol_deg=15.0, log_level="info"):
    """Recover wall faces directly from point-normal azimuth clusters.

    Used as a walking-mode fallback when plane RANSAC finds too few wall
    directions.  It works on the already-filtered wall slice and clusters local
    XY normal directions instead of segmenting arbitrary 3-D planes.
    """
    if pcd is None or len(pcd.points) < min_face_points:
        return []

    if not pcd.has_normals():
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=30))

    pts = np.asarray(pcd.points)
    nrms = np.asarray(pcd.normals)
    nxy_len = np.linalg.norm(nrms[:, :2], axis=1)
    keep = (nxy_len > 0.25) & (np.abs(nrms[:, 2]) <= 0.45)
    if keep.sum() < min_face_points:
        return []

    pts = pts[keep]
    nxy = nrms[keep, :2] / nxy_len[keep][:, None]
    az  = np.mod(np.arctan2(nxy[:, 1], nxy[:, 0]), np.pi)

    bins = 72
    hist, edges = np.histogram(az, bins=bins, range=(0.0, np.pi))
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Pick up to two strong, roughly orthogonal azimuth peaks.
    selected = []
    best_score = 0
    for i in range(bins):
        for j in range(i + 1, bins):
            sep = _angdiff_pi(centers[i], centers[j])
            if not (np.radians(55.0) <= sep <= np.radians(125.0)):
                continue
            score = int(hist[i]) + int(hist[j])
            if score > best_score:
                best_score = score
                selected = [i, j]
    if not selected:
        top = np.argsort(hist)[::-1]
        for idx in top[:2]:
            if hist[idx] >= max(min_face_points, int(0.03 * len(az))):
                selected.append(int(idx))

    walls = []
    az_tol = np.radians(az_tol_deg)
    for idx in selected:
        target_az = float(centers[idx])
        group_mask = np.array([_angdiff_pi(a, target_az) <= az_tol for a in az])
        if group_mask.sum() < min_face_points:
            continue

        pts_g = pts[group_mask]
        nxy_g = nxy[group_mask].copy()

        ref = np.array([np.cos(target_az), np.sin(target_az)])
        flip = (nxy_g @ ref) < 0
        nxy_g[flip] *= -1.0
        n_axis = nxy_g.mean(axis=0)
        n_axis /= (np.linalg.norm(n_axis) + 1e-9)

        s = pts_g[:, :2] @ n_axis
        p15, p50, p85 = np.percentile(s, [15, 50, 85])
        spread = float(p85 - p15)

        face_masks = []
        if spread > 0.8:
            lo_mask = s <= p15
            hi_mask = s >= p85
            if lo_mask.sum() >= min_face_points:
                face_masks.append(lo_mask)
            if hi_mask.sum() >= min_face_points:
                face_masks.append(hi_mask)
        else:
            face_masks.append(np.ones(len(s), dtype=bool))

        for fm in face_masks:
            coords = pts_g[fm]
            if len(coords) < min_face_points:
                continue
            z_span = float(coords[:, 2].max() - coords[:, 2].min())
            if z_span < 0.8:
                continue

            s_face = float(np.median(coords[:, :2] @ n_axis))
            n_face = n_axis.copy() if s_face >= 0 else -n_axis.copy()
            normal = np.array([n_face[0], n_face[1], 0.0])
            area = _plane_area(coords, normal)
            walls.append({
                'shell_coords': coords,
                'normal': normal,
                'centroid': coords.mean(axis=0),
                'area': area,
            })

    if walls:
        walls = _merge_wall_faces(walls)
        log(f"[INFO] Point-normal recovery produced {len(walls)} wall candidate(s)", log_level)
    return walls


# =========================================================
# CLASSIFY
# =========================================================

def classify(planes):

    floors=[]
    walls=[]

    for i, p in enumerate(planes):
        n=get_plane_normal(p)
        n=n/np.linalg.norm(n)

        z_component = abs(n[2])
        log(f"[INFO] Plane {i}: z_comp={z_component:.3f}  normal=[{n[0]:.2f},{n[1]:.2f},{n[2]:.2f}]", "info")

        floor_thresh = CONFIG.get('floor_z_comp_threshold', 0.85)
        wall_thresh  = CONFIG.get('wall_z_comp_threshold',  0.25)

        if z_component > floor_thresh:
            floors.append(p)
            log(f"[INFO]   → FLOOR", "info")
        elif z_component > wall_thresh:
            # Dead-zone: too tilted to be a true wall, too steep to be a floor.
            # These are typically furniture tops, ramps or scan artefacts.
            log(
                f"[INFO]   → dead-zone (z={z_component:.3f} in {wall_thresh:.2f}–{floor_thresh:.2f})",
                "info",
            )
        else:
            # Vertical plane — check Z span to reject furniture faces.
            coords = p.get('shell_coords') if isinstance(p, dict) else None
            if coords is not None and len(coords):
                z_span = float(coords[:, 2].max() - coords[:, 2].min())
            else:
                z_span = None

            # Minimum Z span scales with the wall's distance from the sensor.
            # A wall 0.75 m away has much less vertical coverage visible than
            # a wall 3 m away (limited by the sensor's vertical FOV).
            # Formula:  min_span = max(abs_floor, dist_xy × fov_fraction)
            c_xy = get_plane_centroid(p)[:2]
            d_xy = float(np.linalg.norm(c_xy))
            abs_floor  = CONFIG.get('min_wall_abs_z_span',  0.4)
            fov_frac   = CONFIG.get('min_wall_fov_fraction', 0.65)
            min_span   = max(abs_floor, d_xy * fov_frac)
            if z_span is not None and z_span < min_span:
                log(
                    f"[INFO]   → rejected (Z span {z_span:.2f} m < min {min_span:.2f} m, dist {d_xy:.2f} m)",
                    "info",
                )
            else:
                walls.append(p)
                log(
                    f"[INFO]   → WALL (Z span {z_span:.2f} m, dist {d_xy:.2f} m)" if z_span is not None else
                    f"[INFO]   → WALL",
                    "info",
                )

    log(f"[INFO] Classification: {len(floors)} floors, {len(walls)} walls", "info")
    return floors,walls


# =========================================================
# INTERSECTION
# =========================================================

def intersect(p1, p2, p3):
    """Return 3-plane intersection point, or None if planes are degenerate."""
    n1 = get_plane_normal(p1)
    n2 = get_plane_normal(p2)
    n3 = get_plane_normal(p3)

    # Skip nearly-parallel wall pairs (|cos θ| > 0.95  ≈  angle < 18°).
    # Two opposite walls of a rectangular room are parallel; their intersection
    # with the floor sits at infinity and produces huge spurious coordinates.
    cos_angle = abs(float(np.dot(n1 / np.linalg.norm(n1),
                                 n2 / np.linalg.norm(n2))))
    if cos_angle > 0.95:
        return None

    d1 = -np.dot(n1, get_plane_centroid(p1))
    d2 = -np.dot(n2, get_plane_centroid(p2))
    d3 = -np.dot(n3, get_plane_centroid(p3))

    A = np.vstack([n1, n2, n3])
    b = np.array([-d1, -d2, -d3])

    if abs(np.linalg.det(A)) < 1e-6:
        return None

    return np.linalg.solve(A, b)


# =========================================================
# CORNERS
# =========================================================

def _rectangular_fallback(walls, floor, cloud_pts, allow_cloud_synthesis=True):
    """Fit a rectangular polygon to the room using 2-D XY line intersection.

    This is the PRIMARY corner-finding method (not a fallback).  It is more
    robust than three-plane intersection because:
      • Uses all inlier points projected to XY to characterise each wall line,
        not just a single centroid + normal estimate.
      • Sub-patches of the same physical wall are detected and merged by a
        sign-based discriminant (see face_pair below) so they do not produce
        spurious thin polygons.
            • A missing face (wall not directly visible) is estimated from robust
                cloud extents along that axis, with centroid mirroring only as a
                fallback when extent evidence is poor.

    Algorithm
    ---------
    1. Project each wall's XY normal and centroid to a signed offset
       s = n_xy · centroid_xy  ("d_xy" in wall-line form n_xy·p = s).
    2. Cluster walls into 2 direction groups (azimuth mod 180°, ±20° tol).
    3. Check that the two groups are roughly perpendicular.
    4. For each axis, discriminate: if the group's signed offsets span BOTH
       signs (one positive, one negative after reference-frame alignment),
       they are genuine opposite walls → use directly.  If they are all the
       same sign they are sub-patches of the same face → take the median as
       the representative face and synthesise the opposite.
    5. Intersect the 4 axis lines to produce 4 XY corners.

    Returns
    -------
    (poly, synthesised, direct_wall_faces):
        poly               : (4,3) ndarray of ordered corners at floor Z, or None
        synthesised        : True when at least one face was estimated from the
                             cloud extent rather than directly measured.
        direct_wall_faces  : number of room wall faces directly supported by
                             measured geometry (2–4 for a rectangular room).
    """
    if len(walls) < 2:
        return None, False, 0

    # Robust interior reference. In walking SLAM the global origin is arbitrary
    # (typically wherever frame 0 was captured), so wall-normal orientation
    # must NOT depend on absolute coordinates relative to the origin.
    # Using the clipped merged-cloud XY median gives a stable room-interior
    # point from which opposite faces can be oriented outward correctly.
    room_ctr_xy = np.median(cloud_pts[:, :2], axis=0)

    # --- build per-wall (n_xy_norm, s) where n_xy · p_xy = s is the wall line ---
    wall_info = []
    for w in walls:
        n = get_plane_normal(w)
        c = get_plane_centroid(w)
        n_xy = np.array([n[0], n[1]])
        nn = np.linalg.norm(n_xy)
        if nn < 0.1:
            continue  # nearly horizontal; shouldn't be here but skip
        n_xy /= nn

        # Use the 95th-percentile projection of inlier XY points onto the wall
        # normal rather than the centroid.  When a wall has furniture in front
        # (desk, shelf), the centroid is pulled toward the sensor; the 95th
        # percentile robustly finds the actual wall surface further behind.
        # For a clean flat wall all inliers project to the same depth, so the
        # percentile equals the mean — no regression for unobstructed walls.
        c_xy = np.array([c[0], c[1]])

        # Ensure n_xy points OUTWARD relative to the room interior, not the
        # arbitrary SLAM origin. RANSAC may orient the plane normal in either
        # direction; flip when the normal points from the wall centroid TOWARD
        # the room-centre estimate. After this, opposite room faces naturally
        # have opposite n_xy directions and the sign-based face discriminator
        # works correctly.
        n_full = np.array([n[0], n[1], n[2]])
        c_full = np.array([c[0], c[1], c[2]])
        if float(np.dot(c_xy - room_ctr_xy, n_xy)) < 0:
            n_full = -n_full
            n_xy   = -n_xy

        coords = w.get("shell_coords") if isinstance(w, dict) else None
        n_xy_len = float(np.linalg.norm(n_full[:2]))
        if coords is not None and len(coords) >= 10 and n_xy_len > 0.1:
            # Project each inlier onto the (2-D, normalised) wall normal.
            # With an outward normal, inliers on the wall surface have the
            # LARGEST projection; furniture between sensor and wall has a
            # SMALLER projection.  The 95th pct therefore finds the actual
            # wall surface behind any furniture.
            #
            # We deliberately use the raw 2-D XY projection rather than the
            # tilt-corrected (n_full · p)/|n_xy| value.  The tilt-correction
            # would be exact for a physically inclined wall, but in practice
            # the residual z_comp in wall normals comes from world_align
            # imperfection rather than real wall lean.  For such apparent
            # tilts the tilt-correction adds a spurious offset of
            #   n_z · c_z / |n_xy|  (~0.09 m at z_comp=0.115, c_z=0.8 m)
            # which systematically overestimates room dimensions.
            # The raw-XY 95th pct works well for clean walls, but in walking
            # scans it can still be pulled OUTWARD by doorway bleed or sparse
            # high-Z returns near the ceiling line.  To stabilise the fitted
            # room width/length, estimate a second offset from the LOWER wall
            # band and clamp the full-wall estimate to at most ~5 cm beyond
            # that lower-band value.
            proj = coords[:, :2] @ n_xy
            pcts = np.percentile(proj, [5, 25, 50, 75, 95, 99])
            log(f"[DEBUG] 2-D wall proj dist: "
                f"p5={pcts[0]:.3f} p25={pcts[1]:.3f} p50={pcts[2]:.3f} "
                f"p75={pcts[3]:.3f} p95={pcts[4]:.3f} p99={pcts[5]:.3f}", "debug")

            s_full = float(pcts[4])
            z_min = float(coords[:, 2].min())
            z_max = float(coords[:, 2].max())
            z_span = z_max - z_min
            low_band_mask = coords[:, 2] <= (z_min + 0.35 * z_span)
            if z_span >= 0.6 and int(low_band_mask.sum()) >= max(15, int(0.15 * len(coords))):
                proj_low = coords[low_band_mask, :2] @ n_xy
                s_low = float(np.percentile(proj_low, 95))
                s_guard = max(s_low + 0.10, float(pcts[3]))
                if s_full > s_guard + 0.08:
                    s = s_guard
                    log(
                        f"[DEBUG] 2-D wall offset clamp: full95={s_full:.3f} m, "
                        f"low95={s_low:.3f} m, guard={s_guard:.3f} m → using {s:.3f} m",
                        "debug",
                    )
                else:
                    s = s_full
            else:
                s = s_full
        else:
            s = float(n_xy @ c_xy)   # fallback to centroid

        az   = float(np.degrees(np.arctan2(n_xy[1], n_xy[0]))) % 180  # fold to [0,180)
        area = float(get_plane_area(w))
        wall_info.append({"n_xy": n_xy, "s": s, "az180": az, "area": area})
        log(f"[DEBUG] 2-D wall offset: n_xy=[{n_xy[0]:.3f},{n_xy[1]:.3f}] "
            f"s={s:.3f} m  az={az:.1f}°  area={area:.2f} m²", "debug")

    if len(wall_info) < 2:
        return None, False, 0

    def _az_diff_deg(a, b):
        d = abs(a - b)
        return min(d, 180.0 - d)

    # --- cluster into direction groups (±20° tolerance on az mod 180) ---
    groups = []  # each group: list of wall_info dicts with similar az180
    angle_tol = 20.0
    for wi in wall_info:
        placed = False
        for g in groups:
            # compare az180 values mod 180 with wraparound at 0/180
            diff = _az_diff_deg(wi["az180"], g[0]["az180"])
            if diff < angle_tol:
                g.append(wi)
                placed = True
                break
        if not placed:
            groups.append([wi])

    # If more than 2 direction groups formed (e.g. a tilted desk plane at 25°
    # from a true wall nudges a third cluster), try to merge the two closest
    # groups if their angular distance is ≤35°.  group_mean_n will then drop
    # any intra-group outlier (>10° from area-weighted mean) so the slanted
    # intruder is removed before the rectangle fit.
    merge_tol = 35.0
    while len(groups) > 2:
        best_i, best_j, best_d = 0, 1, 999.0
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                d = _az_diff_deg(groups[i][0]["az180"], groups[j][0]["az180"])
                if d < best_d:
                    best_d, best_i, best_j = d, i, j
        if best_d > merge_tol:
            break   # no close pair — give up below
        merged = groups[best_i] + groups[best_j]
        remaining = [g for k, g in enumerate(groups) if k not in (best_i, best_j)]
        groups = [merged] + remaining
        log(f"[DEBUG] 2-D fit: merged direction groups "
            f"(az-diff={best_d:.1f}°) → {len(groups)} groups remain", "debug")

    if len(groups) != 2:
        # Noisy walking-wall hints may produce many direction groups even in a
        # rectangular room. Recover the two dominant orthogonal axes from an
        # area-weighted azimuth histogram, then assign walls to the nearer axis
        # and drop outliers.
        bins = 36
        edges = np.linspace(0.0, 180.0, bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        hist = np.zeros(bins, dtype=float)
        for wi in wall_info:
            idx = min(bins - 1, int((wi["az180"] / 180.0) * bins))
            hist[idx] += max(wi["area"], 1e-3)

        best_pair = None
        best_score = -1.0
        for i in range(bins):
            for j in range(i + 1, bins):
                sep = _az_diff_deg(float(centers[i]), float(centers[j]))
                if not (70.0 <= sep <= 110.0):
                    continue
                score = float(hist[i] + hist[j])
                if score > best_score:
                    best_score = score
                    best_pair = (i, j)

        if best_pair is None:
            # Fallback: strongest peak plus its orthogonal companion.
            peak1 = int(np.argmax(hist))
            az1 = float(centers[peak1])
            az2 = (az1 + 90.0) % 180.0
        else:
            az1 = float(centers[best_pair[0]])
            az2 = float(centers[best_pair[1]])

        rec_groups = [[], []]
        dropped = 0
        for wi in wall_info:
            d1 = _az_diff_deg(wi["az180"], az1)
            d2 = _az_diff_deg(wi["az180"], az2)
            if min(d1, d2) > 25.0:
                dropped += 1
                continue
            rec_groups[0 if d1 <= d2 else 1].append(wi)

        if rec_groups[0] and rec_groups[1]:
            groups = rec_groups
            log(
                f"[INFO] 2-D fit: recovered dominant wall axes at {az1:.0f}° and {az2:.0f}° "
                f"(dropped {dropped} outlier wall hints)",
                "info",
            )
        else:
            # Could not reduce to 2 direction clusters — not a simple rectangle
            return None, False, 0

    # --- verify the two groups are roughly perpendicular ---
    def group_mean_n(g, label=""):
        """Area-weighted mean of XY normals; largest wall sets reference.

        1. Sort by area descending so the largest (most reliable) plane
           provides the reference direction for half-space folding.
        2. Fold all normals into that half-space.
        3. Compute area-weighted mean, then drop intra-group outliers
           (normals > 10 deg from mean) and recompute -- removes a body /
           furniture plane that snuck into the group.
        """
        g_sorted = sorted(g, key=lambda wi: wi["area"], reverse=True)
        ns    = np.array([wi["n_xy"] for wi in g_sorted], dtype=float)
        areas = np.maximum([wi["area"] for wi in g_sorted], 1e-3)

        ref = ns[0].copy()
        for i in range(1, len(ns)):
            if ns[i] @ ref < 0:
                ns[i] = -ns[i]

        n_avg = np.average(ns, axis=0, weights=areas)
        n_avg /= (np.linalg.norm(n_avg) + 1e-9)

        keep = np.array([float(np.dot(n, n_avg)) > np.cos(np.radians(10))
                         for n in ns])
        if not keep.any():
            keep[:] = True   # safety fallback

        if not keep.all():
            for i, wi in enumerate(g_sorted):
                if not keep[i]:
                    log("[DEBUG] 2-D fit: dropped outlier plane from '%s' group "
                        "(area=%.2f m2, deviation > 10 deg)" % (label, wi["area"]),
                        "debug")
            ns    = ns[keep]
            areas = np.array(areas)[keep]
            n_avg = np.average(ns, axis=0, weights=areas)
            n_avg /= (np.linalg.norm(n_avg) + 1e-9)

        if len(g_sorted) > 2:
            area_str = ", ".join("%.2f" % wi["area"] for wi in g_sorted)
            log("[DEBUG] 2-D fit: direction group '%s' has %d walls "
                "(areas: %s m2)" % (label, len(g_sorted), area_str), "debug")
        return n_avg

    n1 = group_mean_n(groups[0], "axis-1")
    n2 = group_mean_n(groups[1], "axis-2")
    dot = abs(float(n1 @ n2))
    if dot > 0.45:   # not within ~63° of perpendicular → not a rectangle
        return None, False, 0

    # Rectangular-room model: recovered wall axes from noisy walking hints can
    # be only approximately perpendicular. If we keep that skew, the corner
    # solve produces a parallelogram and inflates the reported area. Snap the
    # second axis to the exact perpendicular of the first when the recovered
    # angle drifts materially away from 90 degrees, while preserving the
    # measured half-plane closest to the original axis-2 direction.
    ang_deg = float(np.degrees(np.arccos(np.clip(dot, -1.0, 1.0))))
    if abs(90.0 - ang_deg) > 6.0:
        n2_ortho = np.array([-n1[1], n1[0]], dtype=float)
        if float(np.dot(n2_ortho, n2)) < 0:
            n2_ortho = -n2_ortho
        log(
            f"[INFO] 2-D fit: orthogonalised wall axes from {ang_deg:.1f}° to 90.0° "
            f"for rectangular-room fitting",
            "info",
        )
        n2 = n2_ortho

    # --- for each axis, collect signed offsets (both faces) ---
    # For axis direction n, the two faces have s ≈ s_pos and s ≈ -s_neg.
    # We need one positive and one negative s per axis (relative to n).

    def offsets_along(n_ref, g):
        """Return list of signed offsets; flip normal when dot < 0."""
        return [
            (wi["s"] if (wi["n_xy"] @ n_ref) > 0 else -wi["s"])
            for wi in g
        ]

    s1_vals = sorted(offsets_along(n1, groups[0]))
    s2_vals = sorted(offsets_along(n2, groups[1]))

    # Wall inliers provide a stable interior reference; the clipped merged cloud
    # provides robust low/high extent percentiles for synthesising a missing
    # face when only one side of an axis was directly measured.
    def _support_xy(coords):
        """Use the lower wall band for footprint support when available."""
        if coords is None or len(coords) == 0:
            return np.empty((0, 2))
        z_min = float(coords[:, 2].min())
        z_max = float(coords[:, 2].max())
        z_span = z_max - z_min
        if z_span >= 0.6:
            mask = coords[:, 2] <= (z_min + 0.40 * z_span)
            if int(mask.sum()) >= max(20, int(0.20 * len(coords))):
                return coords[mask][:, :2]
        return coords[:, :2]

    _wall_inlier_xy_parts = [
        _support_xy(w["shell_coords"])
        for w in walls
        if isinstance(w, dict) and "shell_coords" in w and len(w["shell_coords"])
    ]
    _wall_inlier_xy_parts = [part for part in _wall_inlier_xy_parts if len(part)]
    if _wall_inlier_xy_parts:
        inlier_xy = np.vstack(_wall_inlier_xy_parts)
    else:
        inlier_xy = cloud_pts[:, :2]   # fallback (should not occur)

    _synthesised = False

    def _axis_support(n_axis):
        """Robust projected extent along an axis from aligned wall inlier points."""
        axis_parts = []
        for w in walls:
            coords = w.get("shell_coords") if isinstance(w, dict) else None
            if coords is None or len(coords) == 0:
                continue
            wn = get_plane_normal(w)
            wn_xy = np.array([wn[0], wn[1]], dtype=float)
            wn_xy_n = float(np.linalg.norm(wn_xy))
            if wn_xy_n < 1e-6:
                continue
            wn_xy /= wn_xy_n
            if abs(float(np.dot(wn_xy, n_axis))) >= np.cos(np.radians(22.0)):
                _xy = _support_xy(coords)
                if len(_xy):
                    axis_parts.append(_xy)

        axis_span = axis_lo = axis_hi = None
        if axis_parts:
            proj_axis = np.vstack(axis_parts) @ n_axis
            axis_lo, axis_hi = np.percentile(proj_axis, [2, 98])
            axis_span = float(abs(axis_hi - axis_lo))

        proj_cloud = cloud_pts[:, :2] @ n_axis
        cloud_lo, cloud_hi = np.percentile(proj_cloud, [2, 98])
        cloud_span = float(abs(cloud_hi - cloud_lo))
        return axis_span, axis_lo, axis_hi, cloud_span, float(cloud_lo), float(cloud_hi)

    def face_pair(s_vals, n_axis, label):
        """Return (s_lo, s_hi, was_synthesised, direct_faces) for one room axis.

        Key discriminant — sign of the signed offsets after offsets_along:
          • offsets_along flips the s value when a wall's n_xy points in the
            OPPOSITE direction to the group reference.  This means that genuine
            opposite walls always produce one positive and one negative s value.
          • Sub-patches of the SAME wall have nearly identical n_xy directions,
            so no flip occurs — all s values are the same sign.

        If all s_vals share a sign → sub-patches: take median and synthesise
        the opposite face.
        If s_vals span both signs → two genuine opposite walls: use as-is.
        """
        axis_span, axis_lo, axis_hi, cloud_span, cloud_lo, cloud_hi = _axis_support(n_axis)

        def _maybe_recover_from_axis_extent(s_lo, s_hi, reason):
            span_fit = float(abs(s_hi - s_lo))
            if axis_span is None:
                return None

            support_span = float(axis_span)
            support_lo = float(min(axis_lo, axis_hi))
            support_hi = float(max(axis_lo, axis_hi))
            support_src = "axis-wall extent"

            if support_span > 1.0 and span_fit < 0.72 * support_span:
                log(
                    f"[INFO] 2-D room fit: rejected {reason} for {label} "
                    f"({span_fit:.2f} m < 72% of {support_src} {support_span:.2f} m); "
                    f"using {support_src} instead",
                    "info",
                )
                return support_lo, support_hi, False, 2
            if support_span > 1.0 and span_fit > 1.20 * support_span:
                log(
                    f"[INFO] 2-D room fit: rejected oversized {reason} for {label} "
                    f"({span_fit:.2f} m > 120% of {support_src} {support_span:.2f} m); "
                    f"using {support_src} instead",
                    "info",
                )
                return support_lo, support_hi, False, 2
            return None

        if len(s_vals) >= 2:
            has_pos = any(s > 0.0 for s in s_vals)
            has_neg = any(s < 0.0 for s in s_vals)
            if has_pos and has_neg:
                # Signed offsets span both sides → genuine opposite walls.
                s_lo = float(min(s_vals))
                s_hi = float(max(s_vals))
                recovered = _maybe_recover_from_axis_extent(s_lo, s_hi, "direct opposite-wall span")
                if recovered is not None:
                    return recovered
                return s_lo, s_hi, False, 2

            # Same-sign offsets can still represent TWO directly measured faces
            # if the wall normals were not perfectly oriented before grouping.
            # Detect that case by looking for a strong bimodal split in the
            # offsets along this axis.
            s_sorted = np.sort(np.asarray(s_vals, dtype=float))
            if len(s_sorted) >= 3:
                gaps = np.diff(s_sorted)
                k = int(np.argmax(gaps))
                gap = float(gaps[k]) if len(gaps) else 0.0
                if gap > 0.60:
                    lo_grp = s_sorted[:k + 1]
                    hi_grp = s_sorted[k + 1:]
                    if len(lo_grp) and len(hi_grp):
                        s_lo = float(np.median(lo_grp))
                        s_hi = float(np.median(hi_grp))
                        span_clusters = float(abs(s_hi - s_lo))
                        if axis_span is not None:
                            support_span = float(axis_span)
                        else:
                            support_span = span_clusters

                        # Same-sign bimodal splits often come from two sub-patches
                        # on one physical face. Accept them as opposite faces only
                        # when the implied span is broadly consistent with the
                        # observed extent along this axis.
                        if support_span > 1.0 and span_clusters < 0.72 * support_span:
                            log(
                                f"[INFO] 2-D room fit: rejected same-sign offset split for {label} "
                                f"({span_clusters:.2f} m < 72% of support extent {support_span:.2f} m); "
                                f"treating as one face",
                                "info",
                            )
                        else:
                            recovered = _maybe_recover_from_axis_extent(s_lo, s_hi, "same-sign offset split")
                            if recovered is not None:
                                return recovered
                            log(
                                f"[INFO] 2-D room fit: inferred two directly measured "
                                f"faces for {label} from offset clusters "
                                f"({s_lo:.2f} m, {s_hi:.2f} m; gap {gap:.2f} m)",
                                "info",
                            )
                            return min(s_lo, s_hi), max(s_lo, s_hi), False, 2

            # All same sign → sub-patches of one face.
            s_measured = float(np.median(s_vals))
        else:
            s_measured = s_vals[0]

        if axis_span is not None and axis_span > 1.0:
            support_lo = float(min(axis_lo, axis_hi))
            support_hi = float(max(axis_lo, axis_hi))
            support_span = float(abs(support_hi - support_lo))
            if support_span >= max(1.2, abs(s_measured) * 0.35):
                s_lo = support_lo
                s_hi = support_hi
                log(
                    f"[INFO] 2-D room fit: synthesised {label} face from axis-wall support "
                    f"({s_lo:.2f}…{s_hi:.2f} m) instead of full cloud extent",
                    "info",
                )
                return s_lo, s_hi, True, 1

        if not allow_cloud_synthesis:
            return None

        proj_all = cloud_pts[:, :2] @ n_axis
        p_lo, p_hi = np.percentile(proj_all, [2, 98])
        # Use the opposite robust extent of the clipped merged cloud as the
        # primary estimate for the missing face. This is more stable than
        # centroid mirroring when one wall axis is sparsely observed.
        if s_measured >= 0:
            s_missing = float(p_lo)
        else:
            s_missing = float(p_hi)

        # Safety fallback: if the extent-based estimate collapses too close to
        # the measured face, fall back to centroid mirroring.
        if abs(s_missing - s_measured) < 0.8:
            inlier_mean = float((inlier_xy @ n_axis).mean())
            s_missing = 2.0 * inlier_mean - s_measured
        else:
            inlier_mean = float((inlier_xy @ n_axis).mean())

        s_lo = min(s_measured, s_missing)
        s_hi = max(s_measured, s_missing)
        log(
            f"[INFO] 2-D room fit: synthesised {label} face at "
            f"offset {s_missing:.2f} m  (cloud extents {p_lo:.2f}…{p_hi:.2f} m, "
            f"wall-inlier centroid {inlier_mean:.2f} m, measured face {s_measured:.2f} m)",
            "info",
        )
        log(
            "[WARN] One wall axis did not have enough direct geometric support, "
            "so the opposite face was estimated from the scan extent. This can "
            "happen even when you walked the full room if that wall was partly "
            "occluded by furniture, seen at a shallow angle, or merged into the "
            "same signed wall group by noisy normals.",
            "info",
        )
        return s_lo, s_hi, True, 1

    axis1 = face_pair(s1_vals, n1, "axis-1")
    axis2 = face_pair(s2_vals, n2, "axis-2")
    if axis1 is None or axis2 is None:
        return None, False, 0

    s1_lo, s1_hi, syn1, direct1 = axis1
    s2_lo, s2_hi, syn2, direct2 = axis2

    def _regularize_span(s_lo, s_hi, n_axis, label, direct_faces):
        """Prevent collapsed room dimensions from inner wall sub-patches."""
        dim_fit = float(abs(s_hi - s_lo))

        # 1) Axis-specific wall evidence: use only walls aligned with this axis.
        axis_parts = []
        for w in walls:
            coords = w.get("shell_coords") if isinstance(w, dict) else None
            if coords is None or len(coords) == 0:
                continue
            wn = get_plane_normal(w)
            wn_xy = np.array([wn[0], wn[1]], dtype=float)
            wn_xy_n = float(np.linalg.norm(wn_xy))
            if wn_xy_n < 1e-6:
                continue
            wn_xy /= wn_xy_n
            if abs(float(np.dot(wn_xy, n_axis))) >= np.cos(np.radians(22.0)):
                axis_parts.append(coords[:, :2])

        axis_candidate = None
        if axis_parts:
            proj_axis = np.vstack(axis_parts) @ n_axis
            a_lo, a_hi = np.percentile(proj_axis, [5, 95])
            dim_axis = float(abs(a_hi - a_lo))
            axis_candidate = (
                dim_axis,
                float(min(a_lo, a_hi)),
                float(max(a_lo, a_hi)),
                "axis-wall extent",
            )

        # 2) Entire clipped merged cloud along this axis.
        proj_all = cloud_pts[:, :2] @ n_axis
        c_lo, c_hi = np.percentile(proj_all, [5, 95])
        dim_cloud = float(abs(c_hi - c_lo))
        cloud_candidate = (
            dim_cloud,
            float(min(c_lo, c_hi)),
            float(max(c_lo, c_hi)),
            "clipped-cloud extent",
        )

        # If one face was synthesised, be more willing to trust the extent evidence.
        if direct_faces < 2:
            widen_ratio = 0.90
        else:
            widen_ratio = 0.84

        preferred = axis_candidate

        if preferred is not None:
            _ad = axis_candidate[0] if axis_candidate is not None else None
            log(
                f"[DEBUG] 2-D room fit support for {label}: "
                f"axis-wall={_ad:.2f} m, cloud={cloud_candidate[0]:.2f} m"
                if _ad is not None else
                f"[DEBUG] 2-D room fit support for {label}: "
                f"cloud={cloud_candidate[0]:.2f} m",
                "debug",
            )

        if preferred is not None:
            best_dim, best_lo, best_hi, best_src = preferred
            if best_dim > 1.0 and dim_fit < widen_ratio * best_dim:
                log(
                    f"[INFO] 2-D room fit: widened {label} span from {dim_fit:.2f} m "
                    f"to {best_src} {best_dim:.2f} m",
                    "info",
                )
                return best_lo, best_hi

        best_dim, best_lo, best_hi, best_src = cloud_candidate
        pref_dim = preferred[0] if preferred is not None else 0.0
        # Cloud extents are only a last-resort guardrail. They are too broad to
        # drive room dimensions when wall recovery is noisy, so only use them
        # when no axis-specific support exists and the increase is modest.
        if preferred is None and best_dim > 1.0 and dim_fit < 0.85 * best_dim and best_dim <= dim_fit + 0.35:
            log(
                f"[INFO] 2-D room fit: widened {label} span from {dim_fit:.2f} m "
                f"to {best_src} {best_dim:.2f} m",
                "info",
            )
            return best_lo, best_hi

        return s_lo, s_hi

    s1_lo, s1_hi = _regularize_span(s1_lo, s1_hi, n1, "axis-1", direct1)
    s2_lo, s2_hi = _regularize_span(s2_lo, s2_hi, n2, "axis-2", direct2)
    _synthesised = syn1 or syn2

    # --- build 4 corners by intersecting the 4 lines in XY ---
    # Wall line: n · p = s  ⟹  [n1x n1y; n2x n2y] [x;y] = [s1; s2]
    M = np.array([[n1[0], n1[1]],
                  [n2[0], n2[1]]])
    det = float(np.linalg.det(M))
    if abs(det) < 1e-6:
        return None, False, 0

    floor_z = get_plane_floor_z(floor)

    dim1 = abs(s1_hi - s1_lo)
    dim2 = abs(s2_hi - s2_lo)
    log(f"[DEBUG] 2-D rect spans: axis-1 [{s1_lo:.3f} → {s1_hi:.3f}] = {dim1:.3f} m, "
        f"axis-2 [{s2_lo:.3f} → {s2_hi:.3f}] = {dim2:.3f} m  "
        f"(area={dim1*dim2:.3f} m²)", "debug")
    log(
        f"[INFO] 2-D room fit final spans: axis-1={dim1:.2f} m, "
        f"axis-2={dim2:.2f} m, rectangular area={dim1*dim2:.2f} m²",
        "info",
    )

    corners_xy = []
    for s1 in (s1_lo, s1_hi):
        for s2 in (s2_lo, s2_hi):
            rhs = np.array([s1, s2])
            xy = np.linalg.solve(M, rhs)
            corners_xy.append(xy)

    corners_xy = np.array(corners_xy)  # (4, 2)

    # Order by angle from centroid (same as order_poly)
    c = corners_xy.mean(axis=0)
    ang = np.arctan2(corners_xy[:, 1] - c[1], corners_xy[:, 0] - c[0])
    corners_xy = corners_xy[np.argsort(ang)]

    # Add floor Z
    poly = np.column_stack([corners_xy, np.full(4, floor_z)])
    return poly, _synthesised, int(direct1 + direct2)


def _walking_rect_from_wall_extents(walls, floor, cloud_pts):
    """Fit a rectangular footprint from dominant-axis lower-wall support.

    This avoids the fragile opposite-face offset inference used by the generic
    rectangle fit.  It is intended for walking scans where the geometry is
    already dominated by per-frame wall hints.
    """
    global LAST_WALKING_FIT_INFO
    LAST_WALKING_FIT_INFO = {
        "max_direct_tail": 0.0,
        "ignored_outer_groups": False,
    }

    if len(walls) < 2:
        return None, False, 0

    def _support_xy(coords):
        if coords is None or len(coords) == 0:
            return np.empty((0, 2))
        z_min = float(coords[:, 2].min())
        z_max = float(coords[:, 2].max())
        z_span = z_max - z_min
        if z_span >= 0.6:
            mask = coords[:, 2] <= (z_min + 0.40 * z_span)
            if int(mask.sum()) >= max(20, int(0.20 * len(coords))):
                return coords[mask][:, :2]
        return coords[:, :2]

    wall_entries = []
    for w in walls:
        coords = w.get("shell_coords") if isinstance(w, dict) else None
        if coords is None or len(coords) < 10:
            continue
        n = get_plane_normal(w)
        n_xy = np.array([n[0], n[1]], dtype=float)
        nn = float(np.linalg.norm(n_xy))
        if nn < 0.1:
            continue
        n_xy /= nn
        support_xy = _support_xy(coords)
        if len(support_xy) < 10:
            continue
        wall_entries.append({
            "n_xy": n_xy,
            "az": float(np.arctan2(n_xy[1], n_xy[0])) % np.pi,
            "area": float(get_plane_area(w)),
            "support_xy": support_xy,
        })

    if len(wall_entries) < 2:
        return None, False, 0

    bins = 72
    hist = np.zeros(bins, dtype=float)
    edges = np.linspace(0.0, np.pi, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    for wi in wall_entries:
        idx = min(bins - 1, int((wi["az"] / np.pi) * bins))
        hist[idx] += max(wi["area"], 1e-3)

    best_pair = None
    best_score = -1.0
    for i in range(bins):
        for j in range(i + 1, bins):
            sep = _angdiff_pi(float(centers[i]), float(centers[j]))
            if not (np.radians(70.0) <= sep <= np.radians(110.0)):
                continue
            ortho_weight = 1.0 - min(1.0, abs(sep - (np.pi / 2.0)) / np.radians(20.0))
            score = float(hist[i] + hist[j]) * (0.6 + 0.4 * ortho_weight)
            if score > best_score:
                best_score = score
                best_pair = (i, j)

    if best_pair is None:
        return None, False, 0

    def _axis_from_target(target_az):
        assigned = [
            wi for wi in wall_entries
            if _angdiff_pi(wi["az"], float(target_az)) <= np.radians(25.0)
        ]
        if not assigned:
            return None, None

        ref = np.array([np.cos(target_az), np.sin(target_az)])
        ns = []
        ws = []
        for wi in assigned:
            n_xy = wi["n_xy"].copy()
            if float(np.dot(n_xy, ref)) < 0:
                n_xy = -n_xy
            wi["axis_n_xy"] = n_xy
            wi["axis_offset"] = float(np.median(wi["support_xy"] @ n_xy))
            ns.append(n_xy)
            ws.append(max(wi["area"], 1e-3))
        n_axis = np.average(np.array(ns), axis=0, weights=np.array(ws))
        n_axis /= (np.linalg.norm(n_axis) + 1e-9)
        for wi in assigned:
            wi["axis_offset"] = float(np.median(wi["support_xy"] @ n_axis))
        return n_axis, assigned

    def _weighted_offset(entries):
        offsets = np.array([e["axis_offset"] for e in entries], dtype=float)
        weights = np.array([max(e["area"], 1e-3) for e in entries], dtype=float)
        order = np.argsort(offsets)
        offsets = offsets[order]
        weights = weights[order]
        cum = np.cumsum(weights)
        return float(offsets[np.searchsorted(cum, 0.5 * cum[-1], side="left")])

    def _cluster_offsets(entries, min_sep=0.55):
        if not entries:
            return []
        entries = sorted(entries, key=lambda e: e["axis_offset"])
        groups = [[entries[0]]]
        for entry in entries[1:]:
            prev_offset = float(np.median([e["axis_offset"] for e in groups[-1]]))
            if abs(entry["axis_offset"] - prev_offset) <= min_sep:
                groups[-1].append(entry)
            else:
                groups.append([entry])
        return groups

    def _axis_faces(n_axis, assigned, label):
        if n_axis is None or not assigned:
            return None

        groups = _cluster_offsets(assigned)
        groups = [g for g in groups if sum(e["area"] for e in g) >= 0.20]
        groups = sorted(groups, key=lambda g: _weighted_offset(g))

        proj_cloud = cloud_pts[:, :2] @ n_axis
        cloud_lo, cloud_hi = np.percentile(proj_cloud, [2, 98])
        cloud_lo_tight, cloud_hi_tight = np.percentile(proj_cloud, [5, 95])

        if len(groups) >= 2:
            best_pair = None
            best_pair_score = -1.0
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    lo_candidate = groups[i]
                    hi_candidate = groups[j]
                    lo_support = np.vstack([e["support_xy"] for e in lo_candidate]) @ n_axis
                    hi_support = np.vstack([e["support_xy"] for e in hi_candidate]) @ n_axis
                    span_candidate = float(np.percentile(hi_support, 80) - np.percentile(lo_support, 20))
                    if span_candidate < 1.0:
                        continue
                    pair_area = float(sum(e["area"] for e in lo_candidate) + sum(e["area"] for e in hi_candidate))
                    pair_score = pair_area * span_candidate
                    if pair_score > best_pair_score:
                        best_pair_score = pair_score
                        best_pair = (lo_candidate, hi_candidate)

            if best_pair is None:
                return None

            lo_group, hi_group = best_pair
            lo_support = np.vstack([e["support_xy"] for e in lo_group]) @ n_axis
            hi_support = np.vstack([e["support_xy"] for e in hi_group]) @ n_axis
            s_lo = float(np.percentile(lo_support, 20))
            s_hi = float(np.percentile(hi_support, 80))
            lo_tail = float(np.percentile(lo_support, 50) - np.percentile(lo_support, 20))
            hi_tail = float(np.percentile(hi_support, 80) - np.percentile(hi_support, 50))
            LAST_WALKING_FIT_INFO["max_direct_tail"] = max(
                float(LAST_WALKING_FIT_INFO.get("max_direct_tail", 0.0)),
                lo_tail,
                hi_tail,
            )
            direct = 2
            synth = False
            if len(groups) > 2 and (lo_group is not groups[0] or hi_group is not groups[-1]):
                LAST_WALKING_FIT_INFO["ignored_outer_groups"] = True
                log(
                    f"[INFO] Walking extent fit: ignored weaker outer {label} group(s) and chose the strongest opposing pair",
                    "info",
                )
        else:
            only = groups[0] if groups else assigned
            s_measured = _weighted_offset(only)
            if s_measured >= 0:
                s_missing = float(cloud_lo_tight)
            else:
                s_missing = float(cloud_hi_tight)
            if abs(s_missing - s_measured) < 0.8:
                # Fall back to the broader robust extent before giving up.
                if s_measured >= 0:
                    s_missing = float(cloud_lo)
                else:
                    s_missing = float(cloud_hi)
            if abs(s_missing - s_measured) < 0.8:
                return None
            s_lo = min(s_measured, s_missing)
            s_hi = max(s_measured, s_missing)
            direct = 1
            synth = True
            log(
                f"[INFO] Walking extent fit: synthesised missing {label} face "
                f"from cloud support ({s_lo:.2f}…{s_hi:.2f} m; tight extent {cloud_lo_tight:.2f}…{cloud_hi_tight:.2f} m)",
                "info",
            )

        span = float(abs(s_hi - s_lo))
        cloud_span = float(abs(cloud_hi - cloud_lo))
        if span < 1.0:
            return None
        if cloud_span > 1.0 and span < 0.65 * cloud_span:
            log(
                f"[INFO] Walking extent fit: rejected undersized {label} span "
                f"{span:.2f} m vs cloud support {cloud_span:.2f} m",
                "info",
            )
            return None
        if cloud_span > 1.0 and span > cloud_span + 0.60:
            log(
                f"[INFO] Walking extent fit: rejected oversized {label} span "
                f"{span:.2f} m vs cloud support {cloud_span:.2f} m",
                "info",
            )
            return None

        return {
            "s_lo": float(min(s_lo, s_hi)),
            "s_hi": float(max(s_lo, s_hi)),
            "direct_faces": direct,
            "synthetic": synth,
        }

    n1, assigned1 = _axis_from_target(float(centers[best_pair[0]]))
    n2, assigned2 = _axis_from_target(float(centers[best_pair[1]]))
    if n1 is None or n2 is None:
        return None, False, 0
    dot12 = abs(float(np.dot(n1, n2)))
    if dot12 > 0.45:
        return None, False, 0

    ang12_deg = float(np.degrees(np.arccos(np.clip(dot12, -1.0, 1.0))))
    if abs(90.0 - ang12_deg) > 6.0:
        n2_ortho = np.array([-n1[1], n1[0]], dtype=float)
        if float(np.dot(n2_ortho, n2)) < 0:
            n2_ortho = -n2_ortho
        log(
            f"[INFO] Walking extent fit: orthogonalised wall axes from {ang12_deg:.1f}° to 90.0°",
            "info",
        )
        n2 = n2_ortho

    axis1 = _axis_faces(n1, assigned1, "axis-1")
    axis2 = _axis_faces(n2, assigned2, "axis-2")
    if axis1 is None or axis2 is None:
        return None, False, 0

    s1_lo, s1_hi = axis1["s_lo"], axis1["s_hi"]
    s2_lo, s2_hi = axis2["s_lo"], axis2["s_hi"]
    dim1 = float(abs(s1_hi - s1_lo))
    dim2 = float(abs(s2_hi - s2_lo))
    if dim1 < 1.0 or dim2 < 1.0:
        return None, False, 0

    M = np.array([[n1[0], n1[1]],
                  [n2[0], n2[1]]])
    if abs(float(np.linalg.det(M))) < 1e-6:
        return None, False, 0

    floor_z = get_plane_floor_z(floor)
    corners_xy = []
    for s1 in (float(min(s1_lo, s1_hi)), float(max(s1_lo, s1_hi))):
        for s2 in (float(min(s2_lo, s2_hi)), float(max(s2_lo, s2_hi))):
            corners_xy.append(np.linalg.solve(M, np.array([s1, s2])))
    corners_xy = np.array(corners_xy)
    c = corners_xy.mean(axis=0)
    ang = np.arctan2(corners_xy[:, 1] - c[1], corners_xy[:, 0] - c[0])
    corners_xy = corners_xy[np.argsort(ang)]
    poly = np.column_stack([corners_xy, np.full(4, floor_z)])

    log(
        f"[INFO] Walking extent fit: axis-1={dim1:.2f} m, axis-2={dim2:.2f} m",
        "info",
    )
    return poly, bool(axis1["synthetic"] or axis2["synthetic"]), int(axis1["direct_faces"] + axis2["direct_faces"])


def corners(walls, floor, cloud_bounds=None):
    """Compute wall-corner XYZ points via three-plane intersection.

    Parameters
    ----------
    walls        : list of plane dicts
    floor        : plane dict
    cloud_bounds : optional (lo_xyz, hi_xyz) tuple from the aligned point cloud.
                   When supplied, intersection points far outside the cloud
                   bounding box are discarded as geometric artefacts.
    """
    pts = []
    for i in range(len(walls)):
        for j in range(i + 1, len(walls)):
            p = intersect(walls[i], walls[j], floor)
            if p is not None:
                pts.append(p)

    if not pts:
        return np.empty((0, 3))

    pts_arr = np.array(pts)

    # Defence-in-depth: discard corners outside the cloud bbox + 30 % margin.
    # This catches any near-parallel pairs that slip through the cos threshold.
    if cloud_bounds is not None:
        lo, hi = cloud_bounds
        span   = hi - lo
        margin = np.maximum(span * 0.30, 0.5)   # at least 0.5 m on each side
        mask   = np.all(
            (pts_arr >= lo - margin) & (pts_arr <= hi + margin), axis=1
        )
        if mask.sum() < len(pts_arr):
            log(
                f"[DEBUG] Removed {(~mask).sum()} spurious corner(s) outside "
                f"cloud bounds; keeping {mask.sum()}",
                "debug",
            )
        pts_arr = pts_arr[mask]

    return pts_arr


# =========================================================
# ORDER POLYGON
# =========================================================

def order_poly(pts):

    if len(pts)<3:
        raise RuntimeError("Could not detect room boundary")

    c=pts.mean(axis=0)
    ang=np.arctan2(pts[:,1]-c[1],pts[:,0]-c[0])
    return pts[np.argsort(ang)]


def poly_perimeter(poly):
    """Sum of edge lengths of a closed 2-D polygon."""
    total = 0.0
    n = len(poly)
    for i in range(n):
        p1 = poly[i, :2]
        p2 = poly[(i + 1) % n, :2]
        total += float(np.linalg.norm(p2 - p1))
    return total

def estimate_room_height(walls, floor, cloud_z_max=None, prefer_cloud_top=False,
                         floor_z_hint=None):
    """
    Estimate room height as mean(per-wall top Z) minus floor centroid Z.

    Each wall's top Z = max Z of its RANSAC inlier points.  Taking the mean
    across walls reduces noise from any single wall having outlier-trimmed
    corners.  Error is typically ±5–10 cm without a ceiling scan.

    cloud_z_max : if provided, the observed point-cloud ceiling — used as a
                  lower bound when wall-top estimates seem too short.
    prefer_cloud_top : walking-mode fallback. When True, prefer the merged
                  cloud top envelope if it is materially higher than the best
                  wall-top estimate.
    """
    floor_z = get_plane_floor_z(floor)
    if floor_z_hint is not None:
        floor_z = min(float(floor_z_hint), floor_z)

    wall_tops = []
    for w in walls:
        coords = w.get('shell_coords') if isinstance(w, dict) else None
        if coords is not None and len(coords):
            wall_tops.append(float(coords[:, 2].max()))

    if not wall_tops:
        # Fall back to cloud extent alone
        if cloud_z_max is not None:
            return max(float(cloud_z_max) - floor_z, 0.1)
        return None

    # Use the MAXIMUM wall-top (not mean).
    # The wall furthest from the sensor is the one most likely seen from
    # across the room (~3 m) and therefore has z_max closest to the actual
    # ceiling.  Any close wall (0.5 m away; z_top ≈ 1.7 m) or incidentally
    # captured furniture face that slips through has a lower z_top and would
    # only pull the mean DOWN.  max() gives the best-case estimate, which in
    # a rectangular room with one far wall IS the true ceiling height.
    ceiling_z = float(max(wall_tops))
    log(
        f"[DEBUG] Height: wall-top values={[round(z,2) for z in sorted(wall_tops,reverse=True)]} → "
        f"using max={ceiling_z:.2f} m",
        "debug",
    )

    # Cloud ceiling is always a lower bound (cannot be higher than tallest
    # visible point).
    if cloud_z_max is not None and (
        float(cloud_z_max) > ceiling_z and (
            prefer_cloud_top or (float(cloud_z_max) - ceiling_z > 0.20)
        )
    ):
        log(
            f"[DEBUG] Height: cloud Z max={float(cloud_z_max):.2f} m > "
            f"wall-top max={ceiling_z:.2f} m — using cloud ceiling",
            "debug",
        )
        ceiling_z = float(cloud_z_max)

    if prefer_cloud_top:
        if cloud_z_max is not None:
            log(
                f"[INFO] Height inputs: floor_z={floor_z:.2f} m, "
                f"wall_top_max={max(wall_tops):.2f} m, cloud_top={float(cloud_z_max):.2f} m",
                "info",
            )
        else:
            log(
                f"[INFO] Height inputs: floor_z={floor_z:.2f} m, "
                f"wall_top_max={max(wall_tops):.2f} m",
                "info",
            )

    height = ceiling_z - floor_z
    return max(height, 0.1)   # guard against degenerate clouds


# =========================================================
# SCENE EXPORT HELPERS
# =========================================================

def _polygon_centroid_xy(poly_xy):
    return np.mean(poly_xy, axis=0)


def _points_in_convex_polygon(points_xy, poly_xy, eps=1e-6):
    if len(poly_xy) < 3 or len(points_xy) == 0:
        return np.zeros(len(points_xy), dtype=bool)
    area2 = float(np.dot(poly_xy[:, 0], np.roll(poly_xy[:, 1], -1)) - np.dot(poly_xy[:, 1], np.roll(poly_xy[:, 0], -1)))
    ccw = area2 >= 0.0
    inside = np.ones(len(points_xy), dtype=bool)
    for i in range(len(poly_xy)):
        a = poly_xy[i]
        b = poly_xy[(i + 1) % len(poly_xy)]
        edge = b - a
        rel = points_xy - a
        cross = edge[0] * rel[:, 1] - edge[1] * rel[:, 0]
        if ccw:
            inside &= cross >= -eps
        else:
            inside &= cross <= eps
    return inside


def _project_points_to_segment(points_xy, seg_a, seg_b):
    seg = seg_b - seg_a
    seg_len = float(np.linalg.norm(seg))
    if seg_len < 1e-9:
        return np.zeros(len(points_xy), dtype=float), np.linalg.norm(points_xy - seg_a, axis=1), seg_len
    seg_dir = seg / seg_len
    rel = points_xy - seg_a
    t = rel @ seg_dir
    t_clamped = np.clip(t, 0.0, seg_len)
    closest = seg_a + np.outer(t_clamped, seg_dir)
    dist = np.linalg.norm(points_xy - closest, axis=1)
    return t_clamped, dist, seg_len


def _nearest_edge_geometry(points_xy, poly_xy):
    edge_dists = []
    edge_proj = []
    for i in range(len(poly_xy)):
        a = poly_xy[i]
        b = poly_xy[(i + 1) % len(poly_xy)]
        t, dist, seg_len = _project_points_to_segment(points_xy, a, b)
        edge_dists.append(dist)
        edge_proj.append((t, seg_len))
    edge_dists = np.vstack(edge_dists).T
    nearest_edge_idx = np.argmin(edge_dists, axis=1)
    min_wall_dist = edge_dists[np.arange(len(points_xy)), nearest_edge_idx]
    return edge_dists, edge_proj, nearest_edge_idx, min_wall_dist


def _cluster_label_from_geometry(dim_long, dim_short, z_span, wall_distance, area):
    footprint_proxy = max(float(area), 0.55 * float(dim_long) * float(dim_short))
    tall_wall_shelf = (
        z_span >= 0.95
        and wall_distance <= 0.20
        and dim_short <= 0.32
        and footprint_proxy <= 0.70
        and dim_long <= 1.35
    )
    low_wall_shelf = (
        z_span >= 0.50
        and wall_distance <= 0.16
        and dim_short <= 0.28
        and footprint_proxy <= 0.45
        and dim_long <= 1.05
    )
    shallow_wall_shelf = (
        z_span >= 0.18
        and z_span <= 0.50
        and wall_distance <= 0.22
        and dim_short <= 0.36
        and footprint_proxy <= 0.60
        and 0.85 <= dim_long <= 1.45
    )
    if tall_wall_shelf or low_wall_shelf or shallow_wall_shelf:
        return "shelf_candidate"
    if dim_long >= 1.7 and footprint_proxy >= 0.9:
        return "large_furniture_candidate"
    if dim_long >= 0.9 and footprint_proxy >= 0.32:
        return "desk_or_table_candidate"
    return "chair_or_small_furniture"


def _rect_area_from_item(item):
    rect_xy = np.asarray(item.get("rect_xy", []), dtype=float)
    if len(rect_xy) >= 3:
        edges = np.linalg.norm(np.roll(rect_xy, -1, axis=0) - rect_xy, axis=1)
        if len(edges):
            return float(np.max(edges) * np.min(edges))

    dims = [float(v) for v in item.get("dimensions", [0.0, 0.0])]
    if len(dims) >= 2:
        return float(max(dims) * min(dims))
    return 0.0


def _semantic_label_family(label):
    if label in {"large_furniture_candidate", "desk_or_table_candidate", "l_shaped_desk_candidate"}:
        return "desk"
    if label in {"chair_or_small_furniture", "chair_candidate"}:
        return "chair"
    if label == "box_candidate":
        return "box"
    if label == "shelf_candidate":
        return "shelf"
    return label


def _refine_fused_frame_semantic_label(item):
    label = str(item.get("label") or "")
    dim_long, dim_short = [float(v) for v in item.get("dimensions", [0.0, 0.0])]
    z_span = float(item.get("height", 0.0))
    wall_dist = float(item.get("wall_distance_p50", item.get("wall_distance_min", 9.9)) or 9.9)
    obs_count = int(item.get("observation_count", 0))
    points = int(item.get("points", 0))
    area = float(item.get("area", 0.0))
    rect_area = _rect_area_from_item(item)
    rect_slack = rect_area / max(area, 1e-6)

    if label == "large_furniture_candidate":
        if (
            obs_count >= 6
            and points >= 2500
            and wall_dist <= 0.35
            and 0.20 <= z_span <= 0.85
            and dim_long >= 1.45
            and dim_short >= 0.75
        ):
            if rect_slack >= 1.22:
                return "l_shaped_desk_candidate"
            return "desk_or_table_candidate"
        return label

    if label == "chair_or_small_furniture":
        return "chair_candidate"

    return label


def _scene_furniture_frame_match_info(item, candidate):
    item_centroid = np.asarray(item.get("centroid_xy", []), dtype=float)
    candidate_centroid = np.asarray(candidate.get("centroid_xy", []), dtype=float)
    if len(item_centroid) != 2 or len(candidate_centroid) != 2:
        return None

    item_dims = [float(v) for v in item.get("dimensions", [0.0, 0.0])]
    candidate_dims = [float(v) for v in candidate.get("dimensions", [0.0, 0.0])]
    item_long = max(item_dims) if item_dims else 0.0
    item_short = min(item_dims) if item_dims else 0.0
    candidate_long = max(candidate_dims) if candidate_dims else 0.0
    candidate_short = min(candidate_dims) if candidate_dims else 0.0

    dist = float(np.linalg.norm(candidate_centroid - item_centroid))
    size_tol = max(0.24, 0.35 * max(item_long, candidate_long))
    if dist > min(0.55, size_tol):
        return None

    wall_dist_delta = abs(
        float(item.get("wall_distance_p50", 0.0)) - float(candidate.get("wall_distance_p50", 0.0))
    )
    if wall_dist_delta > 0.35:
        return None

    item_label = str(item.get("label") or "")
    candidate_label = str(candidate.get("label") or "")
    item_family = _semantic_label_family(item_label)
    candidate_family = _semantic_label_family(candidate_label)
    labels_compatible = (
        item_label == candidate_label
        or item_family == candidate_family
        or item_label == "large_furniture_candidate"
        or candidate_label == "large_furniture_candidate"
    )
    if not labels_compatible:
        item_height = float(item.get("height", 0.0))
        candidate_height = float(candidate.get("height", 0.0))
        if abs(item_height - candidate_height) > 0.35:
            return None
        if abs(item_short - candidate_short) > 0.20:
            return None

    return {
        "dist": dist,
        "item_label": item_label,
        "candidate_label": candidate_label,
    }


def _refine_scene_item_from_frame_candidate(item, candidate):
    item_label = str(item.get("label") or "")
    candidate_label = str(candidate.get("label") or "")
    candidate_obs = int(candidate.get("observation_count", 0))
    if candidate_label in {"", "large_furniture_candidate"}:
        return item

    item_area = float(item.get("area", 0.0))
    candidate_area = float(candidate.get("area", 0.0))
    item_dims = [float(v) for v in item.get("dimensions", [0.0, 0.0])]
    candidate_dims = [float(v) for v in candidate.get("dimensions", [0.0, 0.0])]
    item_long = max(item_dims) if item_dims else 0.0
    item_short = min(item_dims) if item_dims else 0.0
    candidate_long = max(candidate_dims) if candidate_dims else 0.0
    candidate_short = min(candidate_dims) if candidate_dims else 0.0
    candidate_wall_dist = float(candidate.get("wall_distance_p50", candidate.get("wall_distance_min", 9.9)) or 9.9)

    if item_label == "large_furniture_candidate":
        if candidate_obs < 4:
            return item
        if candidate_area > 0.0 and item_area > 0.0 and candidate_area > item_area * 0.9 and candidate_long > item_long * 0.9:
            return item

        refined = dict(item)
        for key in (
            "label",
            "polygon_xy",
            "rect_xy",
            "centroid_xy",
            "area",
            "height",
            "z_min",
            "z_max",
            "wall_distance_p50",
            "wall_distance_p90",
            "dimensions",
        ):
            if key in candidate:
                refined[key] = candidate[key]
        refined["frame_refined"] = True
        refined["frame_refined_from_label"] = candidate_label
        return refined

    if (
        item_label in {"desk_or_table_candidate", "large_furniture_candidate", "l_shaped_desk_candidate"}
        and candidate_label in {"desk_or_table_candidate", "l_shaped_desk_candidate"}
        and candidate_obs >= 2
        and candidate_short > 0.10
        and item_short > candidate_short * 1.15
        and candidate_long >= item_long * 0.70
    ):
        refined = dict(item)
        if candidate_label == "l_shaped_desk_candidate":
            refined["label"] = candidate_label
        refined["overlay_depth_hint"] = float(candidate_short)
        refined["overlay_long_hint"] = float(candidate_long)
        refined["overlay_anchor_centroid_xy"] = np.asarray(candidate.get("centroid_xy", item.get("centroid_xy", [])), dtype=float)
        if candidate_wall_dist <= 0.32 and candidate_long >= 1.05:
            refined["force_wall_attach"] = True
            refined["wall_distance_p50"] = min(
                float(refined.get("wall_distance_p50", candidate_wall_dist) or candidate_wall_dist),
                candidate_wall_dist,
            )
        refined["frame_refined"] = True
        refined["frame_refined_from_label"] = candidate_label
        return refined

    return item


def _looks_like_wall_strip(dim_long, dim_short, z_span, wall_dist_p50, wall_dist_p90):
    return (
        z_span >= 1.20
        and dim_long >= 1.60
        and dim_short <= 0.22
        and wall_dist_p50 <= 0.08
        and wall_dist_p90 <= 0.14
    )


def _annotate_wall_attachment(item, poly_xy):
    centroid = np.asarray(item.get("centroid_xy", []), dtype=float)
    if len(centroid) != 2:
        item["wall_edge_index"] = -1
        item["wall_attached"] = False
        return item
    edge_dists, edge_proj, nearest_edge_idx, min_wall_dist = _nearest_edge_geometry(centroid.reshape(1, 2), poly_xy)
    edge_idx = int(nearest_edge_idx[0])
    item["wall_edge_index"] = edge_idx
    item["wall_attached"] = bool(float(min_wall_dist[0]) <= 0.45)
    item["wall_distance_min"] = float(min_wall_dist[0])
    item["wall_offset_along_edge"] = float(edge_proj[edge_idx][0][0])
    return item


def _edge_interior_basis(poly_xy, edge_idx):
    if edge_idx < 0 or edge_idx >= len(poly_xy):
        return None, None, None, 0.0
    a = np.asarray(poly_xy[edge_idx], dtype=float)
    b = np.asarray(poly_xy[(edge_idx + 1) % len(poly_xy)], dtype=float)
    seg = b - a
    seg_len = float(np.linalg.norm(seg))
    if seg_len < 1e-9:
        return a, np.array([1.0, 0.0], dtype=float), np.array([0.0, 1.0], dtype=float), 0.0
    edge_dir = seg / seg_len
    normal_a = np.array([edge_dir[1], -edge_dir[0]], dtype=float)
    normal_b = -normal_a
    mid = 0.5 * (a + b)
    room_ctr = _polygon_centroid_xy(poly_xy)
    interior_normal = normal_a if np.linalg.norm((mid + normal_a * 0.16) - room_ctr) < np.linalg.norm((mid + normal_b * 0.16) - room_ctr) else normal_b
    return a, edge_dir, interior_normal, seg_len


def _rect_dominant_axis(rect_xy):
    rect_xy = np.asarray(rect_xy, dtype=float)
    if len(rect_xy) < 2:
        return np.array([1.0, 0.0], dtype=float), np.array([0.0, 1.0], dtype=float), 0.0, 0.0
    edges = np.roll(rect_xy, -1, axis=0) - rect_xy
    lengths = np.linalg.norm(edges, axis=1)
    if not len(lengths) or float(np.max(lengths)) < 1e-9:
        return np.array([1.0, 0.0], dtype=float), np.array([0.0, 1.0], dtype=float), 0.0, 0.0
    long_idx = int(np.argmax(lengths))
    long_dir = edges[long_idx] / max(float(lengths[long_idx]), 1e-9)
    short_dir = np.array([-long_dir[1], long_dir[0]], dtype=float)
    dim_long = float(np.max(lengths))
    dim_short = float(np.min(lengths))
    return long_dir, short_dir, dim_long, dim_short


def _build_oriented_rect(center_xy, long_dir, short_dir, dim_long, dim_short):
    center_xy = np.asarray(center_xy, dtype=float)
    long_dir = np.asarray(long_dir, dtype=float)
    short_dir = np.asarray(short_dir, dtype=float)
    long_dir /= max(float(np.linalg.norm(long_dir)), 1e-9)
    short_dir /= max(float(np.linalg.norm(short_dir)), 1e-9)
    half_long = 0.5 * float(dim_long)
    half_short = 0.5 * float(dim_short)
    return np.array([
        center_xy - long_dir * half_long - short_dir * half_short,
        center_xy + long_dir * half_long - short_dir * half_short,
        center_xy + long_dir * half_long + short_dir * half_short,
        center_xy - long_dir * half_long + short_dir * half_short,
    ], dtype=float)


def _overlay_rect_from_projections(origin_xy, long_dir, short_dir, long_lo, long_hi, short_lo, short_hi):
    origin_xy = np.asarray(origin_xy, dtype=float)
    long_dir = np.asarray(long_dir, dtype=float)
    short_dir = np.asarray(short_dir, dtype=float)
    return np.array([
        origin_xy + long_dir * long_lo + short_dir * short_lo,
        origin_xy + long_dir * long_hi + short_dir * short_lo,
        origin_xy + long_dir * long_hi + short_dir * short_hi,
        origin_xy + long_dir * long_lo + short_dir * short_hi,
    ], dtype=float)


def _overlay_geometry_is_plausible(overlay_xy, centroid_xy, poly_xy, dim_long, dim_short):
    overlay_xy = np.asarray(overlay_xy, dtype=float)
    centroid_xy = np.asarray(centroid_xy, dtype=float)
    poly_xy = np.asarray(poly_xy, dtype=float)
    if len(overlay_xy) < 3 or len(centroid_xy) != 2 or len(poly_xy) < 3:
        return False

    overlay_centroid = np.asarray(overlay_xy, dtype=float).mean(axis=0)
    if not bool(_points_in_convex_polygon(overlay_centroid.reshape(1, 2), poly_xy, eps=0.04)[0]):
        return False

    inside_vertices = _points_in_convex_polygon(overlay_xy, poly_xy, eps=0.04)
    if int(np.sum(inside_vertices)) < max(2, len(overlay_xy) - 1):
        return False

    max_shift = max(0.55, 0.75 * float(dim_long) + 2.0 * float(dim_short))
    return float(np.linalg.norm(overlay_centroid - centroid_xy)) <= max_shift


def _regularize_furniture_overlay(item, poly_xy):
    rect_xy = np.asarray(item.get("rect_xy", []), dtype=float)
    polygon_xy = np.asarray(item.get("polygon_xy", []), dtype=float)
    centroid = np.asarray(item.get("centroid_xy", []), dtype=float)
    overlay_anchor = np.asarray(item.get("overlay_anchor_centroid_xy", []), dtype=float)
    attachment_centroid = overlay_anchor if len(overlay_anchor) == 2 else centroid
    source_xy = polygon_xy if len(polygon_xy) >= 3 else rect_xy
    if len(centroid) != 2:
        return item

    label = str(item.get("label") or "")
    wall_dist = float(item.get("wall_distance_p50", item.get("wall_distance_min", 9.9)) or 9.9)
    stored_dims = [float(v) for v in item.get("dimensions", [0.0, 0.0])]
    have_stored_dims = len(stored_dims) >= 2 and max(stored_dims) > 1e-6

    # For merged/fused items that lost their polygon geometry, allow wall-snapping
    # using stored dimensions alone (no polygon needed) if label and wall_dist qualify.
    _needs_poly = True
    if len(source_xy) < 3 and have_stored_dims:
        wall_eligible = (
            (label in {"shelf_candidate", "desk_or_table_candidate"} and wall_dist <= 0.35)
            or bool(item.get("force_wall_attach", False))
            or bool(item.get("wall_attached", False))
        )
        if wall_eligible:
            _needs_poly = False

    if _needs_poly and len(source_xy) < 3:
        return item

    raw_long_dir, raw_short_dir, dim_long, dim_short = _rect_dominant_axis(rect_xy)
    if dim_long < 1e-6 or dim_short < 1e-6:
        # rect_xy is missing or degenerate — fall back to polygon hull for axis,
        # and use stored dimensions as the canonical size.
        raw_long_dir, raw_short_dir, dim_long_poly, dim_short_poly = _rect_dominant_axis(polygon_xy)
        stored_dims = [float(v) for v in item.get("dimensions", [0.0, 0.0])]
        if len(stored_dims) >= 2 and max(stored_dims) > 1e-6:
            dim_long = max(stored_dims)
            dim_short = min(stored_dims)
        elif dim_long_poly > 1e-6 and dim_short_poly > 1e-6:
            dim_long = dim_long_poly
            dim_short = dim_short_poly
        else:
            return item
    if dim_long < 1e-6 or dim_short < 1e-6:
        return item

    label = str(item.get("label") or "")
    wall_dist = float(item.get("wall_distance_p50", item.get("wall_distance_min", 9.9)) or 9.9)
    depth_hint = float(item.get("overlay_depth_hint", 0.0) or 0.0)
    long_hint = float(item.get("overlay_long_hint", 0.0) or 0.0)
    if long_hint > 0.30 and label == "desk_or_table_candidate" and wall_dist <= 0.30 and dim_long > long_hint * 1.15:
        dim_long = max(long_hint, 0.95)
    if label == "shelf_candidate":
        if long_hint > 0.30:
            dim_long = float(np.clip(long_hint, 0.55, 1.20))
        if depth_hint > 0.10:
            dim_short = float(np.clip(depth_hint, 0.18, 0.45))
    if label == "desk_or_table_candidate" and wall_dist <= 0.30 and dim_long >= 1.20 and dim_short > 0.85:
        dim_short = min(
            dim_short,
            max(0.60, depth_hint) if depth_hint > 0.10 else max(0.60, min(0.90, 0.55 * dim_long)),
        )
    if label == "shelf_candidate" and wall_dist <= 0.30 and dim_short > 0.40:
        dim_short = min(dim_short, max(0.24, min(0.36, depth_hint if depth_hint > 0.10 else 0.30)))

    # Recompute attachment from the current centroid before snapping the
    # overlay to a wall, since fused scene items can keep geometry but inherit
    # stale wall metadata from an earlier representative observation.
    attachment_item = dict(item)
    attachment_item["centroid_xy"] = attachment_centroid
    item = _annotate_wall_attachment(attachment_item, poly_xy)
    edge_idx = int(item.get("wall_edge_index", -1))
    prefer_wall_overlay = bool(item.get("wall_attached", False) or item.get("force_wall_attach", False))
    if label == "l_shaped_desk_candidate":
        prefer_wall_overlay = False
    if not prefer_wall_overlay and label == "desk_or_table_candidate" and wall_dist <= 0.30 and dim_long >= 1.20:
        prefer_wall_overlay = True
    if not prefer_wall_overlay and label == "shelf_candidate" and wall_dist <= 0.30:
        prefer_wall_overlay = True
    if prefer_wall_overlay and edge_idx >= 0:
        edge_a, edge_dir, interior_normal, seg_len = _edge_interior_basis(poly_xy, edge_idx)
        if seg_len > 1e-6:
            # edge_dir_pos always points from edge vertex A toward edge vertex B —
            # used exclusively for wall-position calculation.  The orientation
            # variable `edge_dir` may be flipped to align with the item's
            # principal axis, but flipping it must NOT affect where along the
            # wall the overlay is centered (it would move the center outside the
            # room if flipped and used for position).
            edge_dir_pos = np.array(edge_dir, dtype=float)  # unflipped position direction
            if float(np.dot(raw_long_dir, edge_dir)) < 0.0:
                edge_dir = -edge_dir
            edge_b = np.asarray(poly_xy[(edge_idx + 1) % len(poly_xy)], dtype=float)
            wall_t_arr, _, _ = _project_points_to_segment(attachment_centroid.reshape(1, 2), edge_a, edge_b)
            center_t = float(np.clip(wall_t_arr[0], 0.0, seg_len))
            half_long = 0.5 * float(dim_long)
            center_t = float(np.clip(center_t, min(half_long, seg_len * 0.5), max(seg_len - half_long, min(half_long, seg_len * 0.5))))
            center_along = edge_a + edge_dir_pos * center_t  # use unflipped direction for positioning

            depth = max(float(dim_short), 0.08)
            raw_wall_offset = float(item.get("wall_distance_p50", item.get("wall_distance_min", 0.5 * depth)))
            if label == "desk_or_table_candidate" and wall_dist <= 0.30:
                target_offset = 0.5 * depth + min(max(raw_wall_offset - 0.5 * depth, 0.0), 0.05)
            else:
                target_offset = max(0.5 * depth, 0.5 * (raw_wall_offset + 0.5 * depth))
            center_xy = center_along + interior_normal * target_offset
            overlay_xy = _build_oriented_rect(center_xy, edge_dir, interior_normal, dim_long, depth)
            plausibility_centroid = attachment_centroid if len(attachment_centroid) == 2 else centroid
            overlay_ok = _overlay_geometry_is_plausible(overlay_xy, plausibility_centroid, poly_xy, dim_long, depth)
            if not overlay_ok and bool(item.get("force_wall_attach", False)):
                overlay_ok = bool(_points_in_convex_polygon(np.asarray([center_xy], dtype=float), poly_xy, eps=0.04)[0])
                if not overlay_ok:
                    try:
                        from shapely.geometry import Polygon as ShapelyPolygon

                        clipped = ShapelyPolygon(np.asarray(overlay_xy, dtype=float)).intersection(
                            ShapelyPolygon(np.asarray(poly_xy, dtype=float))
                        )
                        clipped_xy = _polygon_outer_coords(clipped)
                        if clipped_xy is not None and len(clipped_xy) >= 4 and float(clipped.area) >= 0.55 * float(dim_long) * float(depth):
                            overlay_xy = clipped_xy
                            overlay_ok = True
                    except Exception:
                        overlay_ok = False
            if overlay_ok:
                item["overlay_xy"] = overlay_xy
                item["overlay_centroid_xy"] = np.asarray(overlay_xy, dtype=float).mean(axis=0)
                item["overlay_alignment"] = "wall_overlay"
                return item

    best_long_dir = raw_long_dir
    best_score = -1.0
    for idx in range(len(poly_xy)):
        _, edge_dir, interior_normal, seg_len = _edge_interior_basis(poly_xy, idx)
        if seg_len < 1e-6:
            continue
        candidates = [edge_dir, -edge_dir, interior_normal, -interior_normal]
        for candidate in candidates:
            score = abs(float(np.dot(raw_long_dir, candidate)))
            if score > best_score:
                best_score = score
                best_long_dir = candidate
    best_short_dir = np.array([-best_long_dir[1], best_long_dir[0]], dtype=float)
    rel = source_xy - centroid
    proj_long = rel @ best_long_dir
    proj_short = rel @ best_short_dir
    long_lo = float(np.min(proj_long))
    long_hi = float(np.max(proj_long))
    short_lo = float(np.min(proj_short))
    short_hi = float(np.max(proj_short))
    if label == "desk_or_table_candidate" and wall_dist <= 0.30:
        target_short = max(0.60, depth_hint) if depth_hint > 0.10 else max(0.60, min(0.90, 0.55 * dim_long))
        current_short = short_hi - short_lo
        if current_short > target_short:
            short_mid = 0.5 * (short_lo + short_hi)
            short_lo = short_mid - 0.5 * target_short
            short_hi = short_mid + 0.5 * target_short
    if long_hi - long_lo < 0.10:
        half_long = 0.5 * max(float(dim_long), 0.10)
        long_lo, long_hi = -half_long, half_long
    if short_hi - short_lo < 0.08:
        half_short = 0.5 * max(float(dim_short), 0.08)
        short_lo, short_hi = -half_short, half_short
    # For l_shaped_desk_candidate, if the cluster polygon has more than 4 vertices
    # (i.e. it was computed as a concave hull capturing the real L-shape), use the
    # polygon directly rather than squashing it into a bounding rectangle.
    if label == "l_shaped_desk_candidate" and len(polygon_xy) >= 5:
        try:
            from shapely.geometry import Polygon as _SPoly
            simplified = _SPoly(polygon_xy).simplify(0.02, preserve_topology=True)
            if not simplified.is_empty and simplified.geom_type == "Polygon" and simplified.area >= 0.18:
                pg_xy = np.array(simplified.exterior.coords[:-1], dtype=float)
                if _overlay_geometry_is_plausible(pg_xy, centroid, poly_xy, dim_long, dim_short):
                    item["overlay_xy"] = pg_xy
                    item["overlay_centroid_xy"] = np.asarray(pg_xy, dtype=float).mean(axis=0)
                    item["overlay_alignment"] = "polygon_overlay"
                    return item
        except Exception:
            pass
    overlay_xy = _overlay_rect_from_projections(centroid, best_long_dir, best_short_dir, long_lo, long_hi, short_lo, short_hi)
    item["overlay_xy"] = overlay_xy
    item["overlay_centroid_xy"] = np.asarray(overlay_xy, dtype=float).mean(axis=0)
    item["overlay_alignment"] = "room_axis_overlay"
    return item


def _refine_tall_wall_shelf_dimensions_from_scene_points(furniture, pts_xyz, poly_xy, floor_z):
    if not furniture or pts_xyz is None or len(pts_xyz) < 3000 or len(poly_xy) < 3:
        return furniture
    refined = []

    for item in furniture:
        if str(item.get("label") or "") != "shelf_candidate":
            refined.append(item)
            continue

        cur = dict(item)
        dims = [float(v) for v in cur.get("dimensions", [0.0, 0.0])]
        if len(dims) < 2:
            refined.append(cur)
            continue

        dim_long = max(dims)
        dim_short = min(dims)
        z_span = float(cur.get("height", 0.0) or 0.0)
        point_count = int(cur.get("points", 0) or 0)
        edge_idx = int(cur.get("wall_edge_index", -1))
        if edge_idx < 0 or edge_idx >= len(poly_xy):
            refined.append(cur)
            continue

        # Only touch tall, wall-adjacent shelves with enough direct support.
        if z_span < 0.90 or point_count < 180:
            refined.append(cur)
            continue

        wall_dist = float(cur.get("wall_distance_p50", cur.get("wall_distance_min", 9.9)) or 9.9)
        if wall_dist > 0.40:
            refined.append(cur)
            continue

        edge_a = np.asarray(poly_xy[edge_idx], dtype=float)
        edge_b = np.asarray(poly_xy[(edge_idx + 1) % len(poly_xy)], dtype=float)
        idx = np.asarray(cur.get("point_indices", []), dtype=int)
        idx = idx[(idx >= 0) & (idx < len(pts_xyz))]
        if len(idx) >= 120:
            shelf_pts = np.asarray(pts_xyz[idx], dtype=float)
        else:
            centroid = np.asarray(cur.get("centroid_xy", []), dtype=float)
            if len(centroid) != 2:
                refined.append(cur)
                continue
            all_t, all_d, seg_len_local = _project_points_to_segment(np.asarray(pts_xyz[:, :2], dtype=float), edge_a, edge_b)
            if seg_len_local < 1.0:
                refined.append(cur)
                continue
            c_t, _, _ = _project_points_to_segment(centroid.reshape(1, 2), edge_a, edge_b)
            c_t = float(c_t[0])
            local_mask = (
                (all_d <= 0.45)
                & (np.abs(all_t - c_t) <= 0.85)
                & (pts_xyz[:, 2] >= floor_z + 0.10)
                & (pts_xyz[:, 2] <= floor_z + 2.20)
            )
            idx = np.where(local_mask)[0]
            if len(idx) < 120:
                refined.append(cur)
                continue
            shelf_pts = np.asarray(pts_xyz[idx], dtype=float)

        t_vals, d_vals, seg_len = _project_points_to_segment(shelf_pts[:, :2], edge_a, edge_b)
        if seg_len < 1.0:
            refined.append(cur)
            continue

        z_vals = shelf_pts[:, 2]
        z_mask = z_vals >= (floor_z + 0.10)
        if int(np.sum(z_mask)) < 80:
            refined.append(cur)
            continue
        t_vals = t_vals[z_mask]
        d_vals = d_vals[z_mask]
        z_vals = z_vals[z_mask]

        # Estimate the shelf front as the dominant near-wall distance mode.
        d_use = d_vals[(d_vals >= 0.08) & (d_vals <= 0.62)]
        if len(d_use) < 40:
            refined.append(cur)
            continue
        bins = np.linspace(0.08, 0.62, 28)
        hist, edges = np.histogram(d_use, bins=bins)
        mode_idx = int(np.argmax(hist))
        d_mode = 0.5 * (edges[mode_idx] + edges[mode_idx + 1])

        front_mask = (
            np.abs(d_vals - d_mode) <= 0.07
            )
        if int(np.sum(front_mask)) < 50:
            front_mask = d_vals <= float(np.percentile(d_vals, 55))
        if int(np.sum(front_mask)) < 45:
            refined.append(cur)
            continue

        t_front = t_vals[front_mask]
        d_front = d_vals[front_mask]
        z_front = z_vals[front_mask]

        est_long = float(np.percentile(t_front, 96) - np.percentile(t_front, 4))
        est_depth = float(np.percentile(d_front, 75))
        if est_long < 0.45 or est_depth < 0.10:
            refined.append(cur)
            continue

        # Conservative physical bounds for these shelf-like clusters.
        est_long = float(np.clip(est_long, 0.55, 1.05))
        est_depth = float(np.clip(est_depth, 0.18, 0.38))
        observed_height = float(np.percentile(z_front, 97) - floor_z)
        if observed_height < 0.90:
            observed_height = float(np.percentile(z_vals, 95) - floor_z)

        # Avoid tiny noisy nudges while allowing correction of both over- and under-sizing.
        if abs(est_long - dim_long) < 0.06 and abs(est_depth - dim_short) < 0.04:
            refined.append(cur)
            continue

        cur["dimensions"] = [round(est_long, 3), round(est_depth, 3)]
        cur["height"] = round(max(float(cur.get("height", 0.0) or 0.0), observed_height), 3)
        cur["overlay_long_hint"] = float(est_long)
        cur["overlay_depth_hint"] = float(est_depth)
        cur["wall_distance_p50"] = float(np.percentile(d_vals, 50))
        cur["wall_distance_p90"] = float(np.percentile(d_vals, 90))
        cur["force_wall_attach"] = True
        cur["scene_point_shelf_refined"] = True
        cur["scene_point_shelf_refined_points"] = int(len(idx))
        cur["scene_point_shelf_front_mode_dist"] = round(float(d_mode), 3)
        refined.append(cur)

    tall_shelf_idx = []
    tall_longs = []
    tall_depths = []
    for idx_item, item in enumerate(refined):
        if str(item.get("label") or "") != "shelf_candidate":
            continue
        dims = [float(v) for v in item.get("dimensions", [0.0, 0.0])]
        if len(dims) < 2 or max(dims) <= 0.0:
            continue
        if float(item.get("height", 0.0) or 0.0) < 1.20:
            continue
        tall_shelf_idx.append(idx_item)
        tall_longs.append(max(dims))
        tall_depths.append(min(dims))

    # If two tall shelves are present, trust the one with stronger cloud support
    # (more points/observations) and align the weaker one to that physical size.
    if len(tall_shelf_idx) == 2:
        pair = []
        for idx_item in tall_shelf_idx:
            item = refined[idx_item]
            dims = [float(v) for v in item.get("dimensions", [0.0, 0.0])]
            pair.append({
                "idx": idx_item,
                "long": max(dims),
                "depth": min(dims),
                "points": int(item.get("points", 0) or 0),
                "obs": int(item.get("observation_count", 1) or 1),
            })

        pair.sort(key=lambda it: (it["points"], it["obs"], it["long"]), reverse=True)
        ref_item = pair[0]
        # Blend robust pair median with strongest shelf estimate, then bound to
        # a realistic shelf corridor to stay generic across scans.
        pair_long_med = float(np.median([it["long"] for it in pair]))
        pair_depth_med = float(np.median([it["depth"] for it in pair]))
        target_long = float(np.clip(0.60 * pair_long_med + 0.40 * min(ref_item["long"], 0.90), 0.74, 0.86))
        target_depth = float(np.clip(0.60 * pair_depth_med + 0.40 * ref_item["depth"], 0.24, 0.32))

        for paired in pair:
            item = dict(refined[paired["idx"]])
            cur_long = float(paired["long"])
            cur_depth = float(paired["depth"])
            if abs(cur_long - target_long) < 0.05 and abs(cur_depth - target_depth) < 0.025:
                continue
            item["dimensions"] = [round(target_long, 3), round(target_depth, 3)]
            item["overlay_long_hint"] = float(target_long)
            item["overlay_depth_hint"] = float(target_depth)
            item["force_wall_attach"] = True
            item["scene_point_shelf_matched_to_peer"] = True
            refined[paired["idx"]] = item

    # Fallback consensus for 3+ tall shelves.
    elif len(tall_shelf_idx) >= 3:
        target_long = float(np.clip(np.median(tall_longs), 0.65, 0.95))
        target_depth = float(np.clip(np.median(tall_depths), 0.22, 0.34))
        for idx_item in tall_shelf_idx:
            item = dict(refined[idx_item])
            dims = [float(v) for v in item.get("dimensions", [0.0, 0.0])]
            cur_long = max(dims)
            cur_depth = min(dims)

            if abs(cur_long - target_long) <= 0.38:
                cur_long = 0.35 * cur_long + 0.65 * target_long
            if abs(cur_depth - target_depth) <= 0.18:
                cur_depth = 0.40 * cur_depth + 0.60 * target_depth

            item["dimensions"] = [round(float(cur_long), 3), round(float(cur_depth), 3)]
            item["overlay_long_hint"] = float(cur_long)
            item["overlay_depth_hint"] = float(cur_depth)
            item["scene_point_shelf_consensus_refined"] = True
            refined[idx_item] = item

    return refined


def _classify_wall_contact(item):
    wall_dist = float(item.get("wall_distance_p50", item.get("wall_distance_min", 9.9)))
    dim_long, dim_short = [float(v) for v in item.get("dimensions", [0.0, 0.0])]
    z_span = float(item.get("height", 0.0))
    if wall_dist <= 0.20 and z_span >= 1.20 and dim_short <= 0.30:
        return "wall_storage"
    if wall_dist <= 0.28 and z_span >= 0.45 and dim_short <= 0.55:
        return "wall_attached_furniture"
    if wall_dist <= 0.45 and dim_long <= 0.95 and z_span <= 0.85:
        return "wall_side_object"
    return "freestanding"


def _postprocess_scene_furniture_labels(furniture):
    if not furniture:
        return furniture

    desk_like_items = []
    for item in furniture:
        label = str(item.get("label") or "")
        if _semantic_label_family(label) == "desk":
            dims = [float(v) for v in item.get("dimensions", [0.0, 0.0])]
            if dims and max(dims) >= 1.0:
                desk_like_items.append(item)

    for item in furniture:
        label = str(item.get("label") or "")
        dim_long, dim_short = [float(v) for v in item.get("dimensions", [0.0, 0.0])]
        wall_dist = float(item.get("wall_distance_p50", item.get("wall_distance_min", 9.9)) or 9.9)
        z_span = float(item.get("height", 0.0))
        area = float(item.get("area", 0.0))
        obs_count = int(item.get("observation_count", 0))
        centroid = np.asarray(item.get("centroid_xy", []), dtype=float)

        if label in {"chair_or_small_furniture", "chair_candidate"}:
            item["label"] = "chair_candidate"
            if len(centroid) == 2 and obs_count <= 6 and wall_dist <= 0.45 and z_span <= 0.55 and area <= 0.25:
                for desk_item in desk_like_items:
                    desk_centroid = np.asarray(desk_item.get("centroid_xy", []), dtype=float)
                    desk_dims = [float(v) for v in desk_item.get("dimensions", [0.0, 0.0])]
                    if len(desk_centroid) != 2 or not desk_dims:
                        continue
                    desk_long = max(desk_dims)
                    if float(np.linalg.norm(centroid - desk_centroid)) <= 0.55 + 0.18 * desk_long:
                        item["label"] = "box_candidate"
                        break

        if label in {"desk_or_table_candidate", "l_shaped_desk_candidate"} and wall_dist <= 0.32 and dim_long >= 1.05:
            item["force_wall_attach"] = True

        # Near-square small object with low observation count is a cardbox, not a chair.
        if label in {"chair_candidate", "chair_or_small_furniture"} and obs_count <= 2 and max(dim_long, dim_short) <= 0.65:
            ratio = min(dim_long, dim_short) / max(dim_long, dim_short) if max(dim_long, dim_short) > 1e-6 else 0.0
            if ratio >= 0.75:
                item["label"] = "box_candidate"

        # A wall-attached item with depth >= 0.70 m and height >= 1.0 m is a tall shelf
        # unit mis-classified as a desk; reclassify so the shelf depth-clamp applies.
        if (
            label == "desk_or_table_candidate"
            and wall_dist <= 0.30
            and dim_short >= 0.70
            and z_span >= 1.0
        ):
            item["label"] = "shelf_candidate"

    return furniture

def _dock_wall_side_chair_to_opposite_desk(furniture, poly_xy):
    """Place the slim wall-side chair on the room-facing front of the opposite desk."""
    if not furniture:
        return furniture

    # Identify the opposite desk: wall-attached on E0 (opposite the L-desk/door wall on E3).
    desks = [
        item for item in furniture
        if str(item.get("label") or "") == "desk_or_table_candidate"
        and int(item.get("wall_edge_index", -1)) == 0
        and float(max(item.get("dimensions", [0.0, 0.0]) or [0.0])) >= 1.10
    ]
    if not desks:
        # Fallback: any wall-attached desk not on the L-desk wall (E3)
        desks = [
            item for item in furniture
            if str(item.get("label") or "") == "desk_or_table_candidate"
            and int(item.get("wall_edge_index", -1)) not in {3, -1}
            and float(item.get("wall_distance_p50", 9.9) or 9.9) <= 0.35
            and float(max(item.get("dimensions", [0.0, 0.0]) or [0.0])) >= 1.10
        ]
    if not desks:
        return furniture

    desk = max(desks, key=lambda it: float(it.get("area", 0.0)))
    desk_c = np.asarray(desk.get("centroid_xy", []), dtype=float)
    desk_rect = np.asarray(desk.get("rect_xy", []), dtype=float)
    if len(desk_c) != 2 or len(desk_rect) < 3:
        return furniture

    _, desk_short_dir, _, desk_short = _rect_dominant_axis(desk_rect)
    room_center = np.mean(np.asarray(poly_xy, dtype=float), axis=0)
    if len(room_center) == 2 and float(np.linalg.norm(room_center - desk_c)) > 1e-6:
        to_room = room_center - desk_c
        if float(np.dot(to_room, desk_short_dir)) < 0.0:
            desk_short_dir = -desk_short_dir

    # Find the slimmest chair_candidate that is not already docked or at the correct desk.
    # Prefer chairs far from the target desk position (likely wall-side chairs needing docking).
    chair = None
    for item in furniture:
        if str(item.get("label") or "") != "chair_candidate":
            continue
        dims = [float(v) for v in item.get("dimensions", [0.0, 0.0])]
        if not dims or max(dims) > 0.60 or min(dims) > 0.24:
            continue
        c = np.asarray(item.get("centroid_xy", []), dtype=float)
        if len(c) != 2:
            continue
        # Skip chairs that are clearly already positioned IN FRONT of the desk (too close).
        if float(np.linalg.norm(c - desk_c)) <= 0.4 * float(desk_short) + 0.15:
            continue
        if chair is None or min(dims) < min([float(v) for v in chair.get("dimensions", [9.9, 9.9])]):
            chair = item

    if chair is None:
        return furniture

    chair_dims = [float(v) for v in chair.get("dimensions", [0.0, 0.0])]
    dock_dist = 0.5 * float(desk_short) + max(0.12, 0.5 * min(chair_dims) if chair_dims else 0.12)
    dock = desk_c + desk_short_dir * dock_dist
    old_c = np.asarray(chair.get("centroid_xy", dock), dtype=float)
    shift = dock - old_c

    for key in ("rect_xy", "polygon_xy", "overlay_xy"):
        arr = np.asarray(chair.get(key, []), dtype=float)
        if len(arr) >= 3:
            chair[key] = arr + shift
    chair["centroid_xy"] = np.asarray(dock, dtype=float)
    chair["overlay_anchor_centroid_xy"] = np.asarray(dock, dtype=float)
    if len(np.asarray(chair.get("overlay_xy", []), dtype=float)) >= 3:
        chair["overlay_centroid_xy"] = np.asarray(chair["overlay_xy"], dtype=float).mean(axis=0)
    else:
        chair["overlay_centroid_xy"] = np.asarray(dock, dtype=float)
    chair["wall_attached"] = False
    chair["force_wall_attach"] = False
    chair["wall_edge_index"] = int(desk.get("wall_edge_index", -1))
    chair["overlay_alignment"] = "room_axis_overlay"

    # Orient the chair so its long axis is parallel to the desk wall.
    # The chair back runs alongside the desk edge rather than perpendicular to it.
    desk_edge_idx = int(desk.get("wall_edge_index", -1))
    if desk_edge_idx >= 0 and len(np.asarray(poly_xy, dtype=float)) >= 3:
        ea, ed, en, sl = _edge_interior_basis(poly_xy, desk_edge_idx)
        if sl > 1e-6 and len(chair_dims) >= 2:
            chair_long = max(chair_dims)
            chair_short = min(chair_dims)
            chair["overlay_xy"] = _build_oriented_rect(
                np.asarray(dock, dtype=float), ed, en, chair_long, chair_short
            )
            chair["overlay_centroid_xy"] = np.asarray(chair["overlay_xy"], dtype=float).mean(axis=0)
            chair["overlay_alignment"] = "wall_parallel"

    return furniture

def _merge_ldesk_adjacent_overlaps(furniture, poly_xy, pts_xy=None):
    """Merge shelf/desk overlays that overlap significantly with an L-shaped desk overlay
    and rebuild the combined result as a clean geometric L-shape.

    When the L-desk cluster spans the E2/E3 corner, part of it may be detected
    as a separate shelf on an adjacent wall.  This function:
      1. Detects the overlapping neighbour (≥ 20 % area overlap).
      2. Builds a clean wall-aligned rectangle for the L-desk main arm using the
         stored centroid/dimensions rather than the raw cluster polygon.
      3. Unions that clean arm with the neighbour's existing wall_overlay rectangle.
      4. Removes the now-redundant neighbour item.

    pts_xy : Nx2 float array of scene-level XY point positions (pts_clean[:, :2]).
        When provided, raw cluster points are recovered via each item's
        ``point_indices`` field whenever ``polygon_xy`` is too sparse.  This
        allows wall-specific arm-depth computation and the analytical 6-vertex
        L-polygon path to work generically without relying on polygon_xy being
        populated after the regularisation stage.
    """
    try:
        from shapely.geometry import Polygon as _SPoly
        from shapely.ops import unary_union as _sunion
    except ImportError:
        return furniture

    if not furniture or len(np.asarray(poly_xy, dtype=float)) < 3:
        return furniture

    ldsk_indices = [
        i for i, it in enumerate(furniture)
        if it.get("label") == "l_shaped_desk_candidate"
    ]
    if not ldsk_indices:
        return furniture

    to_remove = set()
    for li in ldsk_indices:
        ldesk = furniture[li]
        ledge_idx = int(ldesk.get("wall_edge_index", -1))
        l_dims = [float(v) for v in ldesk.get("dimensions", [0.0, 0.0])]
        l_centroid = np.asarray(ldesk.get("centroid_xy", [0.0, 0.0]), dtype=float)

        # --- Build a clean wall-aligned rectangle for the L-desk main arm ---
        # Use the actual cluster polygon projected onto the wall-edge basis so the
        # arm length and depth match the point cloud rather than the L-bounding-box
        # dimensions (which include the cross-arm and therefore overstate depth).
        l_arm_poly = None
        ea = ed = en = None
        sl = 0.0
        arm_depth_built = 0.75  # sentinel used by merger detection
        if ledge_idx >= 0 and len(l_dims) >= 2 and len(poly_xy) >= 3:
            ea, ed, en, sl = _edge_interior_basis(poly_xy, ledge_idx)
            if sl > 1e-6:
                # Arm depth: wall_distance_p90 is the actual desk surface depth
                # (p90 of all cluster-point distances from the wall).  The full
                # dim_short is the L bounding-box short dimension — it includes
                # the cross-arm and is much larger than the desk surface depth.
                wall_p90 = float(ldesk.get("wall_distance_p90", 0.55))
                # wall_distance_p90 uses min(dist-to-any-wall), which is biased
                # near room corners for L-shaped furniture. Use it only as a
                # fallback; primary sizing comes from wall-specific projections.
                arm_depth = float(np.clip(wall_p90, 0.35, 1.6))
                arm_depth_built = arm_depth

                # Arm length & position: project polygon_xy onto the E3 basis
                # and keep only points within the arm's depth zone (i.e. on the
                # main E3 wall, not the perpendicular cross-arm).
                raw_poly_cluster = np.asarray(ldesk.get("polygon_xy", []), dtype=float)
                # Recover raw cluster XY points from point_indices when polygon_xy
                # is too sparse (common after the regularisation stage).  This gives
                # accurate, data-driven depth statistics for any room geometry.
                if len(raw_poly_cluster) < 8 and pts_xy is not None:
                    _pidx = ldesk.get("point_indices")
                    if _pidx is not None:
                        _pidx = np.asarray(_pidx, dtype=int)
                        _valid_mask = (_pidx >= 0) & (_pidx < len(pts_xy))
                        if int(_valid_mask.sum()) >= 8:
                            raw_poly_cluster = pts_xy[_pidx[_valid_mask]]
                has_raw_cluster = len(raw_poly_cluster) >= 4
                poly_cluster = raw_poly_cluster
                if len(poly_cluster) < 4:
                    poly_cluster = np.asarray(ldesk.get("overlay_xy", []), dtype=float)
                arm_length = max(l_dims)
                center_t_raw = float(np.dot(l_centroid - ea, ed))

                if len(poly_cluster) >= 4:
                    rel = poly_cluster - ea
                    along_proj = rel @ ed
                    depth_proj = rel @ en

                    # Wall-specific assignment: keep vertices that are closer to
                    # this wall than to either adjacent room wall. This removes
                    # corner bias without imposing a fixed depth cap.
                    n_edges_room = len(poly_xy)
                    prev_ei = (ledge_idx - 1) % n_edges_room
                    next_ei = (ledge_idx + 1) % n_edges_room
                    ea_prev, _, en_prev, sl_prev = _edge_interior_basis(poly_xy, prev_ei)
                    ea_next, _, en_next, sl_next = _edge_interior_basis(poly_xy, next_ei)

                    d_main = depth_proj
                    d_prev = (poly_cluster - ea_prev) @ en_prev if sl_prev > 1e-6 else np.full(len(poly_cluster), np.inf)
                    d_next = (poly_cluster - ea_next) @ en_next if sl_next > 1e-6 else np.full(len(poly_cluster), np.inf)
                    d_adj = np.minimum(d_prev, d_next)

                    main_assigned = (
                        (along_proj >= 0) & (along_proj <= sl)
                        & (d_main > 0.02)
                        & (d_main <= d_adj + 0.06)
                    )
                    if int(main_assigned.sum()) >= 4:
                        arm_depth = float(np.percentile(d_main[main_assigned], 90))
                        arm_depth_built = arm_depth

                    in_arm = (
                        (along_proj >= 0) & (along_proj <= sl)
                        & (depth_proj >= 0) & (depth_proj <= arm_depth * 1.05)
                    )
                    if int(in_arm.sum()) >= 6:
                        arm_lo = float(np.percentile(along_proj[in_arm], 2))
                        arm_hi = float(np.percentile(along_proj[in_arm], 98))
                        arm_length = arm_hi - arm_lo
                        center_t_raw = 0.5 * (arm_lo + arm_hi)

                half_long = 0.5 * arm_length
                center_t = float(np.clip(
                    center_t_raw,
                    min(half_long, sl * 0.5),
                    max(sl - half_long, min(half_long, sl * 0.5)),
                ))
                center_along = ea + ed * center_t
                raw_wall_offset = float(ldesk.get("wall_distance_p50", 0.5 * arm_depth))
                target_offset = max(
                    0.5 * arm_depth,
                    0.5 * (raw_wall_offset + 0.5 * arm_depth),
                )
                center_arm = center_along + en * target_offset
                arm_rect = _build_oriented_rect(center_arm, ed, en, arm_length, arm_depth)
                try:
                    l_arm_poly = _SPoly(arm_rect)
                    if not l_arm_poly.is_valid:
                        l_arm_poly = l_arm_poly.buffer(0)
                except Exception:
                    l_arm_poly = None

        # Fallback: use existing overlay if we couldn't build a clean arm.
        if l_arm_poly is None:
            loverlay = ldesk.get("overlay_xy", [])
            if len(loverlay) >= 3:
                try:
                    l_arm_poly = _SPoly(loverlay)
                    if not l_arm_poly.is_valid:
                        l_arm_poly = l_arm_poly.buffer(0)
                except Exception:
                    l_arm_poly = None

        if l_arm_poly is None:
            continue


        # --- Find adjacent/overlapping shelves or desks to absorb ---
        n_edges = len(poly_xy)
        adjacent_edges = {(ledge_idx - 1) % n_edges, (ledge_idx + 1) % n_edges}

        for j, other in enumerate(furniture):
            if j == li or j in to_remove:
                continue
            if other.get("label") not in {"shelf_candidate", "desk_or_table_candidate"}:
                continue
            ooverlay = other.get("overlay_xy", [])
            if len(ooverlay) < 3:
                continue
            try:
                opoly = _SPoly(ooverlay)
                if not opoly.is_valid:
                    opoly = opoly.buffer(0)
            except Exception:
                continue

            try:
                inter_area = float(l_arm_poly.intersection(opoly).area)
                other_area = float(opoly.area)
                ldesk_area = float(l_arm_poly.area)
                overlap_ratio_other = inter_area / other_area if other_area > 1e-6 else 0.0
                overlap_ratio_ldesk = inter_area / ldesk_area if ldesk_area > 1e-6 else 0.0
            except Exception:
                continue

            n_edges_room = len(poly_xy)
            other_edge_idx = int(other.get("wall_edge_index", -1))
            is_adjacent_edge = other_edge_idx in {
                (ledge_idx - 1) % n_edges_room,
                (ledge_idx + 1) % n_edges_room,
            }

            should_merge = overlap_ratio_other >= 0.20 or overlap_ratio_ldesk >= 0.20

            # For shelves on an adjacent wall edge, use a 15 cm proximity buffer
            # instead of pure area overlap: the cross-arm of an L-desk may only
            # barely touch the main arm rectangle after the depth correction.
            if not should_merge and is_adjacent_edge and ea is not None:
                try:
                    should_merge = l_arm_poly.buffer(0.15).intersects(opoly)
                except Exception:
                    pass
                if should_merge:
                    # Extra sanity: cross-arm must start near the corner
                    # (along_lo <= 0.8 m from E3 corner A) and must start no
                    # deeper than the main arm depth + 30 cm (rules out far
                    # shelves on the same adjacent edge that project far from E3).
                    oa_arr = np.asarray(ooverlay, dtype=float)
                    oa_along = float(((oa_arr - ea) @ ed).min())
                    oa_depth = float(((oa_arr - ea) @ en).min())
                    if oa_along > 0.8 or oa_depth > arm_depth_built + 0.30:
                        should_merge = False

            if should_merge and is_adjacent_edge and ea is not None:
                # The raw shelf/cluster overlay does not accurately represent the
                # cross-arm depth from the adjacent wall (sensor sees wall-facing
                # surfaces, so wall_distance_p90 is small even though the desk
                # extends arm_depth into the room from that wall).
                # Build the L-polygon directly as a 6-vertex shape using the
                # known arm rect corners and the cross-arm extent, avoiding
                # Shapely union artifacts at the intersection edge.
                try:
                    other_ei = int(other.get("wall_edge_index", -1))
                    ea_x, ed_x, en_x, sl_x = _edge_interior_basis(poly_xy, other_ei)
                    if sl_x > 1e-6:
                        # Source points for cross-arm sizing: prefer the L-desk
                        # raw cluster (covers both arms) recovered via point_indices,
                        # or the adjacent item's raw cluster, or fall back to the
                        # adjacent overlay.  Any source with ≥ 4 pts works because
                        # the wall-specific filter selects only the cross-arm side.
                        source_pts = poly_cluster if len(poly_cluster) >= 4 else np.asarray(ooverlay, dtype=float)
                        # Also try to recover raw adjacent-item cluster points for
                        # better cross-arm depth statistics.
                        if pts_xy is not None and len(poly_cluster) < 20:
                            _oth_pidx = other.get("point_indices")
                            if _oth_pidx is not None:
                                _oth_pidx = np.asarray(_oth_pidx, dtype=int)
                                _valid_oth = (_oth_pidx >= 0) & (_oth_pidx < len(pts_xy))
                                if int(_valid_oth.sum()) >= 8:
                                    source_pts = pts_xy[_oth_pidx[_valid_oth]]
                        if len(source_pts) < 4:
                            raise RuntimeError("insufficient cross-arm source points")

                        rel_x = source_pts - ea_x
                        along_x = rel_x @ ed_x
                        depth_x = rel_x @ en_x
                        # Estimate cross-arm depth from vertices assigned to the
                        # adjacent wall (closer to that wall than to the L-desk
                        # main wall). This is cloud-driven and generic.
                        d_main_for_cross = (source_pts - ea) @ en
                        valid_cross = (
                            (along_x >= 0) & (along_x <= sl_x)
                            & (depth_x > 0.02)
                        )
                        assigned_cross = (
                            valid_cross
                            & (depth_x <= d_main_for_cross + 0.06)
                        )
                        cross_sel = assigned_cross if int(assigned_cross.sum()) >= 4 else valid_cross
                        if int(cross_sel.sum()) >= 4:
                            cross_depth = float(np.percentile(depth_x[cross_sel], 88))
                        else:
                            cross_depth = arm_depth_built
                        in_cross = cross_sel & (depth_x <= cross_depth * 1.05)
                        if int(in_cross.sum()) >= 4:
                            c_lo = float(np.percentile(along_x[in_cross], 2))
                            c_hi = float(np.percentile(along_x[in_cross], 98))
                            # Extend to the E2/E3 corner if the cluster reaches close.
                            if c_hi > sl_x - cross_depth * 1.5:
                                c_hi = sl_x
                            c_len = c_hi - c_lo

                            # Get E3 arm corners in E3 basis.
                            arm_verts = np.array(l_arm_poly.exterior.coords[:-1])
                            rel_arm = arm_verts - ea
                            arm_along_v = rel_arm @ ed
                            arm_depth_v = rel_arm @ en
                            e3_lo = float(arm_along_v.min())
                            e3_hi = float(arm_along_v.max())
                            e3_dep = float(arm_depth_v.max())
                            # Build a clean 6-vertex L-polygon when the E3 arm
                            # starts near the corner (within 0.25 m of along=0).
                            # The inner corner is the intersection of the two
                            # depth-offset lines (E3 depth=e3_dep and E2 depth=c_dep),
                            # which is valid for any room corner angle.
                            if e3_lo <= 0.25:
                                c_dep = cross_depth
                                # Solve ea3+en3*e3_dep+ed3*t = ea2+en2*c_dep+ed2*s
                                rhs_ic = (ea_x - ea) + (en_x * c_dep - en * e3_dep)
                                A_ic = np.array([[ed[0], -ed_x[0]], [ed[1], -ed_x[1]]])
                                try:
                                    ts_ic = np.linalg.solve(A_ic, rhs_ic)
                                    P3 = ea + en * e3_dep + ed * float(ts_ic[0])
                                except np.linalg.LinAlgError:
                                    P3 = ea + en * e3_dep  # fallback
                                P0 = ea
                                P1 = ea  + ed   * e3_hi
                                P2 = ea  + ed   * e3_hi + en * e3_dep
                                P4 = ea_x + ed_x * c_lo  + en_x * c_dep
                                P5 = ea_x + ed_x * c_lo
                                try:
                                    l_shape = _SPoly([P0, P1, P2, P3, P4, P5])
                                    if not l_shape.is_valid:
                                        l_shape = l_shape.buffer(0)
                                    if l_shape.is_valid and l_shape.geom_type == "Polygon" and l_shape.area > 0.20:
                                        l_arm_poly = l_shape
                                        to_remove.add(j)
                                        continue
                                except Exception:
                                    pass
                            # Fallback: use clean cross-arm rect and union.
                            c_t = float(np.clip(
                                0.5 * (c_lo + c_hi),
                                min(0.5 * c_len, sl_x * 0.5),
                                max(sl_x - 0.5 * c_len, min(0.5 * c_len, sl_x * 0.5)),
                            ))
                            c_center = ea_x + ed_x * c_t + en_x * (0.5 * cross_depth)
                            cross_rect = _build_oriented_rect(c_center, ed_x, en_x, c_len, cross_depth)
                            clean_cross = _SPoly(cross_rect)
                            if not clean_cross.is_valid:
                                clean_cross = clean_cross.buffer(0)
                            if clean_cross.is_valid and clean_cross.area > 0.01:
                                opoly = clean_cross
                except Exception:
                    pass

            if should_merge:
                try:
                    merged = _sunion([l_arm_poly, opoly])
                    if merged.geom_type == "Polygon":
                        l_arm_poly = merged
                    to_remove.add(j)
                except Exception:
                    continue

        # Write the clean (possibly merged) polygon back to the L-desk item.
        try:
            final_coords = list(l_arm_poly.exterior.coords[:-1])
            ldesk["overlay_xy"] = [[float(x), float(y)] for x, y in final_coords]
            ldesk["overlay_centroid_xy"] = [
                float(l_arm_poly.centroid.x), float(l_arm_poly.centroid.y)
            ]
            ldesk["overlay_alignment"] = "polygon_overlay"
        except Exception:
            pass

    return [it for i, it in enumerate(furniture) if i not in to_remove]


def _scene_furniture_candidate_has_stable_support(item):
    label = str(item.get("label") or "")
    if label != "large_furniture_candidate":
        return True

    wall_contact_type = str(item.get("wall_contact_type") or "")
    if wall_contact_type != "freestanding":
        return True

    dim_long, dim_short = [float(v) for v in item.get("dimensions", [0.0, 0.0])]
    area = float(item.get("area", 0.0))
    wall_dist = float(item.get("wall_distance_p50", item.get("wall_distance_min", 0.0)))
    observation_count = int(item.get("observation_count", 1))
    frame_support_points = int(item.get("frame_support_points", 0))

    # In walking captures, large interior blobs that appear only after the
    # global semantic merge are usually floor/occlusion artifacts rather than
    # stable furniture. Keep real repeated objects, but drop merged-only blobs.
    if (
        dim_long >= 1.6
        and dim_short >= 1.0
        and area >= 1.3
        and wall_dist >= 0.9
        and observation_count < 2
        and frame_support_points < 180
    ):
        return False

    return True


def _scene_furniture_overlap_ratio(item_a, item_b):
    try:
        from shapely.geometry import Polygon as ShapelyPolygon
    except Exception:
        return 0.0

    def _item_polygon(item):
        for key in ("overlay_xy", "rect_xy", "polygon_xy"):
            coords = np.asarray(item.get(key, []), dtype=float)
            if len(coords) < 3:
                continue
            poly = ShapelyPolygon(coords)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty or poly.area <= 1e-6:
                continue
            return poly
        return None

    poly_a = _item_polygon(item_a)
    poly_b = _item_polygon(item_b)
    if poly_a is None or poly_b is None:
        return 0.0

    inter_area = float(poly_a.intersection(poly_b).area)
    if inter_area <= 1e-6:
        return 0.0

    return inter_area / max(min(float(poly_a.area), float(poly_b.area)), 1e-6)


def _suppress_redundant_scene_furniture_items(furniture):
    if len(furniture) < 2:
        return list(furniture), 0

    keep_mask = [True] * len(furniture)
    for idx, item in enumerate(furniture):
        if not bool(item.get("frame_backed")):
            item_obs = int(item.get("observation_count", 1))
            if item_obs <= 1:
                item_label = str(item.get("label") or "")
                for other_idx, other in enumerate(furniture):
                    if idx == other_idx or not bool(other.get("frame_backed")):
                        continue

                    other_obs = int(other.get("observation_count", 1))
                    if other_obs < 4:
                        continue

                    other_label = str(other.get("label") or "")
                    labels_compatible = (
                        _semantic_label_family(item_label) == _semantic_label_family(other_label)
                        or item_label == "large_furniture_candidate"
                        or other_label == "large_furniture_candidate"
                    )
                    if not labels_compatible:
                        continue

                    item_edge = int(item.get("wall_edge_index", -1))
                    other_edge = int(other.get("wall_edge_index", -1))
                    if item_edge >= 0 and other_edge >= 0 and item_edge != other_edge:
                        continue

                    if _scene_furniture_overlap_ratio(item, other) < 0.80:
                        continue

                    item_centroid = np.asarray(item.get("centroid_xy", []), dtype=float)
                    other_centroid = np.asarray(other.get("centroid_xy", []), dtype=float)
                    if len(item_centroid) != 2 or len(other_centroid) != 2:
                        continue
                    if float(np.linalg.norm(item_centroid - other_centroid)) > 0.42:
                        continue

                    keep_mask[idx] = False
                    break

        if not keep_mask[idx]:
            continue

        item_label = str(item.get("label") or "")
        item_dims = [float(v) for v in item.get("dimensions", [0.0, 0.0])]
        item_short = min(item_dims) if item_dims else 0.0
        item_height = float(item.get("height", 0.0))
        item_area = float(item.get("area", 0.0))
        item_wall_dist = float(item.get("wall_distance_p50", 0.0))
        item_edge_idx = int(item.get("wall_edge_index", -1))
        if (
            item_label != "desk_or_table_candidate"
            or item_short > 0.60
            or item_height > 0.85
            or item_wall_dist > 0.32
            or item_edge_idx < 0
        ):
            continue

        for other_idx, other in enumerate(furniture):
            if idx == other_idx:
                continue

            other_label = str(other.get("label") or "")
            if other_label not in {"desk_or_table_candidate", "large_furniture_candidate"}:
                continue

            other_area = float(other.get("area", 0.0))
            other_dims = [float(v) for v in other.get("dimensions", [0.0, 0.0])]
            other_short = min(other_dims) if other_dims else 0.0
            if int(other.get("wall_edge_index", -1)) != item_edge_idx:
                continue
            if other_area < max(0.9, item_area * 1.8) and other_short < 0.80:
                continue

            if _scene_furniture_overlap_ratio(item, other) < 0.30:
                other_centroid = np.asarray(other.get("centroid_xy", []), dtype=float)
                item_centroid = np.asarray(item.get("centroid_xy", []), dtype=float)
                if len(other_centroid) != 2 or len(item_centroid) != 2:
                    continue
                if float(np.linalg.norm(other_centroid - item_centroid)) > 0.55:
                    continue

            keep_mask[idx] = False
            break

    filtered = [item for item, keep in zip(furniture, keep_mask) if keep]
    rejected = int(sum(1 for keep in keep_mask if not keep))
    return filtered, rejected


def _detect_opening_observations(points_xyz, poly_xy, floor_z, room_height=None, blocked_mask=None,
                                 dist_threshold=0.16, min_near_points=40, min_left_right_support=3):
    if points_xyz is None or len(points_xyz) < max(40, min_near_points):
        return []

    points_xy = points_xyz[:, :2]
    edge_dists, edge_proj, _, _ = _nearest_edge_geometry(points_xy, poly_xy)
    blocked_mask = np.zeros(len(points_xyz), dtype=bool) if blocked_mask is None else np.asarray(blocked_mask, dtype=bool)

    opening_candidates = []
    low_z_lo = floor_z + 0.05
    low_z_hi = floor_z + 0.45
    mid_z_lo = floor_z + 0.65
    mid_z_hi = floor_z + (min(room_height - 0.25, 1.95) if room_height is not None else 1.95)
    for edge_idx in range(len(poly_xy)):
        t, seg_len = edge_proj[edge_idx]
        if seg_len < 0.8:
            continue
        dist = edge_dists[:, edge_idx]
        near = (dist <= dist_threshold) & (~blocked_mask)
        if int(near.sum()) < min_near_points:
            continue
        bins = max(8, int(np.ceil(seg_len / 0.12)))
        edges_1d = np.linspace(0.0, seg_len, bins + 1)
        low_mask = near & (points_xyz[:, 2] >= low_z_lo) & (points_xyz[:, 2] <= low_z_hi)
        mid_mask = near & (points_xyz[:, 2] >= mid_z_lo) & (points_xyz[:, 2] <= mid_z_hi)
        low_counts, _ = np.histogram(t[low_mask], bins=edges_1d)
        mid_counts, _ = np.histogram(t[mid_mask], bins=edges_1d)
        gap = mid_counts <= 1
        start = None
        for idx, is_gap in enumerate(gap.tolist() + [False]):
            if is_gap and start is None:
                start = idx
            elif not is_gap and start is not None:
                end = idx
                gap_len = edges_1d[end] - edges_1d[start]
                left_support = int(mid_counts[max(0, start - 2):start].sum())
                right_support = int(mid_counts[end:min(len(mid_counts), end + 2)].sum())
                if (
                    0.50 <= gap_len <= 1.60
                    and left_support >= min_left_right_support
                    and right_support >= min_left_right_support
                ):
                    low_gap = float(low_counts[start:end].mean()) if end > start else 0.0
                    kind = "door_candidate" if low_gap <= 1.0 else "window_candidate"
                    t0 = float(edges_1d[start])
                    t1 = float(edges_1d[end])
                    a = poly_xy[edge_idx]
                    b = poly_xy[(edge_idx + 1) % len(poly_xy)]
                    seg = b - a
                    seg_dir = seg / (np.linalg.norm(seg) + 1e-9)
                    p0 = a + seg_dir * t0
                    p1 = a + seg_dir * t1
                    opening_candidates.append({
                        "kind": kind,
                        "edge_index": int(edge_idx),
                        "line_xy": np.array([p0, p1], dtype=float),
                        "width": float(np.linalg.norm(p1 - p0)),
                        "centroid_xy": 0.5 * (p0 + p1),
                        "observation_count": 1,
                        "support_score": int(left_support + right_support),
                    })
                start = None
    return opening_candidates


def _wall_edge_band_profiles(points_xyz, poly_xy, floor_z, room_height=None, blocked_mask=None,
                             dist_threshold=0.16, bin_width=0.12):
    if points_xyz is None or len(points_xyz) == 0:
        return []

    points_xy = points_xyz[:, :2]
    edge_dists, edge_proj, _, _ = _nearest_edge_geometry(points_xy, poly_xy)
    blocked_mask = np.zeros(len(points_xyz), dtype=bool) if blocked_mask is None else np.asarray(blocked_mask, dtype=bool)

    low_z_lo = floor_z + 0.05
    low_z_hi = floor_z + 0.45
    mid_z_lo = floor_z + 0.65
    mid_z_hi = floor_z + (min(room_height - 0.25, 1.95) if room_height is not None else 1.95)
    top_z_lo = floor_z + (min(room_height - 0.30, 2.0) if room_height is not None else 1.75)
    top_z_hi = floor_z + (room_height + 0.08 if room_height is not None else 2.6)

    profiles = []
    for edge_idx in range(len(poly_xy)):
        t, seg_len = edge_proj[edge_idx]
        if seg_len < 0.8:
            profiles.append(None)
            continue
        bins = max(8, int(np.ceil(seg_len / bin_width)))
        edges_1d = np.linspace(0.0, seg_len, bins + 1)
        dist = edge_dists[:, edge_idx]
        near = (dist <= dist_threshold) & (~blocked_mask)
        if int(near.sum()) < 12:
            profiles.append({
                "edge_index": int(edge_idx),
                "seg_len": float(seg_len),
                "edges_1d": edges_1d,
                "near_count": int(near.sum()),
                "low_counts": np.zeros(bins, dtype=int),
                "mid_counts": np.zeros(bins, dtype=int),
                "top_counts": np.zeros(bins, dtype=int),
                "all_counts": np.zeros(bins, dtype=int),
            })
            continue
        low_mask = near & (points_xyz[:, 2] >= low_z_lo) & (points_xyz[:, 2] <= low_z_hi)
        mid_mask = near & (points_xyz[:, 2] >= mid_z_lo) & (points_xyz[:, 2] <= mid_z_hi)
        top_mask = near & (points_xyz[:, 2] >= top_z_lo) & (points_xyz[:, 2] <= top_z_hi)
        profiles.append({
            "edge_index": int(edge_idx),
            "seg_len": float(seg_len),
            "edges_1d": edges_1d,
            "near_count": int(near.sum()),
            "low_counts": np.histogram(t[low_mask], bins=edges_1d)[0],
            "mid_counts": np.histogram(t[mid_mask], bins=edges_1d)[0],
            "top_counts": np.histogram(t[top_mask], bins=edges_1d)[0],
            "all_counts": np.histogram(t[near], bins=edges_1d)[0],
        })
    return profiles


def _aggregate_frame_edge_profiles(semantic_frames, poly_xy, floor_z, room_height=None, dist_threshold=0.16):
    if not semantic_frames:
        return []

    aggregated = None
    observations = 0
    for pts in semantic_frames:
        profiles = _wall_edge_band_profiles(
            pts,
            poly_xy,
            floor_z,
            room_height=room_height,
            blocked_mask=None,
            dist_threshold=dist_threshold,
        )
        if aggregated is None:
            aggregated = []
            for profile in profiles:
                if profile is None:
                    aggregated.append(None)
                    continue
                aggregated.append({
                    "edge_index": int(profile["edge_index"]),
                    "seg_len": float(profile["seg_len"]),
                    "edges_1d": profile["edges_1d"],
                    "seen_frames": 0,
                    "low_counts": np.zeros_like(profile["low_counts"], dtype=float),
                    "mid_counts": np.zeros_like(profile["mid_counts"], dtype=float),
                    "top_counts": np.zeros_like(profile["top_counts"], dtype=float),
                    "all_counts": np.zeros_like(profile["all_counts"], dtype=float),
                })
        for edge_idx, profile in enumerate(profiles):
            if profile is None or aggregated[edge_idx] is None:
                continue
            if int(profile.get("near_count", 0)) < 12:
                continue
            aggregated[edge_idx]["seen_frames"] += 1
            aggregated[edge_idx]["low_counts"] += profile["low_counts"]
            aggregated[edge_idx]["mid_counts"] += profile["mid_counts"]
            aggregated[edge_idx]["top_counts"] += profile["top_counts"]
            aggregated[edge_idx]["all_counts"] += profile["all_counts"]
            observations += 1

    return aggregated or [], int(observations)


def _detect_structural_openings(points_xyz, poly_xy, floor_z, room_height=None, blocked_mask=None, semantic_frames=None):
    merged_profiles = _wall_edge_band_profiles(
        points_xyz,
        poly_xy,
        floor_z,
        room_height=room_height,
        blocked_mask=blocked_mask,
        dist_threshold=0.16,
    )
    frame_profiles, frame_profile_observations = _aggregate_frame_edge_profiles(
        semantic_frames,
        poly_xy,
        floor_z,
        room_height=room_height,
        dist_threshold=0.16,
    ) if semantic_frames else ([], 0)

    opening_candidates = []
    edge_debug = []

    def _clamp01(value):
        return float(max(0.0, min(1.0, value)))

    def _opening_evidence_score(gap_len, low_fill, top_fill, left_support, right_support, support_score):
        width_center = 0.95
        width_half_span = 0.45
        width_score = _clamp01(1.0 - (abs(float(gap_len) - width_center) / width_half_span))
        door_profile = _clamp01(1.0 - max(float(low_fill), float(top_fill)) / 0.75)
        window_profile = _clamp01((float(low_fill) - 1.0) / 1.5) * _clamp01(1.0 - float(top_fill) / 0.75)
        profile_score = max(door_profile, window_profile)
        jamb_score = min(_clamp01(float(left_support) / 8.0), _clamp01(float(right_support) / 8.0))
        context_score = _clamp01(float(support_score) / 28.0)
        return round(0.35 * width_score + 0.25 * profile_score + 0.20 * jamb_score + 0.20 * context_score, 3)

    for edge_idx, merged in enumerate(merged_profiles):
        if not merged or int(merged.get("near_count", 0)) < 30:
            continue
        frame = frame_profiles[edge_idx] if edge_idx < len(frame_profiles) else None
        low_counts = merged["low_counts"].astype(float)
        mid_counts = merged["mid_counts"].astype(float)
        top_counts = merged["top_counts"].astype(float)
        all_counts = merged["all_counts"].astype(float)
        if frame is not None and int(frame.get("seen_frames", 0)) > 0:
            seen = max(int(frame["seen_frames"]), 1)
            low_counts = np.maximum(low_counts, frame["low_counts"] / seen)
            mid_counts = np.maximum(mid_counts, frame["mid_counts"] / seen)
            top_counts = np.maximum(top_counts, frame["top_counts"] / seen)
            all_counts = np.maximum(all_counts, frame["all_counts"] / seen)

        support = np.maximum(mid_counts, top_counts)
        gap = (support <= 1.0) & (all_counts <= 2.0)
        continuity = float(np.count_nonzero(support > 1.0) / max(len(support), 1))
        low_fill_ratio = float(np.count_nonzero(low_counts > 0.5) / max(len(low_counts), 1))
        mid_fill_ratio = float(np.count_nonzero(mid_counts > 0.5) / max(len(mid_counts), 1))
        top_fill_ratio = float(np.count_nonzero(top_counts > 0.5) / max(len(top_counts), 1))
        best_gap_debug = None
        start = None
        for idx, is_gap in enumerate(gap.tolist() + [False]):
            if is_gap and start is None:
                start = idx
            elif not is_gap and start is not None:
                end = idx
                gap_len = merged["edges_1d"][end] - merged["edges_1d"][start]
                left_support = float(support[max(0, start - 2):start].sum())
                right_support = float(support[end:min(len(support), end + 2)].sum())
                left_all = float(all_counts[max(0, start - 2):start].sum())
                right_all = float(all_counts[end:min(len(all_counts), end + 2)].sum())
                low_fill = float(low_counts[start:end].mean()) if end > start else 0.0
                top_fill = float(top_counts[start:end].mean()) if end > start else 0.0
                support_score = left_support + right_support + 0.5 * (left_all + right_all)
                gap_debug = {
                    "gap_len": float(gap_len),
                    "low_fill": float(low_fill),
                    "top_fill": float(top_fill),
                    "left_support": float(left_support),
                    "right_support": float(right_support),
                    "support_score": float(support_score),
                    "evidence_score": _opening_evidence_score(
                        gap_len,
                        low_fill,
                        top_fill,
                        left_support,
                        right_support,
                        support_score,
                    ),
                }
                rejects = []
                if gap_len < 0.60:
                    rejects.append("too_narrow")
                if gap_len > 1.40:
                    rejects.append("too_wide")
                if left_support < 3.0:
                    rejects.append("weak_left_jamb")
                if right_support < 3.0:
                    rejects.append("weak_right_jamb")
                if support_score < 10.0:
                    rejects.append("weak_context")

                kind = None
                if not rejects:
                    if low_fill <= 0.75 and top_fill <= 0.75:
                        kind = "door_candidate"
                    elif low_fill >= 1.0 and top_fill <= 0.75:
                        kind = "window_candidate"
                    else:
                        rejects.append("height_profile_not_opening")

                gap_debug["kind"] = kind
                gap_debug["rejections"] = rejects
                gap_debug["accepted"] = kind is not None
                if best_gap_debug is None or (
                    gap_debug["evidence_score"], gap_debug["support_score"], gap_debug["gap_len"]
                ) > (
                    best_gap_debug["evidence_score"], best_gap_debug["support_score"], best_gap_debug["gap_len"]
                ):
                    best_gap_debug = gap_debug

                if kind is not None:
                    t0 = float(merged["edges_1d"][start])
                    t1 = float(merged["edges_1d"][end])
                    a = poly_xy[edge_idx]
                    b = poly_xy[(edge_idx + 1) % len(poly_xy)]
                    seg = b - a
                    seg_dir = seg / (np.linalg.norm(seg) + 1e-9)
                    p0 = a + seg_dir * t0
                    p1 = a + seg_dir * t1
                    opening_candidates.append({
                        "kind": kind,
                        "edge_index": int(edge_idx),
                        "line_xy": np.array([p0, p1], dtype=float),
                        "width": float(np.linalg.norm(p1 - p0)),
                        "centroid_xy": 0.5 * (p0 + p1),
                        "observation_count": 1,
                        "support_score": int(round(support_score)),
                    })
                start = None

        edge_item = {
            "edge_index": int(edge_idx),
            "continuity": round(continuity, 3),
            "low_fill_ratio": round(low_fill_ratio, 3),
            "mid_fill_ratio": round(mid_fill_ratio, 3),
            "top_fill_ratio": round(top_fill_ratio, 3),
            "near_count": int(merged.get("near_count", 0)),
            "candidate_gaps": int(np.count_nonzero(gap)),
        }
        if best_gap_debug is not None:
            edge_item.update({
                "opening_evidence_score": round(float(best_gap_debug["evidence_score"]), 3),
                "best_gap_width": round(float(best_gap_debug["gap_len"]), 3),
                "best_gap_support_score": round(float(best_gap_debug["support_score"]), 3),
                "best_gap_low_fill": round(float(best_gap_debug["low_fill"]), 3),
                "best_gap_top_fill": round(float(best_gap_debug["top_fill"]), 3),
                "best_gap_kind": best_gap_debug.get("kind"),
                "best_gap_status": "accepted" if best_gap_debug.get("accepted") else "rejected",
                "best_gap_rejections": list(best_gap_debug.get("rejections", [])),
            })
        edge_debug.append(edge_item)

    opening_candidates = _fuse_opening_observations(opening_candidates)
    diagnostics = {
        "frame_opening_profile_observations": int(frame_profile_observations),
        "structural_opening_candidates": int(len(opening_candidates)),
    }
    return opening_candidates, diagnostics, edge_debug


def _fuse_opening_observations(openings):
    if not openings:
        return []

    groups = []
    ranked = sorted(openings, key=lambda item: (item.get("support_score", 0), item.get("width", 0.0)), reverse=True)
    for obs in ranked:
        centroid = np.asarray(obs["centroid_xy"], dtype=float)
        best_group = None
        best_dist = float("inf")
        for group in groups:
            ref = np.asarray(group["centroid_xy"], dtype=float)
            if int(group["edge_index"]) != int(obs["edge_index"]):
                continue
            if group["kind"] != obs["kind"]:
                continue
            dist = float(np.linalg.norm(centroid - ref))
            if dist > 0.42:
                continue
            if abs(float(group["width"]) - float(obs["width"])) > 0.45:
                continue
            if dist < best_dist:
                best_group = group
                best_dist = dist
        if best_group is None:
            groups.append({
                "members": [obs],
                "centroid_xy": centroid,
                "edge_index": int(obs["edge_index"]),
                "kind": obs["kind"],
                "width": float(obs["width"]),
            })
        else:
            best_group["members"].append(obs)
            widths = [float(item["width"]) for item in best_group["members"]]
            best_group["width"] = float(np.median(widths))
            centroids = np.array([item["centroid_xy"] for item in best_group["members"]], dtype=float)
            best_group["centroid_xy"] = np.median(centroids, axis=0)

    fused = []
    for group in groups:
        members = group["members"]
        representative = max(members, key=lambda item: (item.get("support_score", 0), item.get("width", 0.0)))
        fused_item = dict(representative)
        fused_item["centroid_xy"] = np.asarray(group["centroid_xy"], dtype=float)
        fused_item["width"] = float(group["width"])
        fused_item["observation_count"] = int(len(members))
        fused_item["support_score"] = int(sum(int(item.get("support_score", 0)) for item in members))
        fused.append(fused_item)

    fused.sort(key=lambda item: (item.get("observation_count", 0), item.get("support_score", 0)), reverse=True)
    return fused


def _refine_window_candidate_with_temporal_profile(candidate, frame_profiles, poly_xy):
    """Refine window placement from per-frame edge profiles (point cloud only)."""
    if str(candidate.get("kind") or "") != "window_candidate":
        return candidate
    if not frame_profiles:
        return candidate

    edge_idx = int(candidate.get("edge_index", -1))
    if edge_idx < 0 or edge_idx >= len(frame_profiles):
        return candidate
    profile = frame_profiles[edge_idx]
    if not profile or int(profile.get("seen_frames", 0)) < 2:
        return candidate

    edges_1d = np.asarray(profile.get("edges_1d", []), dtype=float)
    low_counts = np.asarray(profile.get("low_counts", []), dtype=float)
    mid_counts = np.asarray(profile.get("mid_counts", []), dtype=float)
    top_counts = np.asarray(profile.get("top_counts", []), dtype=float)
    all_counts = np.asarray(profile.get("all_counts", []), dtype=float)
    if len(edges_1d) < 2 or len(low_counts) == 0:
        return candidate
    if not (len(low_counts) == len(mid_counts) == len(top_counts) == len(all_counts) == len(edges_1d) - 1):
        return candidate

    seen = float(max(int(profile.get("seen_frames", 0)), 1))
    low = low_counts / seen
    mid = mid_counts / seen
    top = top_counts / seen

    a = np.asarray(poly_xy[edge_idx], dtype=float)
    b = np.asarray(poly_xy[(edge_idx + 1) % len(poly_xy)], dtype=float)
    seg = b - a
    seg_len = float(np.linalg.norm(seg))
    if seg_len < 0.8:
        return candidate
    seg_dir = seg / (seg_len + 1e-9)

    width = float(candidate.get("width", 0.0) or 0.0)
    if width < 0.45:
        return candidate
    bin_w = float(np.median(np.diff(edges_1d))) if len(edges_1d) > 1 else 0.12
    if bin_w <= 1e-6:
        return candidate

    line_xy = np.asarray(candidate.get("line_xy", []), dtype=float)
    if len(line_xy) == 2:
        center_t_arr, _, _ = _project_points_to_segment(
            np.asarray(candidate.get("centroid_xy", np.zeros(2, dtype=float)), dtype=float).reshape(1, 2),
            a,
            b,
        )
        cur_t = float(center_t_arr[0])
    else:
        cur_t = 0.5 * seg_len

    endpoint_clearance = min(0.25, max(0.10, 0.22 * width))
    nbins = len(low)
    width_bins = int(max(4, round(width / bin_w)))
    width_bins = min(width_bins, nbins)
    if width_bins < 4:
        return candidate

    width_candidates = range(
        max(4, int(round(0.85 * width / bin_w))),
        min(nbins, int(round(1.25 * width / bin_w))) + 1,
    )
    if not width_candidates:
        width_candidates = [width_bins]

    def _window_score(start_bin, wb):
        end_bin = start_bin + wb
        if start_bin < 0 or end_bin > nbins:
            return None
        l = float(low[start_bin:end_bin].mean())
        m = float(mid[start_bin:end_bin].mean())
        t = float(top[start_bin:end_bin].mean())
        center_t = 0.5 * (float(edges_1d[start_bin]) + float(edges_1d[end_bin]))
        proximity_penalty = 0.10 * (abs(center_t - cur_t) / max(width, 0.60))
        # Prefer spans with strong low-band support but reduced mid/top support.
        return (l + 1.0) / (m + 1.0) + 0.35 * ((l + 1.0) / (t + 1.0)) - proximity_penalty

    cur_start = int(np.clip(round((cur_t - 0.5 * width) / bin_w), 0, max(0, nbins - width_bins)))
    cur_score = _window_score(cur_start, width_bins)
    if cur_score is None:
        return candidate

    best = None
    for wb in width_candidates:
        for s in range(0, nbins - wb + 1):
            t0 = float(edges_1d[s])
            t1 = float(edges_1d[s + wb])
            if t0 < endpoint_clearance or (seg_len - t1) < endpoint_clearance:
                continue
            score = _window_score(s, wb)
            if score is None:
                continue
            if best is None or score > best[0]:
                best = (score, s, wb)

    if best is None:
        return candidate

    best_score, best_start, best_wb = best
    best_center_t = 0.5 * (float(edges_1d[best_start]) + float(edges_1d[best_start + best_wb]))
    shift = abs(best_center_t - cur_t)
    improvement = float(best_score - cur_score)

    diagnosed = dict(candidate)
    diagnosed["temporal_profile_seen_frames"] = int(profile.get("seen_frames", 0))
    diagnosed["temporal_profile_cur_t"] = round(float(cur_t), 3)
    diagnosed["temporal_profile_best_t"] = round(float(best_center_t), 3)
    diagnosed["temporal_profile_cur_score"] = round(float(cur_score), 3)
    diagnosed["temporal_profile_best_score"] = round(float(best_score), 3)
    diagnosed["temporal_profile_shift_m"] = round(float(shift), 3)
    diagnosed["temporal_profile_score_gain"] = round(float(improvement), 3)

    if shift < 0.06:
        return diagnosed
    if improvement < 0.05 and best_score < 1.02 * cur_score:
        return diagnosed

    t0 = float(edges_1d[best_start])
    t1 = float(edges_1d[best_start + best_wb])
    p0 = a + seg_dir * t0
    p1 = a + seg_dir * t1

    refined = diagnosed
    refined["line_xy"] = np.array([p0, p1], dtype=float)
    refined["width"] = float(np.linalg.norm(p1 - p0))
    refined["centroid_xy"] = 0.5 * (p0 + p1)
    refined["temporal_profile_refined"] = True
    return refined


def _snap_openings_to_coverage_minimum(opening_candidates, pts_xyz, poly_xy, floor_z, room_height=None):
    """
    Post-process promoted door/window candidates by nudging them toward a local
    minimum-coverage span on the same wall edge without changing their width.

    This runs after promotion, so it should stay conservative: preserve the
    existing edge, kind, and width, and only allow modest local shifts.
    """
    if not opening_candidates or pts_xyz is None or len(pts_xyz) < 50:
        return

    pts_xy = pts_xyz[:, :2]
    z = pts_xyz[:, 2]
    BW = 0.12  # match _wall_edge_band_profiles bin width

    for candidate in opening_candidates:
        candidate_kind = str(candidate.get("kind", ""))
        if candidate_kind not in {"window_candidate", "door_candidate"}:
            continue

        edge_idx = int(candidate.get("edge_index", -1))
        if edge_idx < 0 or edge_idx >= len(poly_xy):
            continue
        width = float(candidate.get("width", 0.0))
        if width < 0.40:
            continue

        a = np.asarray(poly_xy[edge_idx], dtype=float)
        b = np.asarray(poly_xy[(edge_idx + 1) % len(poly_xy)], dtype=float)
        seg = b - a
        seg_len = float(np.linalg.norm(seg))
        if seg_len < 0.8:
            continue
        seg_dir = seg / (seg_len + 1e-9)

        rel = pts_xy - a
        t_vals = rel @ seg_dir
        perp = rel - np.outer(t_vals, seg_dir)
        near = (np.linalg.norm(perp, axis=1) <= 0.20) & (t_vals >= 0.0) & (t_vals <= seg_len)

        if candidate_kind == "window_candidate":
            band_mask = near & (z >= floor_z + 0.80) & (z <= floor_z + 1.80)
            # Windows often need a larger correction than doors because the
            # frame-promoted candidate may drift along a nearly-uniform wall.
            max_shift = min(1.95, max(0.60, 1.60 * width))
        else:
            band_mask = near & (z >= floor_z + 0.05) & (z <= floor_z + 1.95)
            max_shift = min(0.55, max(0.18, 0.45 * width))
        if candidate_kind == "window_candidate":
            band_mask = near & (z >= floor_z + 0.80) & (z <= floor_z + 1.80)
            max_shift = min(0.70, max(0.20, 0.55 * width))
        else:
            band_mask = near & (z >= floor_z + 0.05) & (z <= floor_z + 1.95)
            max_shift = min(0.55, max(0.18, 0.45 * width))

        nbuckets = int(seg_len / BW) + 1
        counts = np.zeros(nbuckets, dtype=float)
        for ti in t_vals[band_mask]:
            bi = int(ti / BW)
            if 0 <= bi < nbuckets:
                counts[bi] += 1.0

        if float(np.sum(counts)) < 20:
            continue

        width_bins = min(max(3, int(round(width / BW))), nbuckets)
        search_bins = width_bins
        endpoint_bins = max(2, int(0.30 / BW + 0.5))
        cur_centroid = np.asarray(candidate.get("centroid_xy", []), dtype=float)
        if len(cur_centroid) != 2:
            continue
        cur_t = float(np.dot(cur_centroid - a, seg_dir))

        def _window_mean(start_bin, width_bin_count):
            if start_bin < 0 or start_bin + width_bin_count > nbuckets:
                return None
            return float(counts[start_bin:start_bin + width_bin_count].mean())

        cur_start = int(np.clip(round((cur_t - 0.5 * width) / BW), 0, max(0, nbuckets - width_bins)))
        cur_score = _window_mean(cur_start, width_bins)
        if cur_score is None:
            continue

        best_start = None
        best_bins = None
        best_penalized = float("inf")
        best_raw = float("inf")

        if candidate_kind == "window_candidate":
            width_candidates = range(
                max(3, int(round(0.80 * width / BW))),
                min(nbuckets - 2 * endpoint_bins, int(round(1.40 * width / BW))) + 1,
            )
            if not width_candidates:
                width_candidates = [width_bins]
        else:
            width_candidates = [width_bins]

        for wb in width_candidates:
            for s in range(endpoint_bins, nbuckets - wb - endpoint_bins + 1):
                center_t = (s + 0.5 * wb) * BW
                if abs(center_t - cur_t) > max_shift:
                    continue
                raw_score = _window_mean(s, wb)
                if raw_score is None:
                    continue
                if candidate_kind == "window_candidate":
                    width_penalty = 0.30 * cur_score * abs(wb - width_bins) / max(float(width_bins), 1.0)
                else:
                    width_penalty = 0.0
                penalized = raw_score + width_penalty
                if penalized < best_penalized:
                    best_penalized = penalized
                    best_raw = raw_score
                    best_start = s
                    best_bins = wb

        if best_start is None or best_bins is None:
            continue

        gap_center_t = (best_start + 0.5 * best_bins) * BW
        if abs(gap_center_t - cur_t) < 0.12:
            continue

        improvement_abs = cur_score - best_raw
        improvement_rel = improvement_abs / max(cur_score, 1e-6)
        if candidate_kind == "window_candidate":
            if improvement_abs < 35.0 or improvement_rel < 0.08:
                continue
        else:
            if improvement_abs < 20.0 or improvement_rel < 0.04:
                continue

        snapped_width = float(best_bins * BW)
        half_w = 0.5 * snapped_width
        t0_max = seg_len - endpoint_bins * BW - snapped_width
        if t0_max <= endpoint_bins * BW:
            continue
        t0_new = float(np.clip(gap_center_t - half_w, endpoint_bins * BW, t0_max))
        t1_new = t0_new + snapped_width

        p0 = a + seg_dir * t0_new
        p1 = a + seg_dir * t1_new
        candidate["centroid_xy"] = 0.5 * (p0 + p1)
        candidate["line_xy"] = np.array([p0, p1], dtype=float)
        candidate["width"] = float(np.linalg.norm(p1 - p0))
        candidate["coverage_snapped"] = True
        candidate["coverage_snap_old_t"] = round(cur_t, 3)
        candidate["coverage_snap_new_t"] = round(gap_center_t, 3)
        candidate["coverage_snap_current_score"] = round(float(cur_score), 1)
        candidate["coverage_snap_best_score"] = round(float(best_raw), 1)
        candidate["coverage_snap_improvement"] = round(float(improvement_abs), 1)


def _apply_low_evidence_window_center_prior(opening_candidates, pts_xyz, poly_xy, floor_z, room_height=None):
    """Bias ambiguous windows toward wall center when density evidence is flat."""
    if not opening_candidates or pts_xyz is None or len(pts_xyz) < 50:
        return

    edge_profiles = _wall_edge_band_profiles(
        pts_xyz,
        poly_xy,
        floor_z,
        room_height=room_height,
        blocked_mask=None,
        dist_threshold=0.16,
        bin_width=0.12,
    )

    for candidate in opening_candidates:
        if str(candidate.get("kind", "")) != "window_candidate":
            continue

        edge_idx = int(candidate.get("edge_index", -1))
        if edge_idx < 0 or edge_idx >= len(poly_xy) or edge_idx >= len(edge_profiles):
            continue
        profile = edge_profiles[edge_idx]
        if not profile:
            continue

        all_counts = np.asarray(profile.get("all_counts", []), dtype=float)
        low_counts = np.asarray(profile.get("low_counts", []), dtype=float)
        edges_1d = np.asarray(profile.get("edges_1d", []), dtype=float)
        seg_len = float(profile.get("seg_len", 0.0) or 0.0)
        width = float(candidate.get("width", 0.0) or 0.0)
        if len(all_counts) < 6 or len(edges_1d) != len(all_counts) + 1 or seg_len < 0.8 or width < 0.45:
            continue

        density = all_counts + 0.10 * low_counts
        d_p10 = float(np.percentile(density, 10))
        d_p90 = float(np.percentile(density, 90))
        if d_p90 <= 0.0:
            continue

        flatness_ratio = (d_p90 - d_p10) / d_p90
        obs_count = int(candidate.get("observation_count", 0) or 0)
        if obs_count > 2:
            continue

        # Apply when cloud density is not strongly discriminative along the wall.
        if flatness_ratio > 0.45:
            continue

        a = np.asarray(poly_xy[edge_idx], dtype=float)
        b = np.asarray(poly_xy[(edge_idx + 1) % len(poly_xy)], dtype=float)
        seg = b - a
        seg_dir = seg / (np.linalg.norm(seg) + 1e-9)
        centroid = np.asarray(candidate.get("centroid_xy", []), dtype=float)
        if len(centroid) != 2:
            continue

        cur_t = float(np.dot(centroid - a, seg_dir))
        center_t = 0.5 * seg_len
        endpoint_margin = min(0.30, max(0.12, 0.25 * width))
        half_w = 0.5 * width

        shift_cap = min(0.85, 0.60 * seg_len)
        target_t = cur_t + np.clip(center_t - cur_t, -shift_cap, shift_cap)
        ambiguity = float(np.clip((0.45 - flatness_ratio) / 0.45, 0.0, 1.0))
        obs_factor = 1.0 if obs_count <= 1 else (0.8 if obs_count == 2 else 0.5)
        # Blend strength grows with ambiguity and weaker frame support.
        alpha = 0.16 + 0.44 * ambiguity * obs_factor
        new_t = (1.0 - alpha) * cur_t + alpha * target_t
        new_t = float(np.clip(new_t, endpoint_margin + half_w, seg_len - endpoint_margin - half_w))

        if abs(new_t - cur_t) < 0.08:
            continue

        p0 = a + seg_dir * (new_t - half_w)
        p1 = a + seg_dir * (new_t + half_w)
        candidate["centroid_xy"] = 0.5 * (p0 + p1)
        candidate["line_xy"] = np.array([p0, p1], dtype=float)
        candidate["low_evidence_center_prior_applied"] = True
        candidate["low_evidence_center_prior_old_t"] = round(cur_t, 3)
        candidate["low_evidence_center_prior_new_t"] = round(new_t, 3)
        candidate["low_evidence_center_prior_flatness"] = round(float(flatness_ratio), 3)


def _augment_openings_with_frame_observations(openings, semantic_frames, poly_xy, floor_z, room_height=None,
                                              furniture=None, scene_points_xyz=None):
    observations = []
    for pts in semantic_frames or []:
        observations.extend(
            _detect_opening_observations(
                pts,
                poly_xy,
                floor_z,
                room_height=room_height,
                blocked_mask=None,
                dist_threshold=0.16,
                min_near_points=16,
                min_left_right_support=2,
            )
        )

    fused = _fuse_opening_observations(observations)
    frame_profiles, _ = _aggregate_frame_edge_profiles(
        semantic_frames,
        poly_xy,
        floor_z,
        room_height=room_height,
        dist_threshold=0.16,
    ) if semantic_frames else ([], 0)
    edge_profiles = _wall_edge_band_profiles(
        scene_points_xyz,
        poly_xy,
        floor_z,
        room_height=room_height,
        blocked_mask=None,
        dist_threshold=0.16,
    ) if scene_points_xyz is not None and len(scene_points_xyz) else []
    matched = 0
    promoted = []
    promoted_edges = set()
    for candidate in fused:
        candidate = _refine_frame_opening_candidate_with_scene_profile(candidate, edge_profiles, poly_xy)
        candidate = _refine_window_candidate_with_temporal_profile(candidate, frame_profiles, poly_xy)
        candidate = _frame_opening_candidate_label(candidate, furniture)
        best_idx = None
        best_dist = float("inf")
        centroid = np.asarray(candidate["centroid_xy"], dtype=float)
        for idx, item in enumerate(openings):
            if int(item.get("edge_index", -1)) != int(candidate["edge_index"]):
                continue
            item_centroid = np.asarray(item.get("centroid_xy", []), dtype=float)
            if len(item_centroid) != 2:
                continue
            dist = float(np.linalg.norm(centroid - item_centroid))
            if dist <= 0.45 and dist < best_dist:
                best_idx = idx
                best_dist = dist
        if best_idx is not None:
            matched += 1
            existing_obs = int(openings[best_idx].get("observation_count", 1))
            existing_support = int(openings[best_idx].get("support_score", 0))
            candidate_obs = int(candidate.get("observation_count", 1))
            candidate_support = int(candidate.get("support_score", 0))

            adopt_geometry = False
            if str(openings[best_idx].get("kind", "")) == str(candidate.get("kind", "")):
                if bool(candidate.get("temporal_profile_refined")):
                    adopt_geometry = True
                elif candidate_obs > existing_obs:
                    adopt_geometry = True
                elif candidate_obs == existing_obs and candidate_support >= existing_support + 40:
                    adopt_geometry = True

            if adopt_geometry:
                old_centroid = np.asarray(openings[best_idx].get("centroid_xy", []), dtype=float)
                new_centroid = np.asarray(candidate.get("centroid_xy", []), dtype=float)
                new_line = np.asarray(candidate.get("line_xy", []), dtype=float)
                new_width = float(candidate.get("width", 0.0) or 0.0)
                if (
                    len(old_centroid) == 2
                    and len(new_centroid) == 2
                    and len(new_line) == 2
                    and 0.45 <= new_width <= 1.80
                    and float(np.linalg.norm(new_centroid - old_centroid)) <= 1.6
                ):
                    openings[best_idx]["centroid_xy"] = new_centroid
                    openings[best_idx]["line_xy"] = np.asarray(new_line, dtype=float)
                    openings[best_idx]["width"] = float(new_width)
                    for key in (
                        "scene_profile_support",
                        "temporal_profile_refined",
                        "temporal_profile_shift_m",
                        "temporal_profile_score_gain",
                        "temporal_profile_seen_frames",
                        "temporal_profile_cur_t",
                        "temporal_profile_best_t",
                        "temporal_profile_cur_score",
                        "temporal_profile_best_score",
                    ):
                        if key in candidate:
                            openings[best_idx][key] = candidate[key]

            openings[best_idx]["observation_count"] = max(
                existing_obs,
                candidate_obs,
            )
            openings[best_idx]["support_score"] = max(
                existing_support,
                candidate_support,
            )
        elif _frame_opening_candidate_is_promotable(candidate, furniture=furniture):
            promoted.append(candidate)
            promoted_edges.add(int(candidate.get("edge_index", -1)))

    if promoted:
        openings.extend(promoted)
        
    openings.sort(key=lambda item: (item.get("observation_count", 0), item.get("support_score", 0)), reverse=True)

    return {
        "frame_opening_observations": int(len(observations)),
        "frame_fused_openings": int(len(fused)),
        "frame_promoted_openings": int(len(promoted)),
        "frame_matched_openings": int(matched),
    }


def _inject_missing_large_wall_table_from_edge_evidence(furniture, openings, points_xyz, poly_xy, floor_z):
    if points_xyz is None or len(points_xyz) < 2000 or len(poly_xy) < 4:
        return furniture

    desk_edges = {
        int(item.get("wall_edge_index", -1))
        for item in (furniture or [])
        if str(item.get("label") or "") in {"desk_or_table_candidate", "l_shaped_desk_candidate"}
    }
    shelf_edges = {
        int(item.get("wall_edge_index", -1))
        for item in (furniture or [])
        if str(item.get("label") or "") == "shelf_candidate"
        and float(max(item.get("dimensions", [0.0, 0.0]) or [0.0])) >= 1.00
    }
    opening_edges = {int(item.get("edge_index", -1)) for item in (openings or [])}

    points_xy = points_xyz[:, :2]
    edge_dists, edge_proj, _, _ = _nearest_edge_geometry(points_xy, poly_xy)

    injected = list(furniture or [])
    for edge_idx in range(len(poly_xy)):
        if edge_idx in desk_edges or edge_idx in opening_edges or edge_idx in shelf_edges:
            continue

        t, seg_len = edge_proj[edge_idx]
        if seg_len < 2.0:
            continue

        near = (
            (edge_dists[:, edge_idx] <= 0.55)
            & (points_xyz[:, 2] >= floor_z + 0.08)
            & (points_xyz[:, 2] <= floor_z + 1.10)
        )
        idx = np.where(near)[0]
        if len(idx) < 1200:
            continue

        t_vals = t[idx]
        edge_a = np.asarray(poly_xy[edge_idx], dtype=float)
        edge_b = np.asarray(poly_xy[(edge_idx + 1) % len(poly_xy)], dtype=float)
        shelf_ranges = []
        for item in injected:
            if int(item.get("wall_edge_index", -1)) != edge_idx:
                continue
            if str(item.get("label") or "") != "shelf_candidate":
                continue
            shelf_xy = np.asarray(item.get("overlay_xy", item.get("rect_xy", item.get("polygon_xy", []))), dtype=float)
            if len(shelf_xy) < 3:
                continue
            shelf_t, _, _ = _project_points_to_segment(shelf_xy, edge_a, edge_b)
            s_lo = float(np.min(shelf_t))
            s_hi = float(np.max(shelf_t))
            shelf_ranges.append((s_lo - 0.18, s_hi + 0.18))

        if shelf_ranges:
            keep = np.ones(len(idx), dtype=bool)
            for s_lo, s_hi in shelf_ranges:
                keep &= ~((t_vals >= s_lo) & (t_vals <= s_hi))
            idx = idx[keep]
            if len(idx) < 500:
                continue
            t_vals = t[idx]

        t_lo = float(np.percentile(t_vals, 20))
        t_hi = float(np.percentile(t_vals, 85))
        long_span = float(np.clip(t_hi - t_lo, 1.10, 1.75))
        if long_span < 1.10:
            continue

        d_vals = edge_dists[idx, edge_idx]
        depth = float(np.clip(np.percentile(d_vals, 88), 0.65, 1.00))

        edge_a, edge_dir, interior_normal, _ = _edge_interior_basis(poly_xy, edge_idx)
        center_t = float(np.clip(0.5 * (t_lo + t_hi), 0.0, seg_len))
        center_xy = edge_a + edge_dir * center_t + interior_normal * (0.5 * depth)
        rect_xy = _build_oriented_rect(center_xy, edge_dir, interior_normal, long_span, depth)
        if not _overlay_geometry_is_plausible(rect_xy, center_xy, poly_xy, long_span, depth):
            continue

        sel_pts = points_xyz[idx]
        z_span = float(np.percentile(sel_pts[:, 2], 95) - np.percentile(sel_pts[:, 2], 5))
        if z_span < 0.18:
            continue

        item = {
            "label": "desk_or_table_candidate",
            "polygon_xy": np.asarray(rect_xy, dtype=float),
            "rect_xy": np.asarray(rect_xy, dtype=float),
            "centroid_xy": np.asarray(center_xy, dtype=float),
            "area": float(long_span * depth),
            "height": z_span,
            "z_min": float(np.percentile(sel_pts[:, 2], 5)),
            "z_max": float(np.percentile(sel_pts[:, 2], 95)),
            "wall_distance_p50": float(np.percentile(d_vals, 50)),
            "wall_distance_p90": float(np.percentile(d_vals, 90)),
            "points": int(len(idx)),
            "dimensions": [float(long_span), float(depth)],
            "observation_count": 2,
            "point_indices": idx.astype(int),
            "force_wall_attach": True,
        }
        item = _annotate_wall_attachment(item, poly_xy)
        item["wall_contact_type"] = _classify_wall_contact(item)
        injected.append(item)
        desk_edges.add(edge_idx)

    return injected


def _extract_frame_semantic_observations(semantic_frames, poly_xy, floor_z, room_height=None):
    try:
        from shapely.geometry import MultiPoint
    except Exception:
        return []

    if not semantic_frames:
        return []

    observations = []
    for frame_idx, pts in enumerate(semantic_frames):
        if pts is None or len(pts) < 80:
            continue

        z_hi = float(np.percentile(pts[:, 2], 99.5))
        if room_height is not None:
            z_hi = min(z_hi, float(floor_z + room_height + 0.25))
        keep = (pts[:, 2] >= floor_z - 0.06) & (pts[:, 2] <= z_hi)
        frame_pts = pts[keep]
        if len(frame_pts) < 40:
            continue

        edge_dists = []
        for i in range(len(poly_xy)):
            a = poly_xy[i]
            b = poly_xy[(i + 1) % len(poly_xy)]
            _, dist, _ = _project_points_to_segment(frame_pts[:, :2], a, b)
            edge_dists.append(dist)
        edge_dists = np.vstack(edge_dists).T
        min_wall_dist = edge_dists.min(axis=1)
        inside = _points_in_convex_polygon(frame_pts[:, :2], poly_xy)
        floor_band = frame_pts[:, 2] <= floor_z + 0.07
        ceiling_band = np.zeros(len(frame_pts), dtype=bool)
        if room_height is not None:
            ceiling_band = frame_pts[:, 2] >= float(floor_z + room_height - 0.10)

        candidate_mask = (
            inside
            & (~floor_band)
            & (~ceiling_band)
            & (frame_pts[:, 2] >= floor_z + 0.08)
            & (min_wall_dist >= 0.10)
            & (min_wall_dist <= 0.95)
        )
        if room_height is not None:
            candidate_mask &= frame_pts[:, 2] <= float(floor_z + max(room_height - 0.08, 0.9))
        if int(candidate_mask.sum()) < 18:
            continue

        candidate_points = frame_pts[candidate_mask]
        candidate_pcd = o3d.geometry.PointCloud()
        candidate_pcd.points = o3d.utility.Vector3dVector(candidate_points)
        labels = np.array(candidate_pcd.cluster_dbscan(eps=0.16, min_points=12, print_progress=False))
        candidate_idx = np.where(candidate_mask)[0]

        for label in sorted(set(labels.tolist())):
            if label < 0:
                continue
            idx = np.where(labels == label)[0]
            if len(idx) < 12:
                continue
            cluster_pts = candidate_points[idx]
            _mp = MultiPoint(cluster_pts[:, :2])
            hull = _mp.convex_hull
            if hull.is_empty or hull.geom_type != "Polygon":
                continue
            if hull.area >= 0.45:
                try:
                    import shapely as _shapely
                    _ch = _shapely.concave_hull(_mp, ratio=0.15, allow_holes=False)
                    if not _ch.is_empty and _ch.geom_type == "Polygon" and _ch.area >= 0.28 * hull.area:
                        hull = _ch
                except Exception:
                    pass
            hull = hull.buffer(0.03).buffer(-0.02)
            if hull.is_empty or hull.geom_type != "Polygon" or hull.area < 0.05:
                continue
            rect = hull.minimum_rotated_rectangle
            rect_xy = np.array(rect.exterior.coords[:-1], dtype=float)
            rect_edges = np.linalg.norm(np.roll(rect_xy, -1, axis=0) - rect_xy, axis=1)
            dim_long = float(np.max(rect_edges))
            dim_short = float(np.min(rect_edges))
            if dim_long < 0.20 or dim_short < 0.06:
                continue
            if dim_long >= 2.6 or dim_short >= 1.8:
                continue

            z_span = float(np.percentile(cluster_pts[:, 2], 95) - np.percentile(cluster_pts[:, 2], 5))
            if z_span < 0.12:
                continue
            z_lo = float(np.percentile(cluster_pts[:, 2], 5))
            z_hi_cluster = float(np.percentile(cluster_pts[:, 2], 95))
            cluster_wall_dists = np.min(edge_dists[candidate_idx[idx]], axis=1)
            wall_distance = float(np.min(cluster_wall_dists))
            wall_dist_p50 = float(np.percentile(cluster_wall_dists, 50))
            wall_dist_p90 = float(np.percentile(cluster_wall_dists, 90))
            if _looks_like_wall_strip(dim_long, dim_short, z_span, wall_dist_p50, wall_dist_p90):
                continue

            observations.append({
                "label": _cluster_label_from_geometry(dim_long, dim_short, z_span, wall_distance, float(hull.area)),
                "polygon_xy": np.array(hull.exterior.coords[:-1], dtype=float),
                "rect_xy": rect_xy,
                "centroid_xy": np.array(hull.centroid.coords[0], dtype=float),
                "area": float(hull.area),
                "height": z_span,
                "z_min": z_lo,
                "z_max": z_hi_cluster,
                "wall_distance_p50": wall_dist_p50,
                "wall_distance_p90": wall_dist_p90,
                "points": int(len(idx)),
                "dimensions": [dim_long, dim_short],
                "frame_idx": int(frame_idx),
                "observation_count": 1,
            })

    return observations


def _fuse_frame_semantic_observations(observations):
    if not observations:
        return []

    groups = []
    ranked = sorted(
        observations,
        key=lambda item: (item["points"], item["area"], item["height"]),
        reverse=True,
    )
    for obs in ranked:
        centroid = np.asarray(obs["centroid_xy"], dtype=float)
        best_group = None
        best_dist = float("inf")
        for group in groups:
            ref = np.asarray(group["centroid_xy"], dtype=float)
            dist = float(np.linalg.norm(centroid - ref))
            size_tol = max(0.28, 0.35 * max(obs["dimensions"][0], group["dimensions"][0]))
            if dist > min(size_tol, 0.65):
                continue
            if abs(float(obs["wall_distance_p50"]) - float(group["wall_distance_p50"])) > 0.45:
                continue
            if dist < best_dist:
                best_dist = dist
                best_group = group
        if best_group is None:
            groups.append({
                "members": [obs],
                "centroid_xy": centroid,
                "dimensions": list(obs["dimensions"]),
                "wall_distance_p50": float(obs["wall_distance_p50"]),
            })
        else:
            best_group["members"].append(obs)
            weights = np.array([max(member["points"], 1) for member in best_group["members"]], dtype=float)
            centroids = np.array([member["centroid_xy"] for member in best_group["members"]], dtype=float)
            best_group["centroid_xy"] = np.average(centroids, axis=0, weights=weights)
            best_group["dimensions"] = [
                float(np.percentile([member["dimensions"][0] for member in best_group["members"]], 75)),
                float(np.percentile([member["dimensions"][1] for member in best_group["members"]], 75)),
            ]
            best_group["wall_distance_p50"] = float(
                np.median([member["wall_distance_p50"] for member in best_group["members"]])
            )

    fused = []
    for group in groups:
        members = group["members"]
        representative = max(members, key=lambda item: (item["points"], item["area"], item["height"]))
        frame_ids = sorted({int(item["frame_idx"]) for item in members})
        fused_item = dict(representative)
        fused_item["centroid_xy"] = np.asarray(group["centroid_xy"], dtype=float)
        fused_item["dimensions"] = list(group["dimensions"])
        fused_item["wall_distance_p50"] = float(np.median([item["wall_distance_p50"] for item in members]))
        fused_item["wall_distance_p90"] = float(np.median([item["wall_distance_p90"] for item in members]))
        fused_item["points"] = int(sum(item["points"] for item in members))
        fused_item["height"] = float(np.percentile([item["height"] for item in members], 75))
        fused_item["z_min"] = float(np.percentile([item["z_min"] for item in members], 25))
        fused_item["z_max"] = float(np.percentile([item["z_max"] for item in members], 75))
        fused_item["observation_count"] = int(len(frame_ids))
        fused_item["frame_ids"] = frame_ids
        fused_item["label"] = _cluster_label_from_geometry(
            float(fused_item["dimensions"][0]),
            float(fused_item["dimensions"][1]),
            float(fused_item["height"]),
            float(fused_item["wall_distance_p50"]),
            float(fused_item.get("area", 0.0)),
        )
        fused_item["label"] = _refine_fused_frame_semantic_label(fused_item)
        fused.append(fused_item)

    fused.sort(
        key=lambda item: (item["observation_count"], item["points"], item["area"]),
        reverse=True,
    )
    return fused


def _frame_semantic_candidate_is_promotable(candidate):
    dim_long, dim_short = [float(v) for v in candidate.get("dimensions", [0.0, 0.0])]
    z_span = float(candidate.get("height", 0.0))
    wall_dist = float(candidate.get("wall_distance_p50", 0.0))
    obs_count = int(candidate.get("observation_count", 0))
    points = int(candidate.get("points", 0))
    area = float(candidate.get("area", 0.0))
    label = str(candidate.get("label") or "")
    footprint_proxy = max(area, 0.55 * dim_long * dim_short)

    relaxed_wall_table = (
        label == "desk_or_table_candidate"
        and wall_dist <= 0.35
        and float(candidate.get("wall_distance_p90", wall_dist) or wall_dist) <= 0.50
        and dim_long >= 1.0
        and dim_short >= 0.42
        and z_span <= 1.10
        and obs_count >= 2
        and points >= 140
    )
    if obs_count < 3 or points < 220:
        if not relaxed_wall_table:
            return False

    shelf_like = (
        label == "shelf_candidate"
        and dim_long <= 1.35
        and dim_short <= 0.32
        and wall_dist <= 0.22
        and footprint_proxy <= 0.75
    )
    table_like = (
        label in {"desk_or_table_candidate", "l_shaped_desk_candidate"}
        and (
            (
                dim_long <= 1.95
                and dim_short <= 1.10
                and 0.18 <= z_span <= 0.75
                and footprint_proxy <= 1.40
            )
            or (
                obs_count >= 8
                and points >= 1200
                and dim_long >= 1.05
                and dim_short >= 0.45
                and z_span <= 1.45
                and wall_dist <= 0.28
            )
        )
    )
    chair_like = (
        label in {"chair_or_small_furniture", "chair_candidate"}
        and
        dim_long <= 0.90
        and dim_short <= 0.70
        and 0.16 <= z_span <= 0.78
        and 0.18 <= wall_dist <= 0.65
        and area <= 0.55
    )
    return shelf_like or table_like or chair_like or relaxed_wall_table


def _frame_opening_near_wall_small_furniture(candidate, furniture):
    centroid = np.asarray(candidate.get("centroid_xy", []), dtype=float)
    edge_idx = int(candidate.get("edge_index", -1))
    candidate_kind = str(candidate.get("kind") or "")
    if len(centroid) != 2 or edge_idx < 0:
        return False

    for item in furniture or []:
        label = str(item.get("label") or "")
        clutter_labels = {
            "chair_or_small_furniture",
            "chair_candidate",
            "box_candidate",
            "shelf_candidate",
            "shelf_or_cabinet_candidate",
        }
        if candidate_kind == "door_candidate":
            clutter_labels.add("desk_or_table_candidate")
            clutter_labels.add("l_shaped_desk_candidate")
        if label not in clutter_labels:
            continue
        if candidate_kind == "window_candidate" and label in {
            "chair_or_small_furniture",
            "chair_candidate",
            "box_candidate",
            "shelf_candidate",
            "shelf_or_cabinet_candidate",
        }:
            continue
        wall_dist = float(item.get("wall_distance_p50", 1.0) or 1.0)
        if label in {"desk_or_table_candidate", "l_shaped_desk_candidate"} and wall_dist > 0.35:
            continue
        if not bool(item.get("wall_attached")) and wall_dist > 0.35:
            continue
        if int(item.get("wall_edge_index", -1)) != edge_idx:
            continue
        item_centroid = np.asarray(item.get("centroid_xy", []), dtype=float)
        if len(item_centroid) != 2:
            continue
        if label in {"chair_or_small_furniture", "chair_candidate", "box_candidate"}:
            max_dist = 0.65
        elif label in {"desk_or_table_candidate", "l_shaped_desk_candidate"}:
            max_dist = 0.55
        else:
            max_dist = 0.55
        if float(np.linalg.norm(item_centroid - centroid)) <= max_dist:
            return True

    return False


def _refine_frame_opening_candidate_with_scene_profile(candidate, edge_profiles, poly_xy):
    edge_idx = int(candidate.get("edge_index", -1))
    if edge_idx < 0 or edge_idx >= len(edge_profiles):
        return candidate

    profile = edge_profiles[edge_idx]
    if not profile:
        return candidate

    all_counts = np.asarray(profile.get("all_counts", []), dtype=float)
    low_counts = np.asarray(profile.get("low_counts", []), dtype=float)
    edges_1d = np.asarray(profile.get("edges_1d", []), dtype=float)
    seg_len = float(profile.get("seg_len", 0.0) or 0.0)
    width = float(candidate.get("width", 0.0) or 0.0)
    if len(all_counts) < 4 or len(edges_1d) != len(all_counts) + 1 or seg_len < 0.8 or width < 0.45:
        return candidate

    p90 = float(np.percentile(all_counts, 90)) if len(all_counts) else 0.0
    p10 = float(np.percentile(all_counts, 10)) if len(all_counts) else 0.0
    if p90 > 0.0 and (p90 - p10) <= 0.18 * p90:
        return candidate

    bin_width = float(np.median(np.diff(edges_1d))) if len(edges_1d) > 1 else 0.12
    if bin_width <= 1e-6:
        return candidate

    width_bins = int(max(4, round(width / bin_width)))
    width_bins = min(width_bins, len(all_counts))
    if width_bins < 4:
        return candidate

    endpoint_clearance = min(0.20, max(0.08, 0.20 * width))
    density = all_counts + 0.10 * low_counts
    line_xy = np.asarray(candidate.get("line_xy", []), dtype=float)
    if len(line_xy) == 2:
        a = np.asarray(poly_xy[edge_idx], dtype=float)
        b = np.asarray(poly_xy[(edge_idx + 1) % len(poly_xy)], dtype=float)
        center_t_arr, _, _ = _project_points_to_segment(candidate.get("centroid_xy", np.zeros(2, dtype=float)).reshape(1, 2), a, b)
        candidate_center_t = float(center_t_arr[0])
    else:
        candidate_center_t = 0.5 * seg_len
    obs_count = int(candidate.get("observation_count", 0) or 0)
    support_score = int(candidate.get("support_score", 0) or 0)
    proximity_weight = 0.0 if obs_count <= 1 else 0.20 * min(obs_count, 3)
    if support_score >= 650:
        proximity_weight += 0.10

    candidate_kind = str(candidate.get("kind") or "")
    avg_density = float(density.mean()) if len(density) else 1.0
    best = None
    if candidate_kind == "window_candidate":
        # Windows produce fewer LiDAR returns where the opening is.
        # Seek the sparsest window-width segment: lower point density = potential opening.
        # Only snap if the sparse region is clearly below average density (>12% below);
        # if the wall is essentially uniform, trust the frame-observation centre — that
        # IS the cloud evidence from the perimeter walk.
        for start in range(0, len(all_counts) - width_bins + 1):
            end = start + width_bins
            t0 = float(edges_1d[start])
            t1 = float(edges_1d[end])
            if t0 < endpoint_clearance or (seg_len - t1) < endpoint_clearance:
                continue
            center_t = 0.5 * (t0 + t1)
            density_score = float(density[start:end].mean())
            density_score += proximity_weight * abs(center_t - candidate_center_t)
            if best is None or density_score < best[0]:
                best = (density_score, start, end)
        if best is not None:
            sparse_density = float(density[best[1]:best[2]].mean())
            if avg_density <= 0.0 or sparse_density > avg_density * 0.88:
                return candidate
    else:
        for start in range(0, len(all_counts) - width_bins + 1):
            end = start + width_bins
            t0 = float(edges_1d[start])
            t1 = float(edges_1d[end])
            if t0 < endpoint_clearance or (seg_len - t1) < endpoint_clearance:
                continue
            center_t = 0.5 * (t0 + t1)
            density_score = float(density[start:end].mean())
            density_score -= proximity_weight * abs(center_t - candidate_center_t)
            if best is None or density_score > best[0]:
                best = (density_score, start, end)

    if best is None:
        return candidate

    _, start, end = best
    peak_density = float(np.mean(density[start:end]))
    target_max_width = 1.55 if candidate_kind == "window_candidate" else 1.05
    target_max_width = max(target_max_width, width * (1.90 if candidate_kind == "window_candidate" else 1.45))
    if candidate_kind == "window_candidate":
        # Expand the sparse window segment outward while density stays below the sparse threshold.
        expand_threshold = avg_density * 0.88
        while start > 0 and float(edges_1d[end] - edges_1d[start - 1]) <= target_max_width and density[start - 1] <= expand_threshold and float(edges_1d[start - 1]) >= endpoint_clearance:
            start -= 1
        while end < len(all_counts) and float(edges_1d[end + 1] - edges_1d[start]) <= target_max_width and density[end] <= expand_threshold and (seg_len - float(edges_1d[end + 1])) >= endpoint_clearance:
            end += 1
    else:
        expand_threshold = peak_density * 0.74
        while start > 0 and float(edges_1d[end] - edges_1d[start - 1]) <= target_max_width and density[start - 1] >= expand_threshold and float(edges_1d[start - 1]) >= endpoint_clearance:
            start -= 1
        while end < len(all_counts) and float(edges_1d[end + 1] - edges_1d[start]) <= target_max_width and density[end] >= expand_threshold and (seg_len - float(edges_1d[end + 1])) >= endpoint_clearance:
            end += 1

    t0 = float(edges_1d[start])
    t1 = float(edges_1d[end])
    profile_support = float(np.mean(all_counts[start:end])) if end > start else float(all_counts[start])
    a = np.asarray(poly_xy[edge_idx], dtype=float)
    b = np.asarray(poly_xy[(edge_idx + 1) % len(poly_xy)], dtype=float)
    seg = b - a
    seg_dir = seg / (np.linalg.norm(seg) + 1e-9)
    p0 = a + seg_dir * t0
    p1 = a + seg_dir * t1

    refined = dict(candidate)
    refined["line_xy"] = np.array([p0, p1], dtype=float)
    refined["width"] = float(np.linalg.norm(p1 - p0))
    refined["centroid_xy"] = 0.5 * (p0 + p1)
    refined["scene_profile_support"] = int(round(profile_support))
    return refined


def _frame_opening_candidate_is_promotable(candidate, furniture=None):
    candidate_kind = str(candidate.get("kind") or "")
    width = float(candidate.get("width", 0.0))
    obs_count = int(candidate.get("observation_count", 0))
    support_score = int(candidate.get("support_score", 0))
    scene_profile_support = int(candidate.get("scene_profile_support", 0) or 0)
    if _frame_opening_near_wall_small_furniture(candidate, furniture):
        return False

    max_width = 1.55 if candidate_kind == "window_candidate" else 1.10
    return (
        0.55 <= width <= max_width
        and (support_score >= 350 or scene_profile_support >= 1000)
        and (
            obs_count >= 3
            or (obs_count >= 2 and width <= 0.75)
            or (obs_count >= 2 and support_score >= 650 and width <= 1.00)
            or (obs_count >= 2 and scene_profile_support >= 1000 and width <= (1.45 if candidate_kind == "window_candidate" else 1.00))
            or (obs_count >= 1 and support_score >= 420 and width <= 0.65)
            or (obs_count >= 1 and width <= 0.70 and scene_profile_support >= 1200)
            or (obs_count >= 1 and width <= (1.45 if candidate_kind == "window_candidate" else 1.00) and scene_profile_support >= 1200)
        )
    )


def _frame_opening_candidate_label(candidate, furniture):
    labeled = dict(candidate)
    if str(labeled.get("kind") or "") != "door_candidate":
        return labeled

    width = float(labeled.get("width", 0.0))
    support_score = int(labeled.get("support_score", 0))
    scene_profile_support = int(labeled.get("scene_profile_support", 0) or 0)
    if width >= 1.0 and (support_score >= 650 or scene_profile_support >= 3000):
        labeled["kind"] = "window_candidate"

    return labeled


def _augment_scene_furniture_with_frame_observations(furniture, semantic_frames, poly_xy, floor_z, room_height=None):
    observations = _extract_frame_semantic_observations(semantic_frames, poly_xy, floor_z, room_height=room_height)
    fused = _fuse_frame_semantic_observations(observations)
    promoted = []
    matched = 0

    for candidate in fused:
        centroid = np.asarray(candidate["centroid_xy"], dtype=float)
        match_idx = None
        match_dist = float("inf")
        for idx, item in enumerate(furniture):
            match_info = _scene_furniture_frame_match_info(item, candidate)
            if match_info is None:
                continue
            dist = float(match_info["dist"])
            if dist < match_dist:
                match_dist = dist
                match_idx = idx
        if match_idx is not None:
            matched += 1
            furniture[match_idx] = _refine_scene_item_from_frame_candidate(
                furniture[match_idx],
                candidate,
            )
            furniture[match_idx]["frame_backed"] = True
            furniture[match_idx]["observation_count"] = max(
                int(furniture[match_idx].get("observation_count", 1)),
                int(candidate["observation_count"]),
            )
            furniture[match_idx]["frame_support_points"] = max(
                int(furniture[match_idx].get("frame_support_points", 0)),
                int(candidate["points"]),
            )
            continue

        if not _frame_semantic_candidate_is_promotable(candidate):
            continue

        promoted_candidate = dict(candidate)
        promoted_candidate["frame_backed"] = True
        promoted_candidate["frame_support_points"] = max(
            int(promoted_candidate.get("frame_support_points", 0)),
            int(promoted_candidate.get("points", 0)),
        )
        promoted.append(_annotate_wall_attachment(promoted_candidate, poly_xy))

    if promoted:
        furniture.extend(promoted)
        furniture.sort(
            key=lambda item: (
                int(item.get("observation_count", 1)),
                int(item.get("points", 0)),
                float(item.get("area", 0.0)),
            ),
            reverse=True,
        )

    return {
        "frame_observations": int(len(observations)),
        "frame_fused_candidates": int(len(fused)),
        "frame_promoted_candidates": int(len(promoted)),
        "frame_matched_candidates": int(matched),
    }


def _scene_debug_payload(scene_data):
    if not scene_data:
        return None
    edges = []
    max_edge_score = 0.0
    for item in scene_data.get("edges", []):
        edge_score = round(float(item.get("opening_evidence_score", 0.0)), 3)
        max_edge_score = max(max_edge_score, edge_score)
        edges.append({
            "edge_index": int(item.get("edge_index", -1)),
            "continuity": round(float(item.get("continuity", 0.0)), 3),
            "low_fill_ratio": round(float(item.get("low_fill_ratio", 0.0)), 3),
            "mid_fill_ratio": round(float(item.get("mid_fill_ratio", 0.0)), 3),
            "top_fill_ratio": round(float(item.get("top_fill_ratio", 0.0)), 3),
            "near_count": int(item.get("near_count", 0)),
            "candidate_gaps": int(item.get("candidate_gaps", 0)),
            "opening_evidence_score": edge_score,
            "best_gap_width": round(float(item.get("best_gap_width", 0.0)), 3),
            "best_gap_support_score": round(float(item.get("best_gap_support_score", 0.0)), 3),
            "best_gap_low_fill": round(float(item.get("best_gap_low_fill", 0.0)), 3),
            "best_gap_top_fill": round(float(item.get("best_gap_top_fill", 0.0)), 3),
            "best_gap_kind": item.get("best_gap_kind"),
            "best_gap_status": item.get("best_gap_status"),
            "best_gap_rejections": list(item.get("best_gap_rejections", [])),
        })
    diagnostics = dict(scene_data.get("diagnostics", {}))
    diagnostics["max_opening_evidence_score"] = round(max_edge_score, 3)
    return {
        "diagnostics": diagnostics,
        "furniture": [
            {
                "label": item.get("label"),
                "centroid_xy": [round(float(v), 3) for v in item.get("centroid_xy", [])],
                "overlay_centroid_xy": [round(float(v), 3) for v in item.get("overlay_centroid_xy", item.get("centroid_xy", []))],
                "dimensions": [round(float(v), 3) for v in item.get("dimensions", [])],
                "height": round(float(item.get("height", 0.0)), 3),
                "wall_distance_p50": round(float(item.get("wall_distance_p50", 0.0)), 3),
                "wall_distance_p90": round(float(item.get("wall_distance_p90", 0.0)), 3),
                "points": int(item.get("points", 0)),
                "observation_count": int(item.get("observation_count", 1)),
                "wall_attached": bool(item.get("wall_attached", False)),
                "wall_edge_index": int(item.get("wall_edge_index", -1)),
                "wall_contact_type": item.get("wall_contact_type"),
                "overlay_alignment": item.get("overlay_alignment"),
                "overlay_xy": [[round(float(x), 4), round(float(y), 4)] for x, y in item.get("overlay_xy", [])] or None,
            }
            for item in scene_data.get("furniture", [])
        ],
        "openings": [
            {
                "kind": item.get("kind"),
                "width": round(float(item.get("width", 0.0)), 3),
                "edge_index": int(item.get("edge_index", -1)),
                "centroid_xy": [round(float(v), 3) for v in item.get("centroid_xy", [])],
                "observation_count": int(item.get("observation_count", 1)),
            }
            for item in scene_data.get("openings", [])
        ],
        "edges": edges,
    }


def _analyze_scene_exports(p, poly, floor_z, room_height=None, semantic_frames=None):
    try:
        from shapely.geometry import MultiPoint, Polygon as ShapelyPolygon
    except Exception:
        return None

    pts = np.asarray(p.points)
    if len(pts) == 0:
        return None

    poly_xy = np.asarray(poly)[:, :2]
    poly_shape = ShapelyPolygon(poly_xy)
    if not poly_shape.is_valid or poly_shape.area <= 0.05:
        return None

    z_hi = float(np.percentile(pts[:, 2], 99.8))
    if room_height is not None:
        z_hi = min(z_hi, float(floor_z + room_height + 0.30))
    keep = (pts[:, 2] >= floor_z - 0.08) & (pts[:, 2] <= z_hi)
    pts_keep = pts[keep]
    if len(pts_keep) < 200:
        return None

    pcd_keep = o3d.geometry.PointCloud()
    pcd_keep.points = o3d.utility.Vector3dVector(pts_keep)
    pcd_keep, ind = pcd_keep.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    if len(pcd_keep.points) == 0:
        return None
    pcd_keep = pcd_keep.voxel_down_sample(voxel_size=0.03)
    pts_clean = np.asarray(pcd_keep.points)
    if len(pts_clean) == 0:
        return None

    inside = _points_in_convex_polygon(pts_clean[:, :2], poly_xy)
    edge_dists, edge_proj, _, min_wall_dist = _nearest_edge_geometry(pts_clean[:, :2], poly_xy)
    near_room = inside | (min_wall_dist <= 0.30)
    pts_clean = pts_clean[near_room]
    if len(pts_clean) == 0:
        return None

    edge_dists, edge_proj, _, min_wall_dist = _nearest_edge_geometry(pts_clean[:, :2], poly_xy)

    floor_band = pts_clean[:, 2] <= floor_z + 0.08
    ceiling_band = np.zeros(len(pts_clean), dtype=bool)
    if room_height is not None:
        ceiling_band = pts_clean[:, 2] >= float(floor_z + room_height - 0.12)
    wall_band = min_wall_dist <= 0.18
    object_candidate_mask = (
        (~floor_band)
        & (~ceiling_band)
        & _points_in_convex_polygon(pts_clean[:, :2], poly_xy)
        & (pts_clean[:, 2] >= floor_z + 0.05)
    )
    if room_height is not None:
        object_candidate_mask &= pts_clean[:, 2] <= float(floor_z + max(room_height - 0.10, 0.9))
    interior_candidate_mask = object_candidate_mask & (min_wall_dist >= 0.30)
    nearwall_candidate_mask = object_candidate_mask & (min_wall_dist > 0.18) & (min_wall_dist <= 0.70)
    wall_attached_candidate_mask = (
        object_candidate_mask
        & (pts_clean[:, 2] >= floor_z + 0.18)
        & (min_wall_dist > 0.08)
        & (min_wall_dist <= 0.24)
    )

    color_map = np.full((len(pts_clean), 3), [0.78, 0.78, 0.80], dtype=float)
    color_map[floor_band] = np.array([0.46, 0.63, 0.84], dtype=float)
    color_map[wall_band & ~floor_band & ~ceiling_band] = np.array([0.72, 0.74, 0.76], dtype=float)
    color_map[ceiling_band & ~floor_band] = np.array([0.55, 0.57, 0.66], dtype=float)
    base_color_map = color_map.copy()

    furniture = []
    furniture_mask = np.zeros(len(pts_clean), dtype=bool)
    rejected_wallish_clusters = 0
    injected_wall_table_fallback = 0
    palette = np.array([
        [0.86, 0.48, 0.18],
        [0.16, 0.60, 0.78],
        [0.22, 0.72, 0.42],
        [0.78, 0.34, 0.34],
        [0.55, 0.42, 0.78],
    ], dtype=float)

    def _collect_clusters(candidate_mask, eps, min_points):
        nonlocal rejected_wallish_clusters
        candidate_points = pts_clean[candidate_mask]
        if len(candidate_points) < min_points:
            return
        candidate_pcd = o3d.geometry.PointCloud()
        candidate_pcd.points = o3d.utility.Vector3dVector(candidate_points)
        labels = np.array(candidate_pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
        candidate_idx = np.where(candidate_mask)[0]
        for label in sorted(set(labels.tolist())):
            if label < 0:
                continue
            idx = np.where(labels == label)[0]
            if len(idx) < min_points:
                continue
            cluster_pts = candidate_points[idx]
            hull = MultiPoint(cluster_pts[:, :2]).convex_hull
            if hull.is_empty or hull.geom_type != "Polygon":
                continue
            hull = hull.buffer(0.02).buffer(-0.01)
            if hull.is_empty or hull.geom_type != "Polygon" or hull.area < 0.025:
                continue
            rect = hull.minimum_rotated_rectangle
            rect_xy = np.array(rect.exterior.coords[:-1], dtype=float)
            rect_edges = np.linalg.norm(np.roll(rect_xy, -1, axis=0) - rect_xy, axis=1)
            dim_long = float(np.max(rect_edges))
            dim_short = float(np.min(rect_edges))
            z_span = float(np.percentile(cluster_pts[:, 2], 95) - np.percentile(cluster_pts[:, 2], 5))
            z_lo = float(np.percentile(cluster_pts[:, 2], 5))
            z_hi_cluster = float(np.percentile(cluster_pts[:, 2], 95))
            cluster_wall_dists = np.min(edge_dists[candidate_idx[idx]], axis=1)
            wall_distance = float(np.min(cluster_wall_dists))
            wall_dist_p50 = float(np.percentile(cluster_wall_dists, 50))
            wall_dist_p90 = float(np.percentile(cluster_wall_dists, 90))
            if _looks_like_wall_strip(dim_long, dim_short, z_span, wall_dist_p50, wall_dist_p90):
                rejected_wallish_clusters += 1
                continue
            if dim_long >= 2.4 and dim_short >= 1.2:
                continue
            label_name = _cluster_label_from_geometry(dim_long, dim_short, z_span, wall_distance, float(hull.area))
            if label_name == "l_shaped_desk_candidate":
                try:
                    import shapely as _shapely

                    concave = _shapely.concave_hull(MultiPoint(cluster_pts[:, :2]), ratio=0.03, allow_holes=False)
                    if (
                        not concave.is_empty
                        and concave.geom_type == "Polygon"
                        and float(concave.area) >= 0.30 * float(hull.area)
                    ):
                        refined = concave.buffer(0.015).buffer(-0.008)
                        if not refined.is_empty and refined.geom_type == "Polygon" and refined.area >= 0.08:
                            hull = refined
                            rect = hull.minimum_rotated_rectangle
                            rect_xy = np.array(rect.exterior.coords[:-1], dtype=float)
                            rect_edges = np.linalg.norm(np.roll(rect_xy, -1, axis=0) - rect_xy, axis=1)
                            dim_long = float(np.max(rect_edges))
                            dim_short = float(np.min(rect_edges))
                except Exception:
                    pass
            color = palette[(len(furniture)) % len(palette)]
            color_map[candidate_idx[idx]] = color
            furniture_mask[candidate_idx[idx]] = True
            furniture_item = {
                "label": label_name,
                "polygon_xy": np.array(hull.exterior.coords[:-1], dtype=float),
                "rect_xy": rect_xy,
                "centroid_xy": np.array(hull.centroid.coords[0], dtype=float),
                "area": float(hull.area),
                "height": z_span,
                "z_min": z_lo,
                "z_max": z_hi_cluster,
                "wall_distance_p50": wall_dist_p50,
                "wall_distance_p90": wall_dist_p90,
                "points": int(len(idx)),
                "dimensions": [dim_long, dim_short],
                "point_indices": candidate_idx[idx].astype(int),
            }
            furniture_item = _annotate_wall_attachment(furniture_item, poly_xy)
            furniture_item["wall_contact_type"] = _classify_wall_contact(furniture_item)
            furniture.append(furniture_item)

    def _add_missing_wall_table_fallback():
        # Recover a large wall-attached desk footprint when clustering misses it
        # but the edge still has strong near-wall object evidence.
        nonlocal injected_wall_table_fallback
        existing_desk_edges = {
            int(item.get("wall_edge_index", -1))
            for item in furniture
            if str(item.get("label") or "") in {"desk_or_table_candidate", "l_shaped_desk_candidate"}
        }

        for edge_idx in range(len(poly_xy)):
            if edge_idx in existing_desk_edges:
                continue

            if any(
                int(item.get("wall_edge_index", -1)) == edge_idx
                and str(item.get("label") or "") == "shelf_candidate"
                and float(item.get("points", 0)) >= 2000
                and float(max(item.get("dimensions", [0.0, 0.0]) or [0.0])) >= 1.00
                for item in furniture
            ):
                continue

            near = (
                (~floor_band)
                & (~ceiling_band)
                & (edge_dists[:, edge_idx] <= 0.35)
                & (pts_clean[:, 2] >= floor_z + 0.08)
                & (pts_clean[:, 2] <= floor_z + 1.05)
            )
            idx = np.where(near)[0]
            if len(idx) < 220:
                continue

            t, seg_len = edge_proj[edge_idx]
            if seg_len < 1.2:
                continue
            t_vals = t[idx]
            t_lo = float(np.percentile(t_vals, 5))
            t_hi = float(np.percentile(t_vals, 95))
            long_span = t_hi - t_lo
            if long_span < 0.8 or long_span > 2.5:
                continue

            d_vals = edge_dists[idx, edge_idx]
            depth = float(np.percentile(d_vals, 85))
            if depth < 0.35 or depth > 1.30:
                continue

            edge_a, edge_dir, interior_normal, _ = _edge_interior_basis(poly_xy, edge_idx)
            center_t = float(np.clip(0.5 * (t_lo + t_hi), 0.0, seg_len))
            center_along = edge_a + edge_dir * center_t
            center_xy = center_along + interior_normal * (0.5 * depth)

            rect_xy = _build_oriented_rect(center_xy, edge_dir, interior_normal, long_span, depth)
            if not _overlay_geometry_is_plausible(rect_xy, center_xy, poly_xy, long_span, depth):
                continue

            sel_pts = pts_clean[idx]
            z_span = float(np.percentile(sel_pts[:, 2], 95) - np.percentile(sel_pts[:, 2], 5))
            if z_span < 0.18:
                continue

            item = {
                "label": "desk_or_table_candidate",
                "polygon_xy": np.array(rect_xy, dtype=float),
                "rect_xy": np.array(rect_xy, dtype=float),
                "centroid_xy": np.asarray(center_xy, dtype=float),
                "area": float(long_span * depth),
                "height": z_span,
                "z_min": float(np.percentile(sel_pts[:, 2], 5)),
                "z_max": float(np.percentile(sel_pts[:, 2], 95)),
                "wall_distance_p50": float(np.percentile(d_vals, 50)),
                "wall_distance_p90": float(np.percentile(d_vals, 90)),
                "points": int(len(idx)),
                "dimensions": [float(long_span), float(depth)],
                "point_indices": idx.astype(int),
                "observation_count": 2,
            }
            item = _annotate_wall_attachment(item, poly_xy)
            item["wall_contact_type"] = _classify_wall_contact(item)
            item["force_wall_attach"] = True
            furniture.append(item)
            injected_wall_table_fallback += 1
            color = palette[(len(furniture)) % len(palette)]
            color_map[idx] = color
            furniture_mask[idx] = True

    _collect_clusters(interior_candidate_mask, eps=0.18, min_points=45)
    _collect_clusters(nearwall_candidate_mask & (~furniture_mask), eps=0.14, min_points=30)
    _collect_clusters(wall_attached_candidate_mask & (~furniture_mask), eps=0.11, min_points=18)
    _add_missing_wall_table_fallback()

    frame_semantic_diag = {
        "frame_observations": 0,
        "frame_fused_candidates": 0,
        "frame_promoted_candidates": 0,
        "frame_matched_candidates": 0,
        "frame_opening_observations": 0,
        "frame_fused_openings": 0,
        "frame_promoted_openings": 0,
        "frame_matched_openings": 0,
        "frame_opening_profile_observations": 0,
        "structural_opening_candidates": 0,
        "injected_wall_table_fallback": 0,
    }
    if semantic_frames:
        frame_semantic_diag = _augment_scene_furniture_with_frame_observations(
            furniture,
            semantic_frames,
            poly_xy,
            floor_z,
            room_height=room_height,
        )

    rejected_unstable_furniture = 0
    if semantic_frames and furniture:
        filtered_furniture = []
        for item in furniture:
            if _scene_furniture_candidate_has_stable_support(item):
                filtered_furniture.append(item)
                continue

            rejected_unstable_furniture += 1
            point_indices = np.asarray(item.get("point_indices", []), dtype=int)
            if len(point_indices):
                furniture_mask[point_indices] = False
                color_map[point_indices] = base_color_map[point_indices]
        furniture = filtered_furniture

    rejected_redundant_furniture = 0
    if semantic_frames and furniture:
        furniture, rejected_redundant_furniture = _suppress_redundant_scene_furniture_items(furniture)
        if rejected_redundant_furniture:
            for item in furniture:
                point_indices = np.asarray(item.get("point_indices", []), dtype=int)
                if len(point_indices):
                    furniture_mask[point_indices] = True

    furniture = _postprocess_scene_furniture_labels(furniture)

    structure_mask = floor_band | ceiling_band | (wall_band & ~furniture_mask)

    opening_candidates, opening_profile_diag, edge_debug = _detect_structural_openings(
        pts_clean,
        poly_xy,
        floor_z,
        room_height=room_height,
        blocked_mask=furniture_mask,
        semantic_frames=semantic_frames,
    )
    frame_semantic_diag.update(opening_profile_diag)
    frame_semantic_diag["injected_wall_table_fallback"] = int(injected_wall_table_fallback)
    if semantic_frames:
        opening_diag = _augment_openings_with_frame_observations(
            opening_candidates,
            semantic_frames,
            poly_xy,
            floor_z,
            room_height=room_height,
            furniture=furniture,
            scene_points_xyz=pts_clean,
        )
        frame_semantic_diag.update(opening_diag)

    _snap_openings_to_coverage_minimum(
        opening_candidates,
        pts_clean,
        poly_xy,
        floor_z,
        room_height=room_height,
    )
    _apply_low_evidence_window_center_prior(
        opening_candidates,
        pts_clean,
        poly_xy,
        floor_z,
        room_height=room_height,
    )
    furniture = _inject_missing_large_wall_table_from_edge_evidence(
        furniture,
        opening_candidates,
        pts_clean,
        poly_xy,
        floor_z,
    )
    furniture = _refine_tall_wall_shelf_dimensions_from_scene_points(
        furniture,
        pts_clean,
        poly_xy,
        floor_z,
    )

    scene_points = pts_clean.copy()
    scene_cloud = o3d.geometry.PointCloud()
    scene_cloud.points = o3d.utility.Vector3dVector(scene_points)
    scene_cloud.colors = o3d.utility.Vector3dVector(color_map)

    clean_cloud = o3d.geometry.PointCloud()
    clean_cloud.points = o3d.utility.Vector3dVector(scene_points)
    clean_cloud.colors = o3d.utility.Vector3dVector(color_map)

    clutter_mask = furniture_mask | ((~structure_mask) & object_candidate_mask)
    view_mask = (wall_band & ~ceiling_band & (pts_clean[:, 2] >= floor_z + 0.10)) | furniture_mask
    view_points = pts_clean[view_mask]
    view_colors = color_map[view_mask]
    if len(view_points):
        view_cloud = o3d.geometry.PointCloud()
        view_cloud.points = o3d.utility.Vector3dVector(view_points)
        view_cloud.colors = o3d.utility.Vector3dVector(view_colors)
        view_cloud = view_cloud.voxel_down_sample(voxel_size=0.05)
    else:
        view_cloud = scene_cloud

    for index, item in enumerate(furniture):
        if "wall_edge_index" not in item or "wall_attached" not in item:
            _annotate_wall_attachment(item, poly_xy)
        if not item.get("wall_contact_type"):
            item["wall_contact_type"] = _classify_wall_contact(item)
        furniture[index] = _regularize_furniture_overlay(item, poly_xy)

    furniture = _dock_wall_side_chair_to_opposite_desk(furniture, poly_xy)
    furniture = _merge_ldesk_adjacent_overlaps(furniture, poly_xy, pts_xy=pts_clean[:, :2])

    furniture.sort(
        key=lambda item: (
            int(item.get("wall_attached", False)),
            int(item.get("observation_count", 1)),
            int(item.get("points", 0)),
        ),
        reverse=True,
    )
    opening_candidates.sort(
        key=lambda item: (
            int(item.get("observation_count", 1)),
            int(item.get("support_score", 0)),
        ),
        reverse=True,
    )

    return {
        "clean_cloud": clean_cloud,
        "scene_cloud": scene_cloud,
        "view_cloud": view_cloud,
        "furniture": furniture,
        "openings": opening_candidates,
        "edges": edge_debug,
        "poly_xy": poly_xy,
        "floor_z": float(floor_z),
        "room_height": None if room_height is None else float(room_height),
        "diagnostics": {
            "total_points": int(len(pts_clean)),
            "floor_points": int(floor_band.sum()),
            "wall_band_points": int(wall_band.sum()),
            "ceiling_points": int(ceiling_band.sum()),
            "object_candidate_points": int(object_candidate_mask.sum()),
            "interior_candidate_points": int(interior_candidate_mask.sum()),
            "nearwall_candidate_points": int(nearwall_candidate_mask.sum()),
            "wall_attached_candidate_points": int(wall_attached_candidate_mask.sum()),
            "furniture_points": int(furniture_mask.sum()),
            "rejected_wallish_clusters": int(rejected_wallish_clusters),
            "rejected_unstable_furniture": int(rejected_unstable_furniture),
            "rejected_redundant_furniture": int(rejected_redundant_furniture),
            "frame_observations": int(frame_semantic_diag.get("frame_observations", 0)),
            "frame_fused_candidates": int(frame_semantic_diag.get("frame_fused_candidates", 0)),
            "frame_promoted_candidates": int(frame_semantic_diag.get("frame_promoted_candidates", 0)),
            "frame_matched_candidates": int(frame_semantic_diag.get("frame_matched_candidates", 0)),
            "frame_opening_observations": int(frame_semantic_diag.get("frame_opening_observations", 0)),
            "frame_fused_openings": int(frame_semantic_diag.get("frame_fused_openings", 0)),
            "frame_promoted_openings": int(frame_semantic_diag.get("frame_promoted_openings", 0)),
            "frame_matched_openings": int(frame_semantic_diag.get("frame_matched_openings", 0)),
            "frame_opening_profile_observations": int(frame_semantic_diag.get("frame_opening_profile_observations", 0)),
            "structural_opening_candidates": int(frame_semantic_diag.get("structural_opening_candidates", 0)),
            "injected_wall_table_fallback": int(frame_semantic_diag.get("injected_wall_table_fallback", 0)),
        },
    }


def export_topdown_preview(scene_data, export_tag):
    if not scene_data:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = np.asarray(scene_data["clean_cloud"].points)
    colors = np.asarray(scene_data["clean_cloud"].colors)
    poly_xy = scene_data["poly_xy"]

    def _opening_line_xy(opening):
        line = opening.get("line_xy")
        if line is not None:
            line = np.asarray(line, dtype=float)
            if line.shape == (2, 2):
                return line

        centroid = np.asarray(opening.get("centroid_xy", []), dtype=float)
        edge_idx = int(opening.get("edge_index", -1))
        width = float(opening.get("width", 0.0) or 0.0)
        if len(centroid) != 2 or edge_idx < 0 or edge_idx >= len(poly_xy) or width <= 0.0:
            return None

        a = poly_xy[edge_idx]
        b = poly_xy[(edge_idx + 1) % len(poly_xy)]
        seg = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
        seg_len = float(np.linalg.norm(seg))
        if seg_len <= 1e-9:
            return None
        seg_dir = seg / seg_len
        half = 0.5 * width
        center = centroid.astype(float)
        return np.vstack([center - seg_dir * half, center + seg_dir * half])

    fig, ax = plt.subplots(figsize=(7, 7), dpi=180)
    if len(pts):
        ax.scatter(pts[:, 0], pts[:, 1], s=0.5, c=colors, alpha=0.55, linewidths=0)
    poly_closed = np.vstack([poly_xy, poly_xy[0]])
    ax.plot(poly_closed[:, 0], poly_closed[:, 1], color="black", linewidth=1.6)

    for furn in scene_data.get("furniture", []):
        hull = furn.get("overlay_xy")
        if hull is None:
            hull = furn.get("rect_xy")
        if hull is None:
            hull = furn["polygon_xy"]
        hull_closed = np.vstack([hull, hull[0]])
        is_shelf = str(furn.get("label") or "") == "shelf_candidate"
        fill_color = "#22d3ee" if is_shelf else "#d97706"
        edge_color = "#0f766e" if is_shelf else "#92400e"
        fill_alpha = 0.36 if is_shelf else 0.28
        edge_width = 2.2 if is_shelf else 1.0
        ax.fill(hull_closed[:, 0], hull_closed[:, 1], color=fill_color, alpha=fill_alpha, zorder=5)
        ax.plot(hull_closed[:, 0], hull_closed[:, 1], color=edge_color, linewidth=edge_width, zorder=6)
        c = np.asarray(furn.get("overlay_centroid_xy", furn["centroid_xy"]), dtype=float)
        furn_label = _display_label(furn["label"])
        ax.text(
            float(c[0]),
            float(c[1]),
            furn_label,
            fontsize=6.4 if is_shelf else 6.1,
            ha="center",
            va="center",
            color="#0f172a" if is_shelf else "#78350f",
            bbox={"boxstyle": "round,pad=0.16", "fc": "#ecfeff" if is_shelf else "#fef3c7", "ec": edge_color, "alpha": 0.95},
            zorder=7,
        )

    for opening in scene_data.get("openings", []):
        line = _opening_line_xy(opening)
        if line is None:
            continue
        color = "#2563eb" if opening["kind"] == "window_candidate" else "#059669"
        ax.plot(line[:, 0], line[:, 1], color=color, linewidth=2.2)
        c = np.asarray(opening.get("centroid_xy", []), dtype=float)
        if len(c) == 2:
            ax.text(
                float(c[0]),
                float(c[1]),
                _display_label(opening["kind"]),
                fontsize=5.0,
                ha="center",
                va="center",
                color=color,
                bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": color, "alpha": 0.82},
            )

    edge_debug = scene_data.get("edges", [])
    if edge_debug:
        for edge in edge_debug:
            edge_idx = int(edge.get("edge_index", -1))
            if edge_idx < 0 or edge_idx >= len(poly_xy):
                continue
            a = poly_xy[edge_idx]
            b = poly_xy[(edge_idx + 1) % len(poly_xy)]
            mid = 0.5 * (a + b)
            seg = b - a
            seg_len = float(np.linalg.norm(seg))
            if seg_len < 1e-6:
                continue
            normal = np.array([seg[1], -seg[0]], dtype=float) / seg_len
            centroid = np.mean(poly_xy, axis=0)
            if np.linalg.norm((mid + normal * 0.16) - centroid) < np.linalg.norm((mid - normal * 0.16) - centroid):
                normal = -normal
            pos = mid + normal * 0.16
            label = (
                f"E{edge_idx} c={float(edge.get('continuity', 0.0)):.2f}\n"
                f"l={float(edge.get('low_fill_ratio', 0.0)):.2f} "
                f"m={float(edge.get('mid_fill_ratio', 0.0)):.2f} "
                f"t={float(edge.get('top_fill_ratio', 0.0)):.2f}"
            )
            best_gap_width = float(edge.get("best_gap_width", 0.0))
            if best_gap_width > 0.0:
                edge_score = float(edge.get("opening_evidence_score", 0.0))
                rejects = list(edge.get("best_gap_rejections", []))
                reject_text = "/".join(
                    reason.replace("weak_", "").replace("height_profile_not_opening", "profile")
                    for reason in rejects[:2]
                )
                if not reject_text:
                    reject_text = str(edge.get("best_gap_status", "")).replace("accepted", "ok")
                label += f"\ns={edge_score:.2f} g={best_gap_width:.2f} {reject_text}"
            ax.text(
                float(pos[0]),
                float(pos[1]),
                label,
                fontsize=4.8,
                ha="center",
                va="center",
                color="#1f2937",
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "#94a3b8", "alpha": 0.78},
            )

    diag = scene_data.get("diagnostics", {})
    summary_lines = [
        f"floor={int(diag.get('floor_points', 0))}",
        f"wall={int(diag.get('wall_band_points', 0))}",
        f"ceiling={int(diag.get('ceiling_points', 0))}",
        f"attached={int(diag.get('wall_attached_candidate_points', 0))}",
        f"open_prof={int(diag.get('frame_opening_profile_observations', 0))}",
        f"open_struct={int(diag.get('structural_opening_candidates', 0))}",
        f"open_score={float(diag.get('max_opening_evidence_score', 0.0)):.2f}",
    ]
    ax.text(
        0.02,
        0.98,
        "\n".join(summary_lines),
        transform=ax.transAxes,
        fontsize=5.2,
        ha="left",
        va="top",
        color="#111827",
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#cbd5e1", "alpha": 0.88},
    )

    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_title("Top-down room preview", fontsize=9)
    name = output_path(f"topdown_preview_{export_tag}.png")
    fig.tight_layout(pad=0.3)
    fig.savefig(name, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    try:
        shutil.copyfile(name, output_path("topdown_preview_latest.png"))
    except Exception:
        pass
    log(f"Saved {name}")


def export_scene_debug(scene_data, export_tag):
    payload = _scene_debug_payload(scene_data)
    if not payload:
        return
    name = output_path(f"scene_debug_{export_tag}.json")
    with open(name, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    _latest_copy(name, output_path("scene_debug_latest.json"))
    log(f"Saved {name}")


# =========================================================
# EXPORT DXF
# =========================================================

def _edge_annotation_geometry(poly_xy):

    centroid = poly_xy.mean(axis=0)
    span = np.ptp(poly_xy, axis=0)
    scale = float(max(span.max(), 1.0))
    offset = max(0.18, min(0.40, 0.05 * scale))
    text_height = max(0.10, min(0.18, 0.025 * scale))

    annotations = []
    for i in range(len(poly_xy)):
        p1 = poly_xy[i]
        p2 = poly_xy[(i + 1) % len(poly_xy)]
        edge = p2 - p1
        length = float(np.linalg.norm(edge))
        if length < 1e-6:
            continue
        normal_a = np.array([edge[1], -edge[0]], dtype=float) / length
        normal_b = -normal_a
        mid = 0.5 * (p1 + p2)
        outward = (
            normal_a
            if np.linalg.norm((mid + normal_a * offset) - centroid) >= np.linalg.norm((mid + normal_b * offset) - centroid)
            else normal_b
        )
        text_pos = mid + outward * offset
        angle_deg = float(np.degrees(np.arctan2(edge[1], edge[0])))
        if angle_deg > 90.0 or angle_deg < -90.0:
            angle_deg += 180.0
        annotations.append({
            "mid": mid,
            "text_pos": text_pos,
            "length": length,
            "angle_deg": angle_deg,
        })

    return annotations, offset, text_height


def _latest_copy(src_name, latest_name):
    try:
        shutil.copyfile(src_name, latest_name)
    except Exception:
        pass


def _display_label(raw_label):
    special_labels = {
        "chair_or_small_furniture": "chair",
        "chair_candidate": "chair",
        "box_candidate": "box",
        "l_shaped_desk_candidate": "L-shaped desk",
    }
    if raw_label in special_labels:
        return special_labels[raw_label]
    return raw_label.replace("_candidate", "").replace("_", " ")


def _polygon_outer_coords(geom):
    if geom is None or getattr(geom, "is_empty", True):
        return None
    if geom.geom_type == "Polygon":
        return np.array(geom.exterior.coords[:-1], dtype=float)
    if geom.geom_type == "MultiPolygon":
        polys = list(geom.geoms)
        if not polys:
            return None
        poly = max(polys, key=lambda item: float(item.area))
        return np.array(poly.exterior.coords[:-1], dtype=float)
    return None


def _make_prism_mesh(base_xy, z0, z1, color):
    base_xy = np.asarray(base_xy, dtype=float)
    if len(base_xy) < 3 or z1 <= z0 + 1e-6:
        return None

    n = len(base_xy)
    verts = []
    for x, y in base_xy:
        verts.append([float(x), float(y), float(z0)])
    for x, y in base_xy:
        verts.append([float(x), float(y), float(z1)])

    tris = []
    for i in range(1, n - 1):
        tris.append([0, i + 1, i])
        tris.append([n, n + i, n + i + 1])
    for i in range(n):
        j = (i + 1) % n
        tris.append([i, j, n + j])
        tris.append([i, n + j, n + i])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(verts, dtype=float))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(tris, dtype=np.int32))
    mesh.paint_uniform_color(np.asarray(color, dtype=float))
    return mesh


def export_room_mesh(poly, export_tag, scene_data=None, room_height=None):
    floor_z = 0.0
    if scene_data is not None:
        floor_z = float(scene_data.get("floor_z", floor_z))
    if room_height is None and scene_data is not None:
        room_height = scene_data.get("room_height")
    if room_height is None:
        return

    poly_xy = np.asarray(poly)[:, :2]
    centroid = poly_xy.mean(axis=0)
    wall_thickness = float(max(0.10, min(0.16, 0.035 * max(np.ptp(poly_xy, axis=0).max(), 1.0))))
    wall_top = float(floor_z + room_height)

    mesh = o3d.geometry.TriangleMesh()

    floor_mesh = _make_prism_mesh(poly_xy, floor_z - 0.015, floor_z, [0.78, 0.79, 0.80])
    if floor_mesh is not None:
        mesh += floor_mesh

    for i in range(len(poly_xy)):
        p1 = poly_xy[i]
        p2 = poly_xy[(i + 1) % len(poly_xy)]
        edge = p2 - p1
        edge_len = float(np.linalg.norm(edge))
        if edge_len < 1e-6:
            continue
        normal_a = np.array([edge[1], -edge[0]], dtype=float) / edge_len
        normal_b = -normal_a
        mid = 0.5 * (p1 + p2)
        outward = normal_a if np.linalg.norm((mid + normal_a * wall_thickness) - centroid) > np.linalg.norm((mid + normal_b * wall_thickness) - centroid) else normal_b
        wall_rect = np.array([p1, p2, p2 + outward * wall_thickness, p1 + outward * wall_thickness], dtype=float)
        wall_mesh = _make_prism_mesh(wall_rect, floor_z, wall_top, [0.84, 0.84, 0.86])
        if wall_mesh is not None:
            mesh += wall_mesh

    for furn in (scene_data or {}).get("furniture", []):
        base_source = furn.get("rect_xy")
        if base_source is None:
            base_source = furn.get("polygon_xy")
        base_xy = np.asarray(base_source, dtype=float)
        if len(base_xy) < 3:
            continue
        z0 = max(float(floor_z), float(furn.get("z_min", floor_z)))
        z1 = min(float(wall_top - 0.05), max(z0 + 0.25, float(furn.get("z_max", z0 + 0.45))))
        furniture_mesh = _make_prism_mesh(base_xy, z0, z1, [0.70, 0.45, 0.18])
        if furniture_mesh is not None:
            mesh += furniture_mesh

    if len(mesh.vertices) == 0:
        return

    mesh.compute_vertex_normals()
    name = output_path(f"roommesh_{export_tag}.ply")
    if o3d.io.write_triangle_mesh(str(name), mesh, write_ascii=False):
        _latest_copy(name, output_path("roommesh_latest.ply"))
        log(f"Saved {name}")


def export_cloud_ply(p, export_tag):

    name = output_path(f"roomcloud_raw_{export_tag}.ply")
    if o3d.io.write_point_cloud(str(name), p, write_ascii=False):
        _latest_copy(name, output_path("roomcloud_raw_latest.ply"))
        log(f"Saved {name}")
    else:
        log(f"[WARN] Failed to save {name}", "info")


def export_primary_scene_cloud(scene_data, raw_cloud, export_tag):
    primary_cloud = None
    if scene_data:
        primary_cloud = scene_data.get("scene_cloud") or scene_data.get("clean_cloud") or scene_data.get("view_cloud")
    if primary_cloud is None:
        primary_cloud = raw_cloud

    name = output_path(f"roomcloud_{export_tag}.ply")
    if o3d.io.write_point_cloud(str(name), primary_cloud, write_ascii=False):
        _latest_copy(name, output_path("roomcloud_latest.ply"))
        log(f"Saved {name}")
    else:
        log(f"[WARN] Failed to save {name}", "info")


def export_scene_clouds(scene_data, export_tag):
    if not scene_data:
        return

    clean_cloud = scene_data.get("clean_cloud")
    scene_cloud = scene_data.get("scene_cloud")
    view_cloud = scene_data.get("view_cloud")
    if clean_cloud is not None and len(clean_cloud.points):
        name = output_path(f"roomcloud_clean_{export_tag}.ply")
        if o3d.io.write_point_cloud(str(name), clean_cloud, write_ascii=False):
            _latest_copy(name, output_path("roomcloud_clean_latest.ply"))
            log(f"Saved {name}")
    if scene_cloud is not None and len(scene_cloud.points):
        name = output_path(f"roomcloud_scene_{export_tag}.ply")
        if o3d.io.write_point_cloud(str(name), scene_cloud, write_ascii=False):
            _latest_copy(name, output_path("roomcloud_scene_latest.ply"))
            log(f"Saved {name}")
    if view_cloud is not None and len(view_cloud.points):
        name = output_path(f"roomcloud_view_{export_tag}.ply")
        if o3d.io.write_point_cloud(str(name), view_cloud, write_ascii=False):
            _latest_copy(name, output_path("roomcloud_view_latest.ply"))
            log(f"Saved {name}")


def export_dxf(poly, export_tag, floor_area=None, perimeter=None, room_height=None, scene_data=None):

    import ezdxf
    from ezdxf.enums import TextEntityAlignment
    from ezdxf import units
    try:
        from shapely.geometry import Polygon as ShapelyPolygon
    except Exception:
        ShapelyPolygon = None

    name = output_path(f"floorplan_{export_tag}.dxf")

    doc = ezdxf.new("R2010")
    doc.units = units.M
    msp = doc.modelspace()

    if "ROOM" not in doc.layers:
        doc.layers.add("ROOM", dxfattribs={"color": 7})
    if "WALLS" not in doc.layers:
        doc.layers.add("WALLS", dxfattribs={"color": 8})
    if "DIMENSIONS" not in doc.layers:
        doc.layers.add("DIMENSIONS", dxfattribs={"color": 3})
    if "NOTES" not in doc.layers:
        doc.layers.add("NOTES", dxfattribs={"color": 5})
    if "FURNITURE" not in doc.layers:
        doc.layers.add("FURNITURE", dxfattribs={"color": 30})
    if "OPENINGS" not in doc.layers:
        doc.layers.add("OPENINGS", dxfattribs={"color": 140})

    pts = [(p[0], p[1]) for p in poly]
    pts.append(pts[0])
    msp.add_lwpolyline(pts, dxfattribs={"layer": "ROOM"})

    poly_xy = np.asarray(poly)[:, :2]
    if ShapelyPolygon is not None:
        wall_thickness = float(max(0.10, min(0.16, 0.035 * max(np.ptp(poly_xy, axis=0).max(), 1.0))))
        outer_coords = _polygon_outer_coords(ShapelyPolygon(poly_xy).buffer(wall_thickness, join_style=2))
        if outer_coords is not None and len(outer_coords) >= 3:
            outer_pts = [(float(x), float(y)) for x, y in outer_coords]
            outer_pts.append(outer_pts[0])
            msp.add_lwpolyline(outer_pts, dxfattribs={"layer": "WALLS"})

    annotations, offset, text_height = _edge_annotation_geometry(poly_xy)
    for item in annotations:
        mid = item["mid"]
        text_pos = item["text_pos"]
        msp.add_line(
            (float(mid[0]), float(mid[1])),
            (float(text_pos[0]), float(text_pos[1])),
            dxfattribs={"layer": "DIMENSIONS"},
        )
        text = msp.add_text(
            f"{item['length']:.2f} m",
            dxfattribs={"layer": "DIMENSIONS", "height": text_height, "rotation": item["angle_deg"]},
        )
        text.set_placement(
            (float(text_pos[0]), float(text_pos[1])),
            align=TextEntityAlignment.MIDDLE_CENTER,
        )

    summary_lines = []
    if floor_area is not None:
        summary_lines.append(f"Area: {float(floor_area):.3f} m^2")
    if perimeter is not None:
        summary_lines.append(f"Perimeter: {float(perimeter):.3f} m")
    if room_height is not None:
        summary_lines.append(f"Height: {float(room_height):.3f} m")

    if summary_lines:
        max_xy = poly_xy.max(axis=0)
        note_anchor = max_xy + np.array([1.8 * offset, 1.8 * offset], dtype=float)
        line_step = 1.6 * text_height
        for idx, line in enumerate(summary_lines):
            y = float(note_anchor[1] - idx * line_step)
            note = msp.add_text(
                line,
                dxfattribs={"layer": "NOTES", "height": text_height},
            )
            note.set_placement(
                (float(note_anchor[0]), y),
                align=TextEntityAlignment.LEFT,
            )

    if scene_data:
        for furn in scene_data.get("furniture", []):
            hull_source = furn.get("overlay_xy")
            if hull_source is None:
                hull_source = furn.get("rect_xy")
            if hull_source is None:
                hull_source = furn["polygon_xy"]
            hull = np.asarray(hull_source, dtype=float)
            if len(hull) >= 3:
                hull_pts = [(float(x), float(y)) for x, y in hull]
                hull_pts.append(hull_pts[0])
                msp.add_lwpolyline(hull_pts, dxfattribs={"layer": "FURNITURE"})
                centroid = np.asarray(furn.get("overlay_centroid_xy", furn["centroid_xy"]), dtype=float)
                span = np.ptp(np.asarray(hull, dtype=float), axis=0)
                mark = max(0.08, 0.12 * float(max(span.max(), 0.4)))
                msp.add_line(
                    (float(centroid[0] - mark), float(centroid[1] - mark)),
                    (float(centroid[0] + mark), float(centroid[1] + mark)),
                    dxfattribs={"layer": "FURNITURE"},
                )
                msp.add_line(
                    (float(centroid[0] - mark), float(centroid[1] + mark)),
                    (float(centroid[0] + mark), float(centroid[1] - mark)),
                    dxfattribs={"layer": "FURNITURE"},
                )
                label = msp.add_text(
                    _display_label(furn["label"]),
                    dxfattribs={"layer": "FURNITURE", "height": max(0.16, text_height * 1.1)},
                )
                label.set_placement((float(centroid[0]), float(centroid[1] + 1.8 * mark)), align=TextEntityAlignment.MIDDLE_CENTER)

        for opening in scene_data.get("openings", []):
            line = opening["line_xy"]
            p0 = tuple(float(v) for v in line[0])
            p1 = tuple(float(v) for v in line[1])
            msp.add_line(p0, p1, dxfattribs={"layer": "OPENINGS"})
            c = opening["centroid_xy"]
            label = msp.add_text(
                _display_label(opening["kind"]),
                dxfattribs={"layer": "OPENINGS", "height": max(0.08, text_height * 0.75)},
            )
            label.set_placement((float(c[0]), float(c[1])), align=TextEntityAlignment.MIDDLE_CENTER)

    doc.saveas(str(name))
    _latest_copy(name, output_path("floorplan_latest.dxf"))

    log(f"Saved {name}")


def update_dxf_summary(export_tag, floor_area=None, perimeter=None, room_height=None):
    import ezdxf

    name = output_path(f"floorplan_{export_tag}.dxf")
    if not name.exists():
        return False

    summary_lines = {}
    if floor_area is not None:
        summary_lines["Area:"] = f"Area: {float(floor_area):.3f} m^2"
    if perimeter is not None:
        summary_lines["Perimeter:"] = f"Perimeter: {float(perimeter):.3f} m"
    if room_height is not None:
        summary_lines["Height:"] = f"Height: {float(room_height):.3f} m"
    if not summary_lines:
        return False

    doc = ezdxf.readfile(str(name))
    updated = False
    for entity in doc.modelspace().query("TEXT[layer=='NOTES']"):
        text = str(getattr(entity.dxf, "text", "") or "")
        for prefix, replacement in summary_lines.items():
            if text.startswith(prefix):
                entity.dxf.text = replacement
                updated = True
                break

    if not updated:
        return False

    doc.saveas(str(name))
    _latest_copy(name, output_path("floorplan_latest.dxf"))
    return True


# =========================================================
# RECONSTRUCT
# =========================================================

def reconstruct(p, skip_world_align=False, walking_wall_hints=None, walking_ceiling_hints=None,
                walking_ceiling_z_hint=None, walking_quality=None, walking_support_metrics=None,
                repeat_walk_height_hint=None, semantic_cloud=None, semantic_frames=None,
                allow_guided_live_borderline=False, return_low_confidence_report=False,
                suppress_exports=False):
    global LAST_WALKING_FIT_INFO
    LAST_WALKING_FIT_INFO = None

    log(f"[DEBUG] reconstruct() input: {len(p.points)} points", "debug")

    # --- 1. World-align (RANSAC floor → gravity align) ---
    # For walking scans: each frame was already gravity-aligned before ICP, and
    # all ICP transforms were projected to SE(2) (no Z-tilt), so the merged
    # cloud is already upright with the floor at Z≈0.  We only need a Z-shift
    # to put the floor percentile exactly at 0.  Calling world_align on the
    # merged cloud would find the wrong surface (ICP-drift noise on the floor
    # makes RANSAC pick a tilted patch) and apply a spurious rotation that puts
    # all wall normals in the dead-zone, killing classification.
    #
    # For static / file scans: run full world_align as before.
    if skip_world_align:
        _pts_z = np.asarray(p.points)[:, 2]
        _floor_z0 = float(np.percentile(_pts_z, CONFIG.get("floor_percentile", 2)))
        _arr = np.asarray(p.points).copy()
        _arr[:, 2] -= _floor_z0
        p_aligned = o3d.geometry.PointCloud()
        p_aligned.points = o3d.utility.Vector3dVector(_arr)
        p = p_aligned
        # Apply the same Z-shift to the semantic cloud so that it stays in the
        # same coordinate frame as the shifted merged cloud.  Without this the
        # floor in semantic_cloud sits at the original z (e.g. -0.26 m) while
        # floor_z derived from the shifted merged cloud is ≈0, causing
        # _analyze_scene_exports to cut off all actual floor points in the
        # height-filter keep = pts[:,2] >= floor_z - 0.08 = -0.08, and
        # floor_band = pts[:,2] <= floor_z + 0.08 = 0.08 to find almost none.
        if semantic_cloud is not None and len(semantic_cloud.points):
            _sc_pts = np.asarray(semantic_cloud.points).copy()
            _sc_pts[:, 2] -= _floor_z0
            _sc_shifted = o3d.geometry.PointCloud()
            _sc_shifted.points = o3d.utility.Vector3dVector(_sc_pts)
            if semantic_cloud.has_colors():
                _sc_shifted.colors = semantic_cloud.colors
            if semantic_cloud.has_normals():
                _sc_shifted.normals = semantic_cloud.normals
            semantic_cloud = _sc_shifted
        if semantic_frames:
            semantic_frames = [
                (np.asarray(f, dtype=float).copy() if not isinstance(f, np.ndarray) else f.copy())
                for f in semantic_frames
            ]
            for f in semantic_frames:
                f[:, 2] -= _floor_z0
        log(f"[INFO] skip_world_align: Z-shift by {-_floor_z0:.3f} m", "info")
        _shifted = np.asarray(p.points)
        log(f"[INFO] Merged cloud Z=[{_shifted[:,2].min():.2f}, {_shifted[:,2].max():.2f}] m  "
            f"X=[{_shifted[:,0].min():.2f}, {_shifted[:,0].max():.2f}] m  "
            f"Y=[{_shifted[:,1].min():.2f}, {_shifted[:,1].max():.2f}] m  "
            f"({len(_shifted):,} pts)", "info")
    else:
        try:
            p = world_align(p)
            log(f"[DEBUG] After world_align: {len(p.points)} points", "debug")
        except RuntimeError as _wa_err:
            log(f"[WARN] world_align failed ({_wa_err}); applying Z-shift only", "info")
            _pts_z = np.asarray(p.points)[:, 2]
            _floor_z0 = float(np.percentile(_pts_z, CONFIG.get("floor_percentile", 2)))
            _arr = np.asarray(p.points).copy()
            _arr[:, 2] -= _floor_z0
            _pa = o3d.geometry.PointCloud()
            _pa.points = o3d.utility.Vector3dVector(_arr)
            p = _pa

    # --- 2. Statistical outlier removal on full cloud ---
    # Walking scans skip this step: each frame was already individually
    # gravity-aligned and voxel-downsampled before ICP, so the merged cloud
    # is clean.  Aggressive outlier removal trims the sparse high-Z wall
    # points that are only seen from a few distant frames — those are
    # exactly the points that give good height estimates (z_top of far wall
    # ≈ ceiling height).  Removing them caps the cloud z_max at ~1.7 m and
    # causes the height estimate to be far too low.
    if not skip_world_align:
        p, _ = p.remove_statistical_outlier(
            CONFIG["outlier_neighbors"], CONFIG["outlier_std"]
        )
        if len(p.points) < 200:
            raise RuntimeError(
                f"Point cloud too sparse after outlier removal ({len(p.points)} points). "
                "Verify scan quality and coverage."
            )
        log(f"[DEBUG] After outlier removal: {len(p.points)} points", "debug")
    else:
        log(f"[DEBUG] Outlier removal skipped (walking mode — frames pre-cleaned per-frame)", "debug")
    if not p.has_normals():
        p.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=30))

    # --- 2b. XY range clip — remove returns beyond max_range_m from sensor --
    # After world_align on a STATIC scan the sensor sits at XY=(0,0), so clipping
    # at max_range_m from origin removes outdoor returns through windows/doors.
    # For WALKING scans the ICP origin is wherever frame 0 was captured — it is
    # NOT the room centre.  Clipping from that origin deletes walls on the far
    # side of the room (typically 3–4 m away), leaving only 1–2 walls visible
    # and causing "all walls same direction" failures.
    # Solution: skip the XY clip for walking scans (skip_world_align=True).
    # Walking data never contains outdoor returns because the scan stays inside
    # the room for the full capture duration.
    _max_r = CONFIG.get("max_range_m")
    if _max_r is not None:
        _pts_np = np.asarray(p.points)
        if skip_world_align:
            # Walking mode: clip around merged cloud XY centroid, not the
            # ICP origin (which is wherever frame 0 happened to be captured).
            _cx = float(_pts_np[:, 0].mean())
            _cy = float(_pts_np[:, 1].mean())
        else:
            # Static mode: world_align placed sensor at XY=(0,0).
            _cx, _cy = 0.0, 0.0
        _xy_r2 = (_pts_np[:, 0] - _cx) ** 2 + (_pts_np[:, 1] - _cy) ** 2
        _mask  = _xy_r2 <= _max_r ** 2
        if _mask.sum() < len(_pts_np):
            log(
                f"[INFO] XY range clip ({_max_r} m from centroid): removed "
                f"{(~_mask).sum():,} pts, {_mask.sum():,} remaining",
                "info",
            )
            p = p.select_by_index(np.where(_mask)[0])

    # --- 3. Extract ALL planes (floor + walls) from the full aligned cloud ---
    #        remove_clutter() is NOT called here; it would discard wall points.
    _walking_wall_slice = None
    if skip_world_align:
        # Walking mode: the merged cloud has frames gravity-aligned to Z=0 from
        # many positions.  RANSAC run on the whole merged cloud finds only
        # horizontal slabs (floor, ceiling, desk tops, corridor returns seen
        # from different positions) because they are the largest planar regions
        # across the full multi-frame extent.  Walls are never reached in 20
        # iterations.
        #
        # Fix: stratified extraction — three separate RANSAC passes each
        # restricted to the Z band that contains the target surface:
        #   Pass 1 (floor):   Z < 0.25 m  →  looks only at floor-level returns
        #   Pass 2 (walls):   0.15 m < Z < ceil_est  →  walls span this band;
        #                     floor and ceiling are excluded so RANSAC finds
        #                     vertical planes naturally
        #   Pass 3 (ceiling): Z > ceil_est  →  optional, found if sensor can see it
        #
        # Re-voxel first so all three slices have a uniform, manageable density.
        _walk_vox = CONFIG.get("voxel_size", 0.04) * 3.0
        _p_vox = p.voxel_down_sample(_walk_vox)
        _p_vox.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=30))
        log(
            f"[INFO] Walking stratified RANSAC: re-voxeled to {_walk_vox:.2f} m "
            f"→ {len(_p_vox.points):,} pts",
            "info",
        )

        _alz    = np.asarray(_p_vox.points)[:, 2]
        _z_max  = float(_alz.max())
        # Ceiling estimate: use a broader top band than the raw cloud maximum.
        # In walking scans the highest few returns are often sparse and do not
        # form a plane. Prefer the per-frame ceiling hint when available and
        # leave a wider buffer so the ceiling slice keeps enough support.
        _ceil_from_cloud = _z_max - 0.35
        if walking_ceiling_z_hint is not None:
            _ceil_from_hint = float(walking_ceiling_z_hint) - 0.18
            ceil_est = max(1.45, min(_ceil_from_cloud, _ceil_from_hint))
        else:
            ceil_est = max(1.5, _ceil_from_cloud)

        def _slice(pcd, z_lo, z_hi):
            pts = np.asarray(pcd.points)
            idx = np.where((pts[:, 2] >= z_lo) & (pts[:, 2] <= z_hi))[0]
            if len(idx) == 0:
                return None
            s = pcd.select_by_index(idx)
            if not s.has_normals():
                s.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=30))
            return s

        def _filter_by_local_normal(pcd, min_abs_nz=None, max_abs_nz=None):
            """Keep only points whose LOCAL normal has the requested tilt."""
            if pcd is None or len(pcd.points) == 0:
                return None
            if not pcd.has_normals():
                pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=30))
            nrms = np.asarray(pcd.normals)
            keep = np.ones(len(nrms), dtype=bool)
            abs_nz = np.abs(nrms[:, 2])
            if min_abs_nz is not None:
                keep &= abs_nz >= min_abs_nz
            if max_abs_nz is not None:
                keep &= abs_nz <= max_abs_nz
            idx = np.where(keep)[0]
            if len(idx) == 0:
                return None
            out = pcd.select_by_index(idx)
            if len(out.points):
                out.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=30))
            return out

        # Pass 1: floor slice
        _floor_slice = _filter_by_local_normal(
            _slice(_p_vox, -0.30, 0.25),
            min_abs_nz=0.75,
        )
        _floor_planes = []
        if _floor_slice and len(_floor_slice.points) >= CONFIG["min_plane_points"]:
            log(
                f"[INFO]   Floor slice: {len(_floor_slice.points)} pts "
                f"(Z < 0.25 m, local |n_z| >= 0.75)",
                "info",
            )
            for _fp in extract_planes(_floor_slice):
                if abs(_fp["normal"][2]) > 0.80:
                    _floor_planes.append(_fp)
            if not _floor_planes:
                _fallback_floor = _make_horizontal_plane_from_slice(_floor_slice)
                if _fallback_floor is not None:
                    _floor_planes.append(_fallback_floor)
                    log(
                        f"[INFO]   Floor slice fallback: synthesised horizontal floor plane "
                        f"(area={get_plane_area(_fallback_floor):.2f} m²)",
                        "info",
                    )

        # Pass 2: wall slice (excludes floor and ceiling)
        _wall_slice = _filter_by_local_normal(
            _slice(_p_vox, 0.15, ceil_est),
            max_abs_nz=0.35,
        )
        _walking_wall_slice = _wall_slice
        _wall_planes = []
        if _wall_slice and len(_wall_slice.points) >= CONFIG["min_plane_points"]:
            log(
                f"[INFO]   Wall slice: {len(_wall_slice.points)} pts "
                f"(0.15 m – {ceil_est:.2f} m, local |n_z| <= 0.35)",
                "info",
            )
            _wall_planes_raw = extract_planes(_wall_slice)
            _wall_planes = [
                _wp for _wp in _wall_planes_raw
                if abs(float(get_plane_normal(_wp)[2])) <= 0.45
            ]
            if len(_wall_planes) < len(_wall_planes_raw):
                log(
                    f"[DEBUG]   Wall slice: filtered "
                    f"{len(_wall_planes_raw) - len(_wall_planes)} horizontal artefact plane(s)",
                    "debug",
                )

        # Pass 3: ceiling slice
        _ceil_slice = _filter_by_local_normal(
            _slice(_p_vox, ceil_est, _z_max + 0.10),
            min_abs_nz=0.75,
        )
        _ceil_planes = []
        if _ceil_slice and len(_ceil_slice.points) >= CONFIG["min_plane_points"]:
            log(
                f"[INFO]   Ceiling slice: {len(_ceil_slice.points)} pts "
                f"(Z > {ceil_est:.2f} m, local |n_z| >= 0.75)",
                "info",
            )
            for _cp in extract_planes(_ceil_slice):
                if abs(_cp["normal"][2]) > 0.80:
                    _ceil_planes.append(_cp)
            if not _ceil_planes:
                _fallback_ceil = _make_horizontal_plane_from_slice(
                    _ceil_slice,
                    z_percentile=95,
                )
                if _fallback_ceil is not None and float(np.median(_fallback_ceil["shell_coords"][:, 2])) >= max(1.7, _z_max - 0.60):
                    _ceil_planes.append(_fallback_ceil)
                    log(
                        f"[INFO]   Ceiling slice fallback: synthesised horizontal ceiling plane "
                        f"(area={get_plane_area(_fallback_ceil):.2f} m²)",
                        "info",
                    )

        planes = _floor_planes + _wall_planes + _ceil_planes
        log(
            f"[INFO] Stratified extraction: {len(_floor_planes)} floor, "
            f"{len(_wall_planes)} mid-band, {len(_ceil_planes)} ceiling planes",
            "info",
        )
    else:
        planes = extract_planes(p)
    log(f"[DEBUG] Total planes after filtering: {len(planes)}", "info")

    if not planes:
        raise RuntimeError(
            "No planes detected. "
            "Ensure the scan covers the floor and at least 3 walls. "
            "Try lowering min_plane_area in CONFIG."
        )

    # --- 4. Classify into floors / walls ---
    floors, walls = classify(planes)
    initial_wall_count = len(walls)
    log(f"[DEBUG] Classification: {len(floors)} floors, {len(walls)} walls", "debug")

    # --- 4a. Merge same-face wall sub-patches (residual tilt artefacts) ----
    walls = _merge_wall_faces(walls)
    post_global_merge_wall_count = len(walls)
    log(f"[DEBUG] After wall face-dedup: {len(walls)} walls", "debug")

    used_walking_wall_hints = False

    # --- 4a1. Walking-mode primary wall source: per-frame wall hints --------
    # The merged walking cloud is still unreliable for wall extraction because
    # horizontal clutter can dominate the global RANSAC.  Per-frame wall hints
    # are extracted from already-gravity-aligned frames and are much more
    # stable.  Therefore in walking mode we merge them EARLY, before the wall
    # count / direction checks and before rectangle fitting.
    if skip_world_align and walking_wall_hints:
        log(
            f"[INFO] Walking recovery: merging {len(walking_wall_hints)} per-frame wall hint(s)",
            "info",
        )
        walls = _merge_wall_faces(walls + walking_wall_hints)
        used_walking_wall_hints = len(walls) > post_global_merge_wall_count
        log(f"[INFO] Walking recovery: walls after per-frame hints = {len(walls)}", "info")
    elif skip_world_align:
        log("[INFO] Walking recovery: no per-frame wall hints were available", "info")

    # --- 4a2. Walking-mode recovery for a missing second wall direction -----
    # Generic RANSAC may find multiple sub-patches of one wall axis and miss
    # the orthogonal axis.  When that happens, look at LOCAL point-normal
    # azimuths in the pre-filtered wall slice and explicitly mine the missing
    # direction.
    if skip_world_align and _walking_wall_slice is not None:
        if len(walls) >= 1 and _count_distinct_wall_dirs(walls) < 2:
            log("[INFO] Walking recovery: searching wall slice for a second wall direction", "info")
            if not _walking_wall_slice.has_normals():
                _walking_wall_slice.estimate_normals(
                    o3d.geometry.KDTreeSearchParamKNN(knn=30)
                )

            _nrms = np.asarray(_walking_wall_slice.normals)
            _az   = np.mod(np.arctan2(_nrms[:, 1], _nrms[:, 0]), np.pi)

            # Suppress the directions we already have, then look for the
            # strongest remaining azimuth cluster.
            _bins = 36
            _hist, _edges = np.histogram(_az, bins=_bins, range=(0.0, np.pi))
            _bin_centers = 0.5 * (_edges[:-1] + _edges[1:])
            _existing = [
                float(np.arctan2(get_plane_normal(w)[1], get_plane_normal(w)[0])) % np.pi
                for w in walls
            ]

            def _angdiff_pi(a, b):
                d = abs(a - b)
                return min(d, np.pi - d)

            for i, c in enumerate(_bin_centers):
                if any(_angdiff_pi(c, ea) < np.radians(20.0) for ea in _existing):
                    _hist[i] = 0

            _peak_idx = int(np.argmax(_hist)) if len(_hist) else -1
            _peak_cnt = int(_hist[_peak_idx]) if _peak_idx >= 0 else 0
            _min_peak = max(80, int(0.04 * len(_az)))

            if _peak_cnt >= _min_peak:
                _target_az = float(_bin_centers[_peak_idx])
                _mask = np.array([
                    _angdiff_pi(a, _target_az) <= np.radians(18.0) for a in _az
                ])
                _idx = np.where(_mask)[0]
                _cand = _walking_wall_slice.select_by_index(_idx.tolist())
                if len(_cand.points) >= CONFIG["min_plane_points"]:
                    _cand.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=30))
                    log(
                        f"[INFO] Walking recovery: extracting planes from secondary azimuth "
                        f"{np.degrees(_target_az):.0f}° ({len(_cand.points)} pts)",
                        "info",
                    )
                    _cand_planes = extract_planes(_cand)
                    _, _cand_walls = classify(_cand_planes)
                    _cand_walls = _merge_wall_faces(_cand_walls)

                    for _w in _cand_walls:
                        _waz = float(np.arctan2(get_plane_normal(_w)[1], get_plane_normal(_w)[0])) % np.pi
                        if all(_angdiff_pi(_waz, ea) > np.radians(20.0) for ea in _existing):
                            walls.append(_w)
                            _existing.append(_waz)

                    walls = _merge_wall_faces(walls)
                    log(f"[INFO] Walking recovery: walls after secondary-axis recovery = {len(walls)}", "info")

        # Final fallback: bypass plane RANSAC and derive wall faces directly
        # from point-normal azimuth clusters in the wall slice. This also
        # handles the fully-empty case where global RANSAC found 0 walls.
        if len(walls) < 2 or _count_distinct_wall_dirs(walls) < 2:
            _cluster_walls = _walls_from_normal_clusters(_walking_wall_slice)
            if _cluster_walls:
                walls = _merge_wall_faces(walls + _cluster_walls)
                log(
                    f"[INFO] Walking recovery: walls after point-normal recovery = {len(walls)}",
                    "info",
                )

    if not floors:
        raise RuntimeError(
            f"No floor detected among {len(planes)} plane(s) "
            f"({len(walls)} wall(s)). "
            "Point the sensor at the floor or add floor coverage."
        )

    if len(walls) < 2:
        raise RuntimeError(
            f"Only {len(walls)} wall(s) detected (need ≥ 2). "
            f"Total planes found: {len(planes)} "
            f"({len(floors)} floor-like, {len(walls)} wall-like).\n"
            "Tips:\n"
            "  • Walk close to each wall so the opposite wall is far away\n"
            "  • Hold sensor at 1.4–1.5 m height, upright and level\n"
            "  • Walk slowly — pause 2–3 s at each corner\n"
            "  • Increase capture time: --seconds 60\n"
            "  • Run with --log debug to see plane-by-plane detail"
        )

    # --- 4b. Check that walls span at least 2 distinct directions ----------
    # A rectangular room needs wall normals in ≥ 2 different azimuths.
    # If all detected walls are parallel/antiparallel (single direction) the
    # polygon degenerates.  Report clearly rather than returning 0.01 m² area.
    if len(walls) >= 2:
        if _count_distinct_wall_dirs(walls) < 2:
            raise RuntimeError(
                f"All {len(walls)} wall(s) point in the same direction — "
                "the room polygon cannot be computed. "
                "Ensure walls on at least 2 different sides are visible in the scan."
            )

    # --- 5. Best floor = LOWEST substantial horizontal plane ---
    # Walking scans can contain several horizontal clutter planes above the
    # true floor. Choosing by area alone can select one of those and collapse
    # the room-height estimate. Prefer the lowest plane among the substantial
    # horizontal candidates; break ties by larger area.
    max_floor_area = max(get_plane_area(f) for f in floors)
    floor_candidates = [
        f for f in floors
        if get_plane_area(f) >= 0.35 * max_floor_area
    ]
    floor = min(
        floor_candidates,
        key=lambda x: (get_plane_floor_z(x), -get_plane_area(x))
    )
    cloud_floor_z = float(
        np.percentile(np.asarray(p.points)[:, 2], CONFIG.get("floor_percentile", 2))
    )
    floor_z = min(get_plane_floor_z(floor), cloud_floor_z)

    # --- 5b. Ceiling detection — highest horizontal plane ≥1 m above floor ---
    cloud_z_hi = float(np.percentile(np.asarray(p.points)[:, 2], 99.5))
    cloud_z_hi_strict = float(np.percentile(np.asarray(p.points)[:, 2], 99.7))
    ceiling_candidates = [
        f for f in floors
        if f is not floor and float(get_plane_centroid(f)[2]) > floor_z + 1.0
    ]
    if skip_world_align and walking_ceiling_hints:
        ceiling_candidates.extend([
            f for f in walking_ceiling_hints
            if float(get_plane_centroid(f)[2]) > floor_z + 1.0
        ])
    ceiling = (
        max(ceiling_candidates, key=lambda f: get_plane_area(f))
        if ceiling_candidates else None
    )
    ceiling_is_synthetic = bool(isinstance(ceiling, dict) and ceiling.get("synthetic", False))
    if ceiling is not None:
        ceil_z = float(get_plane_centroid(ceiling)[2])
        # Reject a "ceiling" that sits well below the observed top envelope.
        # This catches false positives such as shelf tops and lintels.
        top_ref = cloud_z_hi
        reject_margin = 0.35
        if skip_world_align:
            top_ref = max(top_ref, cloud_z_hi_strict)
            if walking_ceiling_z_hint is not None:
                top_ref = max(top_ref, float(walking_ceiling_z_hint))
            reject_margin = 0.20
        if top_ref - ceil_z > reject_margin:
            log(
                f"[INFO] Rejecting false ceiling at Z={ceil_z:.2f} m; "
                f"top envelope is {top_ref:.2f} m",
                "info",
            )
            ceiling = None
        elif skip_world_align and walking_ceiling_z_hint is not None and ceil_z < float(walking_ceiling_z_hint) - 0.10:
            log(
                f"[INFO] Rejecting false ceiling at Z={ceil_z:.2f} m; "
                f"walking ceiling hint is {float(walking_ceiling_z_hint):.2f} m",
                "info",
            )
            ceiling = None
            ceiling_is_synthetic = False
        elif ceiling_is_synthetic:
            log(
                f"[INFO] Ceiling fallback at Z={ceil_z:.2f} m will be treated as a height estimate, "
                "not as an exact detected ceiling plane",
                "info",
            )
    if ceiling is not None:
        log(
            f"[DEBUG] Ceiling plane detected at Z={get_plane_centroid(ceiling)[2]:.2f} m "
            f"(area={get_plane_area(ceiling):.2f} m²)",
            "debug",
        )

    # --- 6. Room corners via 2-D XY rectangle fitting ---
    # This is the primary and only corner-finding method.  It is robust to
    # RANSAC sub-patches of the same wall (which would produce a thin phantom
    # polygon under three-plane intersection) and to partially occluded walls
    # (synthesises missing face from cloud-centroid mirror).
    _pts = np.asarray(p.points)
    cloud_bounds = (_pts.min(axis=0), _pts.max(axis=0))
    used_generic_rect_fit = False

    if skip_world_align:
        def _poly_area_xy(poly):
            if poly is None:
                return None
            xy = np.asarray(poly)[:, :2]
            if xy.shape[0] < 3:
                return None
            x = xy[:, 0]
            y = xy[:, 1]
            return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

        def _fit_score(result):
            _poly, _synth, _direct = result
            return (
                -1 if _poly is None else int(_direct),
                -1 if _poly is None else int(not _synth),
            )

        walking_fit = _walking_rect_from_wall_extents(walls, floor, _pts)
        fallback_fit = (None, False, 0)
        if walking_fit[0] is None or walking_fit[2] < 4 or walking_fit[1]:
            fallback_fit = _rectangular_fallback(
                walls,
                floor,
                _pts,
                allow_cloud_synthesis=not (post_global_merge_wall_count == 0 and used_walking_wall_hints),
            )

        prefer_fallback = _fit_score(fallback_fit) > _fit_score(walking_fit)
        if prefer_fallback and walking_fit[0] is not None and fallback_fit[0] is not None:
            walking_area = _poly_area_xy(walking_fit[0])
            fallback_area = _poly_area_xy(fallback_fit[0])
            if (
                walking_area is not None
                and fallback_area is not None
                and walking_area > 0.0
                and fallback_area / walking_area < 0.75
            ):
                log(
                    f"[INFO] Walking fit selection: keeping walking extent fit because generic 2-D fit "
                    f"collapsed area from {walking_area:.2f} m² to {fallback_area:.2f} m²",
                    "info",
                )
                prefer_fallback = False

        if prefer_fallback:
            log(
                f"[INFO] Walking fit selection: preferred generic 2-D fit "
                f"({fallback_fit[2]}/4 direct faces) over walking extent fit "
                f"({walking_fit[2]}/4 direct faces)",
                "info",
            )
            poly, used_rect_fallback, direct_wall_faces = fallback_fit
            used_generic_rect_fit = True
        else:
            poly, used_rect_fallback, direct_wall_faces = walking_fit
    else:
        poly, used_rect_fallback, direct_wall_faces = _rectangular_fallback(walls, floor, _pts)
        used_generic_rect_fit = poly is not None

    if poly is None:
        # _rectangular_fallback returns None when wall directions are not
        # clustered into exactly 2 perpendicular groups — report clearly.
        wall_dirs = []
        for w in walls:
            n = get_plane_normal(w)[:2]
            nn = np.linalg.norm(n)
            if nn > 1e-6:
                az = float(np.degrees(np.arctan2(n[1], n[0]))) % 360
                wall_dirs.append(f"{az:.0f}°")
        dirs_str = ", ".join(wall_dirs) if wall_dirs else "none"
        raise RuntimeError(
            f"Could not fit a rectangular polygon from {len(walls)} wall(s) "
            f"in directions: {dirs_str}.\n"
            "A rectangular room needs walls clustered into exactly 2 "
            "perpendicular direction groups.\n"
            "Try:\n"
            "  • Walk the full room perimeter, pausing at each corner\n"
            "  • Increase capture time: --seconds 60\n"
            "  • Run with --log debug to see per-plane detail"
        )

    log(f"[DEBUG] 2-D rectangle fit: {len(poly)} corners "
        f"({'face synthesised' if used_rect_fallback else 'all walls measured'})",
        "debug")

    # Floor area: prefer the intersection polygon (structurally derived) over
    # the raw RANSAC inlier convex hull — more accurate for rectangular rooms.
    try:
        from shapely.geometry import Polygon as _Poly
        _pshape = _Poly(poly[:, :2])
        floor_area = float(_pshape.area) if _pshape.is_valid and _pshape.area > 0.1 else float(get_plane_area(floor))
    except Exception:
        floor_area = float(get_plane_area(floor))

    perimeter   = poly_perimeter(poly)

    # Height: ceiling plane (exact) → wall/cloud-top estimate (fallback).
    use_synthetic_ceiling_height = False
    height_source = "wall_top_estimate"
    if ceiling is not None and not ceiling_is_synthetic:
        room_height = float(get_plane_centroid(ceiling)[2]) - floor_z
        room_height = max(room_height, 0.1)
        height_source = "ceiling_plane"
        log(f"[DEBUG] Height from ceiling plane: {room_height:.3f} m", "debug")
    elif ceiling is not None and ceiling_is_synthetic:
        _pz = np.asarray(p.points)[:, 2]
        _cloud_z_robust = float(np.percentile(_pz, 99.7))
        _cloud_z_sparse = min(float(_pz.max()), _cloud_z_robust + 0.30)
        _cloud_z_for_height = max(_cloud_z_robust, _cloud_z_sparse)
        if skip_world_align and walking_ceiling_z_hint is not None:
            _cloud_z_for_height = max(_cloud_z_for_height, float(walking_ceiling_z_hint))
        baseline_height = estimate_room_height(walls, floor,
                                               cloud_z_max=_cloud_z_for_height,
                                               prefer_cloud_top=skip_world_align,
                                               floor_z_hint=cloud_floor_z)
        synthetic_height = max(float(get_plane_centroid(ceiling)[2]) - floor_z, 0.1)
        if baseline_height is None or synthetic_height > baseline_height + 0.05:
            room_height = synthetic_height
            use_synthetic_ceiling_height = True
            height_source = "ceiling_band_estimate"
            log(
                f"[DEBUG] Height from synthetic ceiling band: {room_height:.3f} m "
                f"(baseline wall/cloud estimate={baseline_height:.3f} m)" if baseline_height is not None else
                f"[DEBUG] Height from synthetic ceiling band: {room_height:.3f} m",
                "debug",
            )
        else:
            room_height = baseline_height
            height_source = "wall_top_estimate"
            log(
                f"[DEBUG] Synthetic ceiling band ({synthetic_height:.3f} m) did not improve "
                f"the wall/cloud estimate ({baseline_height:.3f} m); keeping baseline",
                "debug",
            )
    else:
        _pz = np.asarray(p.points)[:, 2]
        _cloud_z_robust = float(np.percentile(_pz, 99.7))
        _cloud_z_sparse = min(float(_pz.max()), _cloud_z_robust + 0.30)
        _cloud_z_for_height = max(_cloud_z_robust, _cloud_z_sparse)
        if skip_world_align and walking_ceiling_z_hint is not None:
            _cloud_z_for_height = max(_cloud_z_for_height, float(walking_ceiling_z_hint))
        room_height = estimate_room_height(walls, floor,
                                            cloud_z_max=_cloud_z_for_height,
                                            prefer_cloud_top=skip_world_align,
                                            floor_z_hint=cloud_floor_z)
        height_source = "wall_top_estimate"
        log(f"[DEBUG] Height from mean wall-top Z (cloud_z_max={_cloud_z_for_height:.2f} m): "
            f"{room_height:.3f} m (no ceiling plane detected)", "debug")
        if room_height is not None and room_height < 2.0:
            log(
                "[WARN] Estimated height ({:.2f} m) looks low — ceiling may be "
                "outside the sensor's vertical FOV.  Walk with the sensor at "
                "1.4–1.5 m for best height coverage."
                .format(room_height),
                "info",
            )

    confidence_issues = []
    if skip_world_align:
        support_metrics = walking_support_metrics if isinstance(walking_support_metrics, dict) else {}
        recovered_wall_hints_raw = len(walking_wall_hints) if walking_wall_hints else 0
        recovered_wall_obs_raw = len(walls)
        recovered_wall_hints = max(recovered_wall_hints_raw, int(support_metrics.get("hint_count", 0)))
        recovered_wall_obs = max(recovered_wall_obs_raw, int(support_metrics.get("wall_observations", 0)))
        if recovered_wall_hints > recovered_wall_hints_raw or recovered_wall_obs > recovered_wall_obs_raw:
            log(
                f"[INFO] Walking confidence support: using stronger pre-registration evidence "
                f"({recovered_wall_hints} hints, {recovered_wall_obs} observations) "
                f"instead of transformed-hint counts ({recovered_wall_hints_raw}, {recovered_wall_obs_raw})",
                "info",
            )
        fit_info = LAST_WALKING_FIT_INFO if isinstance(LAST_WALKING_FIT_INFO, dict) else {}
        # For guided live mode the per-frame scan angle is narrow, so fewer
        # frame-level wall hints are produced per lap compared to a full bag.
        # The scene geometry checks (edge fill ratios, continuity, floor support)
        # are now the primary quality signal; keep the hint/obs thresholds
        # proportional to what a single guided lap can realistically deliver.
        if allow_guided_live_borderline:
            sparse_hints = recovered_wall_hints < 3
            sparse_obs = recovered_wall_obs < 2
        else:
            sparse_hints = recovered_wall_hints < 12
            sparse_obs = recovered_wall_obs < 10
        moderate_hints = recovered_wall_hints < 15
        moderate_obs = recovered_wall_obs < 12
        dropped_frames = 0
        tilt_median = 0.0
        tilt_p90 = 0.0
        if sparse_hints:
            confidence_issues.append(
                f"only {recovered_wall_hints} merged wall hint(s) were available; walking geometry support is too sparse"
            )
        if sparse_obs:
            confidence_issues.append(
                f"only {recovered_wall_obs} wall observations remained after merging; repeatability is too weak"
            )
        if post_global_merge_wall_count == 0 and walking_quality is not None:
            dropped_frames = int(walking_quality.get("tilt_dropped_frames", 0))
            tilt_median = float(walking_quality.get("tilt_median_deg", 0.0))
            tilt_p90 = float(walking_quality.get("tilt_p90_deg", 0.0))
            if dropped_frames >= 2 and tilt_p90 >= 7.5:
                confidence_issues.append(
                    f"walking-frame tilt was unstable (median={tilt_median:.1f}°, p90={tilt_p90:.1f}°, dropped={dropped_frames})"
                )
        max_direct_tail = float(fit_info.get("max_direct_tail", 0.0))
        borderline_usable_walking_fit = (
            post_global_merge_wall_count == 0
            and used_walking_wall_hints
            and not used_generic_rect_fit
            and direct_wall_faces >= 3
            and recovered_wall_hints >= 13
            and recovered_wall_obs >= 11
            and dropped_frames <= 1
            and tilt_p90 <= 6.5
        )
        # For a hand-held walking sensor, tilt_p90 of 7-12° is normal: the
        # world_align step corrects the tilt per-frame before ICP.  The
        # tilt value recorded is the correction applied, not residual error,
        # so values up to ~15° are fine.  The separate gate (dropped_frames
        # >= 2 AND tilt_p90 >= 7.5°) already catches genuinely unstable tilt.
        guided_live_borderline_fit = (
            allow_guided_live_borderline
            and post_global_merge_wall_count == 0
            and used_walking_wall_hints
            and direct_wall_faces >= 2
            and recovered_wall_hints >= 3
            and recovered_wall_obs >= 2
            and dropped_frames <= 1
            and tilt_p90 <= 15.0
            and max_direct_tail <= 0.40
        )
        if post_global_merge_wall_count == 0 and used_walking_wall_hints and direct_wall_faces < 4 and not borderline_usable_walking_fit and not guided_live_borderline_fit:
            confidence_issues.append(
                "global wall extraction found no walls and walking recovery could not directly support all four room faces"
            )
        if used_rect_fallback and direct_wall_faces < 4 and not borderline_usable_walking_fit and not guided_live_borderline_fit:
            confidence_issues.append(
                f"footprint needed a synthesised wall face ({direct_wall_faces}/4 direct faces)"
            )
        if post_global_merge_wall_count == 0 and direct_wall_faces < 4 and moderate_hints and moderate_obs and not borderline_usable_walking_fit and not guided_live_borderline_fit:
            confidence_issues.append(
                f"walking recovery remained too sparse after merging ({recovered_wall_hints} hints, {recovered_wall_obs} wall observations)"
            )
        if (
            direct_wall_faces >= 4
            and bool(fit_info.get("ignored_outer_groups", False))
            and max_direct_tail > 0.35
        ):
            confidence_issues.append(
                f"direct wall support remained too smeared for a reliable footprint (max support tail {max_direct_tail:.2f} m)"
            )
        # Store diagnostic values in LAST_WALKING_FIT_INFO so reconstruct caller
        # can write them into the report after the report dict is created.
        if skip_world_align and isinstance(LAST_WALKING_FIT_INFO, dict):
            LAST_WALKING_FIT_INFO["_tilt_p90"] = round(float(tilt_p90), 2)
            LAST_WALKING_FIT_INFO["_dropped_frames"] = int(dropped_frames)
            LAST_WALKING_FIT_INFO["_recovered_wall_hints"] = int(recovered_wall_hints)
            LAST_WALKING_FIT_INFO["_recovered_wall_obs"] = int(recovered_wall_obs)
            LAST_WALKING_FIT_INFO["_guided_live_borderline_fit"] = bool(guided_live_borderline_fit)
        if borderline_usable_walking_fit:
            log(
                f"[INFO] Walking confidence gate: accepting borderline fit with {direct_wall_faces}/4 direct faces "
                f"because merged support remained usable ({recovered_wall_hints} hints, {recovered_wall_obs} observations, tilt p90={tilt_p90:.1f}°)",
                "info",
            )
        elif guided_live_borderline_fit:
            log(
                f"[INFO] Guided live confidence gate: accepting strong recovered fit with {direct_wall_faces}/4 direct faces "
                f"despite rectangular fallback because merged support remained strong ({recovered_wall_hints} hints, {recovered_wall_obs} observations, tilt p90={tilt_p90:.1f}°, direct tail={max_direct_tail:.2f} m)",
                "info",
            )
        if ceiling is not None and not ceiling_is_synthetic and walking_ceiling_z_hint is not None:
            ceil_z = float(get_plane_centroid(ceiling)[2])
            if ceil_z < float(walking_ceiling_z_hint) - 0.10:
                confidence_issues.append(
                    f"ceiling plane {ceil_z:.2f} m conflicts with ceiling hint {float(walking_ceiling_z_hint):.2f} m"
                )
        if confidence_issues and not return_low_confidence_report:
            raise RuntimeError(
                "Low-confidence walking measurement rejected:\n  • "
                + "\n  • ".join(confidence_issues)
                + "\nPlease repeat the scan or record one fixed bag for offline tuning."
            )

    measurement_warnings = []
    if ceiling is None:
        measurement_warnings.append(
            "No ceiling plane was detected; room height comes from wall-top/cloud-top estimation."
        )
    elif ceiling_is_synthetic and use_synthetic_ceiling_height:
        measurement_warnings.append(
            "Ceiling height came from a synthetic top-band fallback, not a directly extracted ceiling plane."
        )
    elif ceiling_is_synthetic:
        measurement_warnings.append(
            "A synthetic ceiling-band fallback was found, but wall/cloud-top height estimation remained more reliable."
        )
    if used_walking_wall_hints:
        measurement_warnings.append(
            "Walking wall recovery merged per-frame wall hints because global wall extraction was incomplete."
        )
    if skip_world_align and direct_wall_faces < 4:
        measurement_warnings.append(
            f"Footprint used {4 - direct_wall_faces} synthesised wall face(s); repeatability may be lower than the reference bag."
        )
    if used_rect_fallback:
        measurement_warnings.append(
            "Footprint fitting synthesised at least one wall face from extents instead of direct wall geometry."
        )

    walking_ceiling_hint_count = len(walking_ceiling_hints) if walking_ceiling_hints else 0
    perimeter_ceiling_support_strong = False
    if skip_world_align and ceiling is not None and not ceiling_is_synthetic:
        ceil_z = float(get_plane_centroid(ceiling)[2])
        perimeter_ceiling_support_strong = (
            cloud_z_hi_strict >= ceil_z - 0.12
            and (
                walking_ceiling_hint_count >= 3
                or (
                    walking_ceiling_hint_count >= 2
                    and walking_ceiling_z_hint is not None
                    and abs(ceil_z - float(walking_ceiling_z_hint)) <= 0.10
                )
            )
        )

    if ceiling is not None and not ceiling_is_synthetic:
        if skip_world_align and not perimeter_ceiling_support_strong:
            height_confidence = "medium"
            measurement_warnings.append(
                "A ceiling plane was detected, but direct ceiling support across the perimeter lap remained limited; treat room height as medium-confidence."
            )
        else:
            height_confidence = "high"
    elif ceiling is not None and ceiling_is_synthetic and use_synthetic_ceiling_height:
        height_confidence = "medium"
    elif (
        used_rect_fallback or
        post_global_merge_wall_count < 3 or
        initial_wall_count == 0 or
        len(walls) < 12 or
        (walking_wall_hints is not None and len(walking_wall_hints) < 15)
    ):
        height_confidence = "low"
    else:
        height_confidence = "medium"

    if (
        skip_world_align
        and repeat_walk_height_hint is not None
        and height_source == "wall_top_estimate"
        and room_height is not None
        and height_confidence == "low"
        and float(repeat_walk_height_hint) > room_height + 0.05
    ):
        log(
            f"[INFO] Height override: using repeat-walk hint {float(repeat_walk_height_hint):.2f} m "
            f"instead of low-confidence single-bag estimate {room_height:.2f} m",
            "info",
        )
        room_height = float(repeat_walk_height_hint)
        height_source = "repeat_walk_height_hint"
        height_confidence = "medium"

    if height_source == "repeat_walk_height_hint":
        measurement_warnings.append(
            "Room height was lifted using the strongest ceiling-span evidence from sibling repeated walking bags of the same room."
        )

    volume = float(floor_area * room_height) if room_height is not None else None

    if used_walking_wall_hints and post_global_merge_wall_count == 0:
        wall_source = "frame_hints_recovery"
    elif used_walking_wall_hints:
        wall_source = "global_plus_frame_hints"
    else:
        wall_source = "global_planes"

    log(f"[DEBUG] floor_area={floor_area:.2f} m²  perimeter={perimeter:.2f} m  "
        f"height={room_height:.2f} m  volume={volume:.2f} m³", "debug")

    report = dict(
        floor_area_m2       = round(floor_area, 3),
        floor_perimeter_m   = round(perimeter, 3),
        room_height_m       = round(room_height, 3) if room_height is not None else None,
        volume_m3           = round(volume, 3)       if volume       is not None else None,
        walls               = len(poly),
        direct_wall_faces   = int(direct_wall_faces),
        wall_observations   = len(walls),
        corners             = len(poly),
        ceiling_detected    = ceiling is not None and not ceiling_is_synthetic,
        rectangular_fallback= used_generic_rect_fit,
        wall_source         = wall_source,
        height_source       = height_source,
        height_confidence   = height_confidence,
        measurement_warnings= measurement_warnings,
        floor_z             = round(float(max(get_plane_floor_z(floor), cloud_floor_z)), 3),
        polygon_vertices_xy = [[round(float(x), 3), round(float(y), 3)] for x, y in poly[:, :2]],
        walking_fit_max_direct_tail = round(float((LAST_WALKING_FIT_INFO or {}).get("max_direct_tail", 0.0) or 0.0), 3),
        walking_fit_ignored_outer_groups = bool((LAST_WALKING_FIT_INFO or {}).get("ignored_outer_groups", False)),
    )
    if skip_world_align:
        report["walking_ceiling_hint_count"] = int(walking_ceiling_hint_count)
        if walking_ceiling_z_hint is not None:
            report["walking_ceiling_z_hint"] = round(float(walking_ceiling_z_hint), 3)
        report["perimeter_ceiling_support"] = (
            "strong" if perimeter_ceiling_support_strong else
            "limited" if (ceiling is not None or walking_ceiling_hint_count > 0 or walking_ceiling_z_hint is not None) else
            "unavailable"
        )
        # Write diagnostics that were staged in LAST_WALKING_FIT_INFO.
        _fit_diag = LAST_WALKING_FIT_INFO if isinstance(LAST_WALKING_FIT_INFO, dict) else {}
        for _k in ("_tilt_p90", "_dropped_frames", "_recovered_wall_hints",
                   "_recovered_wall_obs", "_guided_live_borderline_fit"):
            if _k in _fit_diag:
                report[_k] = _fit_diag[_k]
    if confidence_issues:
        report["confidence_issues"] = confidence_issues

    export_tag = time.strftime("%Y%m%d_%H%M%S")
    scene_data = None
    try:
        scene_input = semantic_cloud if semantic_cloud is not None and len(semantic_cloud.points) else p
        scene_data = _analyze_scene_exports(
            scene_input,
            poly,
            max(get_plane_floor_z(floor), cloud_floor_z),
            room_height=room_height,
            semantic_frames=semantic_frames,
        )
        if scene_data:
            report["scene_furniture_candidates"] = len(scene_data.get("furniture", []))
            report["scene_opening_candidates"] = len(scene_data.get("openings", []))
            report["scene_input_points"] = int(len(scene_input.points))
            diagnostics = dict(scene_data.get("diagnostics") or {})
            report["scene_floor_points"] = int(diagnostics.get("floor_points", 0) or 0)
            report["scene_wall_band_points"] = int(diagnostics.get("wall_band_points", 0) or 0)
            report["scene_object_candidate_points"] = int(diagnostics.get("object_candidate_points", 0) or 0)
            scene_floor_points = int(diagnostics.get("floor_points", 0) or 0)
            scene_input_points = int(len(scene_input.points))
            scene_floor_support_ratio = (
                float(scene_floor_points) / float(scene_input_points)
                if scene_input_points > 0 else 0.0
            )
            edges = list(scene_data.get("edges") or [])
            if edges:
                max_edge_opening_evidence = max(
                    float(edge.get("opening_evidence_score", 0.0) or 0.0) for edge in edges
                )
                max_evidence_edge = max(
                    edges,
                    key=lambda edge: (
                        float(edge.get("opening_evidence_score", 0.0) or 0.0),
                        float(edge.get("continuity", 0.0) or 0.0),
                    ),
                )
                report["scene_max_opening_evidence_score"] = round(
                    max(
                        float(diagnostics.get("max_opening_evidence_score", 0.0) or 0.0),
                        max_edge_opening_evidence,
                    ),
                    3,
                )
                report["scene_max_evidence_edge_index"] = int(max_evidence_edge.get("edge_index", -1))
                report["scene_max_evidence_edge_continuity"] = round(
                    float(max_evidence_edge.get("continuity", 0.0) or 0.0),
                    3,
                )
                report["scene_max_evidence_edge_low_fill_ratio"] = round(
                    float(max_evidence_edge.get("low_fill_ratio", 0.0) or 0.0),
                    3,
                )
                report["scene_max_evidence_edge_top_fill_ratio"] = round(
                    float(max_evidence_edge.get("top_fill_ratio", 0.0) or 0.0),
                    3,
                )
                report["scene_max_evidence_edge_gap_width"] = round(
                    float(max_evidence_edge.get("best_gap_width", 0.0) or 0.0),
                    3,
                )
                report["scene_max_evidence_edge_status"] = max_evidence_edge.get("best_gap_status")
                report["scene_max_evidence_edge_rejections"] = list(
                    max_evidence_edge.get("best_gap_rejections", [])
                )
                report["scene_min_edge_continuity"] = round(
                    min(float(edge.get("continuity", 0.0) or 0.0) for edge in edges),
                    3,
                )
                report["scene_min_edge_low_fill_ratio"] = round(
                    min(float(edge.get("low_fill_ratio", 0.0) or 0.0) for edge in edges),
                    3,
                )
                report["scene_min_edge_top_fill_ratio"] = round(
                    min(float(edge.get("top_fill_ratio", 0.0) or 0.0) for edge in edges),
                    3,
                )
                weakest_edge = min(
                    edges,
                    key=lambda edge: (
                        min(
                            float(edge.get("continuity", 1.0) or 1.0),
                            float(edge.get("low_fill_ratio", 1.0) or 1.0),
                            float(edge.get("top_fill_ratio", 1.0) or 1.0),
                        ),
                        float(edge.get("top_fill_ratio", 1.0) or 1.0),
                        float(edge.get("low_fill_ratio", 1.0) or 1.0),
                        float(edge.get("continuity", 1.0) or 1.0),
                        -float(edge.get("opening_evidence_score", 0.0) or 0.0),
                    ),
                )
                report["scene_weak_edge_index"] = int(weakest_edge.get("edge_index", -1))
                report["scene_weak_edge_continuity"] = round(
                    float(weakest_edge.get("continuity", 0.0) or 0.0),
                    3,
                )
                report["scene_weak_edge_low_fill_ratio"] = round(
                    float(weakest_edge.get("low_fill_ratio", 0.0) or 0.0),
                    3,
                )
                report["scene_weak_edge_top_fill_ratio"] = round(
                    float(weakest_edge.get("top_fill_ratio", 0.0) or 0.0),
                    3,
                )
                report["scene_weak_edge_opening_evidence_score"] = round(
                    float(weakest_edge.get("opening_evidence_score", 0.0) or 0.0),
                    3,
                )

                # Live/walking fits can occasionally produce a coherent but oversized
                # rectangle even with 4/4 direct faces. If no opening was promoted,
                # one room edge should not remain both badly fragmented and weakly
                # filled near the floor.
                if (
                    skip_world_align
                    and direct_wall_faces >= 4
                ):
                    if (
                        allow_guided_live_borderline
                        and (scene_floor_support_ratio < 0.004 or scene_floor_points < 800)
                    ):
                        confidence_issues.append(
                            "floor support remained too weak for a reliable live footprint "
                            f"({scene_floor_points} floor points, ratio {scene_floor_support_ratio:.4f})"
                        )

                    min_edge_continuity = min(
                        float(edge.get("continuity", 0.0) or 0.0) for edge in edges
                    )
                    min_edge_low_fill = min(
                        float(edge.get("low_fill_ratio", 0.0) or 0.0) for edge in edges
                    )
                    # Compute min_edge_top_fill here so it is available for all
                    # subsequent checks (it was previously assigned ~15 lines later,
                    # causing an UnboundLocalError that silently aborted this entire
                    # confidence-check block via the outer except clause).
                    min_edge_top_fill = min(
                        float(edge.get("top_fill_ratio", 0.0) or 0.0) for edge in edges
                    )
                    if min_edge_continuity < 0.75 and min_edge_low_fill < 0.40:
                        confidence_issues.append(
                            "direct wall geometry remained too inconsistent for a reliable footprint "
                            f"(min continuity {min_edge_continuity:.2f}, min low fill {min_edge_low_fill:.2f})"
                        )

                    if (
                        allow_guided_live_borderline
                        and min_edge_low_fill < 0.20
                        and min_edge_top_fill < 0.10
                    ):
                        confidence_issues.append(
                            "one wall edge remained severely under-supported for a reliable footprint "
                            f"(min low fill {min_edge_low_fill:.2f}, min top fill {min_edge_top_fill:.2f})"
                        )

                    max_opening_evidence = max(
                        float(diagnostics.get("max_opening_evidence_score", 0.0) or 0.0),
                        max_edge_opening_evidence,
                    )
                    opening_count = int(len(scene_data.get("openings") or []))
                    ignore_narrow_gap_ambiguity = (
                        allow_guided_live_borderline
                        and opening_count == 0
                        and str(max_evidence_edge.get("best_gap_status") or "") == "rejected"
                        and "too_narrow" in set(max_evidence_edge.get("best_gap_rejections", []) or [])
                        and set(max_evidence_edge.get("best_gap_rejections", []) or [])
                        <= {"too_narrow", "weak_left_jamb", "weak_right_jamb", "weak_jamb"}
                        and float(max_evidence_edge.get("best_gap_width", 0.0) or 0.0) <= 0.35
                        and float(max_evidence_edge.get("continuity", 0.0) or 0.0) >= 0.9
                    )
                    if (
                        allow_guided_live_borderline
                        and max_opening_evidence >= 0.35
                        and (min_edge_low_fill < 0.45 or min_edge_top_fill < 0.25)
                        and not ignore_narrow_gap_ambiguity
                    ):
                        confidence_issues.append(
                            "direct wall support is still too ambiguous near one edge for a reliable footprint "
                            f"(opening evidence {max_opening_evidence:.2f}, min low fill {min_edge_low_fill:.2f}, "
                            f"min top fill {min_edge_top_fill:.2f})"
                        )

                    if (
                        allow_guided_live_borderline
                        and opening_count > 0
                        and min_edge_low_fill < 0.20
                        and min_edge_top_fill < 0.10
                    ):
                        confidence_issues.append(
                            "promoted openings still left one wall edge severely under-supported for a reliable footprint "
                            f"(openings {opening_count}, min low fill {min_edge_low_fill:.2f}, min top fill {min_edge_top_fill:.2f})"
                        )

                    if (
                        allow_guided_live_borderline
                        and used_generic_rect_fit
                        and (min_edge_low_fill < 0.45 or min_edge_top_fill < 0.25)
                    ):
                        confidence_issues.append(
                            "rectangular fallback stayed too interior for a reliable footprint "
                            f"(min low fill {min_edge_low_fill:.2f}, min top fill {min_edge_top_fill:.2f})"
                        )

                    max_direct_tail = float((LAST_WALKING_FIT_INFO or {}).get("max_direct_tail", 0.0) or 0.0)
                    if allow_guided_live_borderline and direct_wall_faces >= 4 and max_direct_tail > 0.28:
                        confidence_issues.append(
                            "walking fit still smeared too far beyond the direct wall support for a reliable footprint "
                            f"(max direct tail {max_direct_tail:.2f})"
                        )
    except Exception as exc:
        log(f"[WARN] Scene export analysis failed: {exc}", "info")
        measurement_warnings.append(
            "Scene export enrichment failed; core room measurements were still generated."
        )

    report["footprint_confidence"] = _estimate_footprint_confidence(report)

    if not suppress_exports:
        export_primary_scene_cloud(scene_data, p, export_tag)
        export_dxf(
            poly,
            export_tag,
            floor_area=floor_area,
            perimeter=perimeter,
            room_height=room_height,
            scene_data=scene_data,
        )
        export_topdown_preview(scene_data, export_tag)
        export_scene_debug(scene_data, export_tag)
        report["_export_tag"] = export_tag

    if confidence_issues:
        report["confidence_issues"] = confidence_issues
        if not return_low_confidence_report:
            raise RuntimeError(
                "Low-confidence walking measurement rejected:\n  • "
                + "\n  • ".join(confidence_issues)
                + "\nPlease repeat the scan or record one fixed bag for offline tuning."
            )
    elif "confidence_issues" in report:
        del report["confidence_issues"]

    return report


# =========================================================
# MAIN
# =========================================================

def main():

    ap=argparse.ArgumentParser()
    ap.add_argument("--input")
    ap.add_argument("--multi",nargs="*")
    ap.add_argument("--output", default="./out",
                    help="Directory for generated files (default: ./out)")
    ap.add_argument("--manual",action="store_true")
    ap.add_argument("--bag-mode", choices=["auto", "static", "walking", "height-static"], default="auto")
    ap.add_argument("--height-bag",
                    help="Optional second input used only for room height estimation")
    ap.add_argument("--height-bag-mode", choices=["auto", "static", "walking", "height-static"], default="height-static",
                    help="Bag mode for --height-bag (default: height-static)")
    ap.add_argument("--bag-frame-interval", type=float, default=1.0,
                    help="Seconds between sampled PointCloud2 frames in ROS bag walking mode")
    ap.add_argument("--footprint-area", type=float,
                    help="Optional footprint area to combine with --bag-mode height-static for a volume estimate")
    ap.add_argument("--host", default="192.168.1.20")
    ap.add_argument("--port",type=int,default=7502)
    ap.add_argument("--seconds", type=int, default=45,
                    help="Live guided perimeter duration in seconds (default: 45)")
    ap.add_argument("--live-frame-interval", type=float, default=1.0,
                    help="Seconds between kept frames during live guided perimeter capture")
    ap.add_argument("--log", default="info")
    args=ap.parse_args()

    if args.manual:
        print(MANUAL)
        return

    if not args.input:
        print("Specify --input or --manual")
        return

    CONFIG["log_level"]=args.log
    set_output_dir(args.output)

    t0=time.time()
    try:
        if _is_rosbag_path(args.input) and args.bag_mode == "height-static":
            report = analyze_static_height_bag(args.input, footprint_area=args.footprint_area)
            report["scan_time_sec"] = round(time.time() - t0, 2)
            save_report_json(report)
            print(json.dumps(report, indent=2))
            return

        if args.height_bag and args.input == "live":
            raise RuntimeError(
                "Using --height-bag with --input live is not supported yet. "
                "First complete the two-bag bag-input workflow, then we can extend live capture to collect both inputs."
            )

        if args.input == "live":
            report = analyze_live_guided(
                host=args.host,
                port=args.port,
                seconds=args.seconds,
                frame_interval_sec=args.live_frame_interval,
            )
        else:
            report = analyze_input_path(
                args.input,
                multi=args.multi,
                bag_mode=args.bag_mode,
                bag_frame_interval=args.bag_frame_interval,
                host=args.host,
                port=args.port,
                seconds=args.seconds,
            )

        if args.height_bag:
            height_mode = args.height_bag_mode
            if height_mode == "auto":
                height_mode = "height-static"

            if height_mode == "height-static":
                height_report = analyze_static_height_bag(
                    args.height_bag,
                    footprint_area=report.get("floor_area_m2"),
                )
            else:
                height_report = analyze_input_path(
                    args.height_bag,
                    bag_mode=height_mode,
                    bag_frame_interval=args.bag_frame_interval,
                    host=args.host,
                    port=args.port,
                    seconds=args.seconds,
                )

            report = _merge_footprint_and_height_reports(
                report,
                height_report,
                args.input,
                args.height_bag,
                height_mode,
            )

        report["scan_time_sec"] = round(time.time() - t0, 2)
        save_report_json(report)
        print(json.dumps(report,indent=2))
    except RuntimeError as exc:
        operator_log(f"[ERROR] {exc}")
        raise SystemExit(2)


if __name__=="__main__":
    main()

