#!/usr/bin/env python3
"""
terrain_to_minecraft.py
=======================
Fetches real-world elevation data from Japan's Geospatial Information Authority
(GSI) for a rectangular region, resamples it to a target grid spacing, and
generates JsMacros-compatible JavaScript that pastes the terrain into Minecraft
using /fill and /setblock commands.

Data source: https://cyberjapandata.gsi.go.jp
  - dem5a  zoom-15  ~4 m/px  (5 m mesh DEM, highest quality)
  - dem5b  zoom-15  ~4 m/px  (5 m mesh DEM, alternative source)
  - dem    zoom-14  ~8 m/px  (10 m mesh DEM, fallback)

Usage:
    python terrain_to_minecraft.py
    python terrain_to_minecraft.py --config example_config.json

Requirements:
    pip install numpy scipy requests
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import requests
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt

# ── Constants ──────────────────────────────────────────────────────────────────

GSI_TILE_URL = "https://cyberjapandata.gsi.go.jp/xyz/{tileset}/{z}/{x}/{y}.txt"
TILE_SIZE    = 256   # GSI DEM tiles are always 256 × 256

# Datasets tried in order — first one with data wins.
# Each entry: (tileset_name, zoom_level, nominal_mesh_size_m)
TILESETS = [
    ("dem5a", 15, 5),
    ("dem5b", 15, 5),
    ("dem",   14, 10),
]

# Minecraft 1.18+ world height bounds
MC_Y_MIN = -64
MC_Y_MAX = 320

# Block choices for terrain layers
BLOCK_SURFACE     = "minecraft:grass_block"
BLOCK_SUBSURFACE  = "minecraft:dirt"
BLOCK_FILL        = "minecraft:stone"

# How many dirt layers sit below the grass surface
DIRT_LAYERS = 3

# Seconds to wait between tile requests (be polite to GSI)
REQUEST_DELAY = 0.05

# ── Coordinate helpers ─────────────────────────────────────────────────────────

def latlon_to_global_pixel(lat: float, lon: float, zoom: int):
    """Return (px, py) global pixel position for a lat/lon at the given zoom."""
    n   = 2.0 ** zoom
    px  = (lon + 180.0) / 360.0 * n * TILE_SIZE
    lat_r = math.radians(lat)
    py  = (1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n * TILE_SIZE
    return px, py


def global_pixel_to_latlon(px: float, py: float, zoom: int):
    """Inverse of latlon_to_global_pixel."""
    n   = 2.0 ** zoom
    lon = px / (TILE_SIZE * n) * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * py / (TILE_SIZE * n)))))
    return lat, lon


def meters_per_pixel(lat: float, zoom: int) -> float:
    """Approximate ground resolution (m/px) at the given latitude and zoom."""
    return 156_543.034 * math.cos(math.radians(lat)) / (2 ** zoom)


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in metres between two lat/lon points."""
    R = 6_371_000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))

# ── GSI tile fetching ──────────────────────────────────────────────────────────

def fetch_tile(tileset: str, zoom: int, tx: int, ty: int) -> np.ndarray | None:
    """
    Fetch one GSI DEM text tile and return a float32 (256, 256) array.
    Nodata cells ('e') become NaN.  Returns None on 404 or parse error.
    """
    url = GSI_TILE_URL.format(tileset=tileset, z=zoom, x=tx, y=ty)
    headers = {
        # GSI servers require a browser-like User-Agent and a Referer to the GSI viewer.
        "User-Agent":  "Mozilla/5.0 (compatible; terrain_to_minecraft/1.0)",
        "Referer":     "https://cyberjapandata.gsi.go.jp/",
        "Accept":      "text/plain, */*",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()

        rows = []
        for line in resp.text.strip().splitlines():
            row = []
            for cell in line.split(","):
                cell = cell.strip()
                row.append(float("nan") if cell in ("e", "") else float(cell))
            rows.append(row)

        arr = np.array(rows, dtype=np.float32)
        if arr.shape != (TILE_SIZE, TILE_SIZE):
            print(f"    ⚠ Unexpected tile shape {arr.shape} for {url}")
            return None
        return arr

    except Exception as exc:
        print(f"    ⚠ Fetch error ({url}): {exc}")
        return None


def fill_nans(arr: np.ndarray) -> np.ndarray:
    """Fill NaN cells using nearest-neighbour propagation."""
    mask = np.isnan(arr)
    if not mask.any():
        return arr
    # distance_transform_edt returns the index of the nearest non-NaN cell
    _, nearest = distance_transform_edt(mask, return_distances=True, return_indices=True)
    return arr[tuple(nearest)]


def fetch_region(lat_tl: float, lon_tl: float, lat_br: float, lon_br: float):
    """
    Try each GSI tileset in order.  Stitches all required tiles into one
    elevation array cropped to the requested bounding box.

    Returns:
        elev   – float32 ndarray, shape (H, W), metres
        lats   – 1-D array of latitudes  for each row (decreasing, N→S)
        lons   – 1-D array of longitudes for each col (increasing, W→E)
    """
    center_lat = (lat_tl + lat_br) / 2

    for tileset, zoom, mesh_m in TILESETS:
        mpp = meters_per_pixel(center_lat, zoom)
        print(f"\n  Trying {tileset}  zoom={zoom}  nominal={mesh_m}m  ~{mpp:.1f} m/px")

        # Global pixel coordinates for the two corners
        px_tl, py_tl = latlon_to_global_pixel(lat_tl, lon_tl, zoom)
        px_br, py_br = latlon_to_global_pixel(lat_br, lon_br, zoom)

        # Note: lat_br < lat_tl  →  py_br > py_tl  (pixel Y grows southward)
        tx0, ty0 = int(px_tl // TILE_SIZE), int(py_tl // TILE_SIZE)
        tx1, ty1 = int(px_br // TILE_SIZE), int(py_br // TILE_SIZE)

        n_tiles = (tx1 - tx0 + 1) * (ty1 - ty0 + 1)
        print(f"  Fetching {n_tiles} tile(s) …")

        canvas_w = (tx1 - tx0 + 1) * TILE_SIZE
        canvas_h = (ty1 - ty0 + 1) * TILE_SIZE
        canvas   = np.full((canvas_h, canvas_w), np.nan, dtype=np.float32)
        fetched  = 0

        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                tile = fetch_tile(tileset, zoom, tx, ty)
                if tile is not None:
                    oy = (ty - ty0) * TILE_SIZE
                    ox = (tx - tx0) * TILE_SIZE
                    canvas[oy : oy + TILE_SIZE, ox : ox + TILE_SIZE] = tile
                    fetched += 1
                time.sleep(REQUEST_DELAY)

        if fetched == 0:
            print(f"  No data — skipping {tileset}")
            continue

        nan_pct = np.isnan(canvas).mean() * 100
        print(f"  {fetched}/{n_tiles} tiles fetched  ({nan_pct:.1f}% NaN before fill)")

        canvas = fill_nans(canvas)

        # ── Crop canvas to the exact bounding-box pixel extent ──────────────
        cx0 = int(px_tl) - tx0 * TILE_SIZE
        cy0 = int(py_tl) - ty0 * TILE_SIZE
        cx1 = math.ceil(px_br) - tx0 * TILE_SIZE
        cy1 = math.ceil(py_br) - ty0 * TILE_SIZE

        canvas = canvas[cy0 : cy1 + 1, cx0 : cx1 + 1]
        H, W   = canvas.shape

        # Build per-row lat and per-col lon arrays from global pixel positions
        lats = np.array([global_pixel_to_latlon(px_tl, py_tl + r, zoom)[0] for r in range(H)])
        lons = np.array([global_pixel_to_latlon(px_tl + c, py_tl,  zoom)[1] for c in range(W)])

        print(
            f"  Cropped to {W}×{H} px  |  "
            f"elev {np.nanmin(canvas):.1f} – {np.nanmax(canvas):.1f} m"
        )
        return canvas, lats, lons

    raise RuntimeError(
        "Could not retrieve elevation data for the requested region from any GSI dataset.\n"
        "Verify the coordinates are within Japan and reachable from this machine."
    )

# ── Resampling ─────────────────────────────────────────────────────────────────

def resample_grid(elev: np.ndarray, lats: np.ndarray, lons: np.ndarray, spacing_m: float):
    """
    Bilinearly resample *elev* to a regular grid with ~spacing_m ground
    resolution.

    Returns:
        elev_out  – float32 ndarray (rows, cols)
        out_lats  – 1-D float64 array, decreasing (N→S)
        out_lons  – 1-D float64 array, increasing (W→E)
    """
    clat = (lats[0] + lats[-1]) / 2

    # degrees of arc per metre at this latitude
    lat_per_m = 1.0 / 111_320.0
    lon_per_m = 1.0 / (111_320.0 * math.cos(math.radians(clat)))

    lat_step = lat_per_m * spacing_m   # always positive
    lon_step = lon_per_m * spacing_m

    # lats is decreasing → arange with negative step
    out_lats = np.arange(lats[0], lats[-1], -lat_step)
    out_lons = np.arange(lons[0], lons[-1],  lon_step)

    # RegularGridInterpolator requires strictly increasing axes.
    # Flip lats so they go low→high, flip elev rows accordingly.
    interp = RegularGridInterpolator(
        (lats[::-1], lons),
        elev[::-1, :],
        method="linear",
        bounds_error=False,
        fill_value=None,
    )

    # Build query points for every (lat, lon) in the output grid
    grid_lon, grid_lat = np.meshgrid(out_lons, out_lats)
    pts  = np.stack([grid_lat.ravel(), grid_lon.ravel()], axis=-1)
    out  = interp(pts).reshape(len(out_lats), len(out_lons)).astype(np.float32)

    print(
        f"  Resampled to {len(out_lons)}×{len(out_lats)} points  "
        f"({len(out_lons) * len(out_lats):,} total)  "
        f"at ~{spacing_m} m/block"
    )
    return out, out_lats, out_lons

# ── Minecraft height mapping ───────────────────────────────────────────────────

def elev_to_mc_y(
    elev:          np.ndarray,
    mc_sea_level:  int   = 64,
    real_sea_m:    float = 0.0,
    vertical_scale: float = 1.0,
) -> np.ndarray:
    """
    Map real-world elevation (metres) → Minecraft Y (integer blocks).

    Formula:  Y = mc_sea_level + round((elev − real_sea_m) × vertical_scale)
    Result is clamped to [MC_Y_MIN, MC_Y_MAX].
    """
    mc_y = np.round(mc_sea_level + (elev - real_sea_m) * vertical_scale).astype(int)
    mc_y = np.clip(mc_y, MC_Y_MIN, MC_Y_MAX)
    return mc_y

# ── JsMacros output ────────────────────────────────────────────────────────────

def _jsm_header(region_w: int, region_h: int, origin_x: int, origin_z: int) -> list[str]:
    return [
        "// ============================================================",
        "// Terrain paste script — generated by terrain_to_minecraft.py",
        f"// Region : {region_w} × {region_h} blocks",
        f"// Origin : X={origin_x}  Z={origin_z}",
        "// Usage  : Load in JsMacros 2.x  →  run with /execute script",
        "// ============================================================",
        "",
    ]


def generate_fill_script(
    mc_y:      np.ndarray,
    origin_x:  int,
    origin_z:  int,
    out_path:  str,
    base_y:    int = MC_Y_MIN,
):
    """
    Efficient output: one /fill per column (stone bedrock → dirt),
    plus one /setblock per cell for the grass surface.

    Each column: fill stone from base_y to surface−DIRT_LAYERS−1,
                 fill dirt from there to surface−1,
                 setblock grass at surface.
    """
    rows, cols = mc_y.shape
    cmds: list[str] = []

    for row in range(rows):
        for col in range(cols):
            mx = origin_x + col
            mz = origin_z + row
            sy = int(mc_y[row, col])

            stone_top = sy - DIRT_LAYERS - 1
            dirt_bot  = sy - DIRT_LAYERS
            dirt_top  = sy - 1

            if stone_top >= base_y:
                cmds.append(f"/fill {mx} {base_y} {mz} {mx} {stone_top} {mz} {BLOCK_FILL}")

            if dirt_top >= base_y:
                cmds.append(
                    f"/fill {mx} {max(base_y, dirt_bot)} {mz} "
                    f"{mx} {dirt_top} {mz} {BLOCK_SUBSURFACE}"
                )

            cmds.append(f"/setblock {mx} {sy} {mz} {BLOCK_SURFACE}")

    lines = (
        _jsm_header(cols, rows, origin_x, origin_z)
        + [
            "// Sends /fill (stone column) + /setblock (surface) per grid point.",
            "// For very large regions, consider chunking or using a data pack instead.",
            "",
            f"var cmds = {json.dumps(cmds)};",
            "",
            "var sent = 0;",
            "for (var i = 0; i < cmds.length; i++) {",
            "    Chat.say(cmds[i]);",
            "    sent++;",
            "}",
            "",
            "log('Terrain paste complete — sent ' + sent + ' commands.');",
        ]
    )

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"  → {out_path}  ({len(cmds):,} commands, {os.path.getsize(out_path) // 1024} KB)")


def generate_setblock_script(
    mc_y:     np.ndarray,
    origin_x: int,
    origin_z: int,
    out_path: str,
):
    """
    Surface-only output: one /setblock per cell.
    Faster to load and run, but only places the topmost block.
    Useful for previewing shape before running the fill script.
    """
    rows, cols = mc_y.shape
    cmds: list[str] = []

    for row in range(rows):
        for col in range(cols):
            mx = origin_x + col
            mz = origin_z + row
            sy = int(mc_y[row, col])
            cmds.append(f"/setblock {mx} {sy} {mz} {BLOCK_SURFACE}")

    lines = (
        _jsm_header(cols, rows, origin_x, origin_z)
        + [
            "// Surface-only preview — places one grass block per column.",
            "",
            f"var cmds = {json.dumps(cmds)};",
            "",
            "for (var i = 0; i < cmds.length; i++) {",
            "    Chat.say(cmds[i]);",
            "}",
            "",
            "log('Surface preview done — ' + cmds.length + ' blocks placed.');",
        ]
    )

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"  → {out_path}  ({len(cmds):,} setblocks, {os.path.getsize(out_path) // 1024} KB)")

# ── Interactive prompts ────────────────────────────────────────────────────────

def prompt_float(msg: str, default: float | None = None) -> float:
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"{msg}{suffix}: ").strip()
        if raw == "" and default is not None:
            return float(default)
        try:
            return float(raw)
        except ValueError:
            print("  Please enter a number.")


def prompt_int(msg: str, default: int | None = None) -> int:
    return int(prompt_float(msg, default))


def interactive_config() -> dict:
    print("┌─────────────────────────────────────────┐")
    print("│  GSI Terrain → Minecraft  (interactive) │")
    print("└─────────────────────────────────────────┘\n")

    print("── Bounding box (WGS-84) ──")
    lat_tl = prompt_float("  Top-left  latitude  (N)")
    lon_tl = prompt_float("  Top-left  longitude (E)")
    lat_br = prompt_float("  Bot-right latitude  (N)")
    lon_br = prompt_float("  Bot-right longitude (E)")

    if lat_br >= lat_tl:
        print("  ⚠ Bottom-right latitude must be south of (less than) top-left latitude.")
        sys.exit(1)
    if lon_br <= lon_tl:
        print("  ⚠ Bottom-right longitude must be east of (greater than) top-left longitude.")
        sys.exit(1)

    print("\n── Resolution ──")
    w_m = haversine_m(lat_tl, lon_tl, lat_tl, lon_br)
    h_m = haversine_m(lat_tl, lon_tl, lat_br, lon_tl)
    print(f"  Region size: ~{w_m:.0f} m wide × {h_m:.0f} m tall")
    spacing = prompt_float("  Target grid spacing in metres (1–100)", 5.0)
    spacing = max(1.0, min(100.0, spacing))
    est_blocks = int((w_m / spacing) * (h_m / spacing))
    print(f"  Estimated grid: ~{int(w_m/spacing)} × {int(h_m/spacing)} = {est_blocks:,} columns")
    if est_blocks > 500_000:
        print("  ⚠ Large region — script file may be several hundred MB.")
        print("    Consider a coarser spacing or a smaller region for testing.")

    print("\n── Minecraft placement ──")
    mc_x   = prompt_int  ("  Origin X (NW corner)", 0)
    mc_z   = prompt_int  ("  Origin Z (NW corner)", 0)
    mc_sea = prompt_int  ("  Sea-level Y           ", 64)
    real_sea = prompt_float("  Real-world sea level (m)", 0.0)
    vscale   = prompt_float("  Vertical scale (1 = 1 block/m)", 1.0)

    print("\n── Output ──")
    out_dir = input("  Output directory [./output]: ").strip() or "./output"

    return dict(
        lat_tl=lat_tl, lon_tl=lon_tl, lat_br=lat_br, lon_br=lon_br,
        spacing_m=spacing,
        mc_x=mc_x, mc_z=mc_z,
        mc_sea_level=mc_sea, real_sea_m=real_sea, vertical_scale=vscale,
        out_dir=out_dir,
    )

# ── Main ───────────────────────────────────────────────────────────────────────

def run(cfg: dict):
    os.makedirs(cfg["out_dir"], exist_ok=True)

    # 1. Fetch raw elevation tiles from GSI
    print("\n[1/4] Fetching elevation data from GSI …")
    elev_raw, lats_raw, lons_raw = fetch_region(
        cfg["lat_tl"], cfg["lon_tl"],
        cfg["lat_br"], cfg["lon_br"],
    )

    # 2. Resample to target grid spacing
    print("\n[2/4] Resampling to target resolution …")
    elev, lats, lons = resample_grid(elev_raw, lats_raw, lons_raw, cfg["spacing_m"])

    # 3. Convert elevation → Minecraft Y
    print("\n[3/4] Mapping to Minecraft coordinates …")
    mc_y = elev_to_mc_y(
        elev,
        mc_sea_level=cfg["mc_sea_level"],
        real_sea_m=cfg["real_sea_m"],
        vertical_scale=cfg["vertical_scale"],
    )
    rows, cols = mc_y.shape
    print(
        f"  Grid : {cols} × {rows} blocks  "
        f"(X: {cfg['mc_x']} – {cfg['mc_x']+cols-1}  "
        f"Z: {cfg['mc_z']} – {cfg['mc_z']+rows-1})"
    )
    print(f"  Y range in-world: {mc_y.min()} – {mc_y.max()}")

    # 4. Write output files
    print("\n[4/4] Writing output files …")

    # Save raw heightmap as CSV (integer Y values)
    csv_path = os.path.join(cfg["out_dir"], "heightmap.csv")
    np.savetxt(csv_path, mc_y, fmt="%d", delimiter=",")
    print(f"  → {csv_path}  (raw integer heightmap)")

    # Full fill script (stone + dirt + grass surface)
    fill_path = os.path.join(cfg["out_dir"], "terrain_fill.js")
    generate_fill_script(mc_y, cfg["mc_x"], cfg["mc_z"], fill_path)

    # Lightweight surface-only preview script
    preview_path = os.path.join(cfg["out_dir"], "terrain_preview.js")
    generate_setblock_script(mc_y, cfg["mc_x"], cfg["mc_z"], preview_path)

    # Save the config used for this run
    cfg_path = os.path.join(cfg["out_dir"], "run_config.json")
    with open(cfg_path, "w") as fh:
        json.dump(cfg, fh, indent=2)

    print(f"\n✓ All done!  Output in: {os.path.abspath(cfg['out_dir'])}")
    print("  Load terrain_fill.js in JsMacros to paste the full terrain.")
    print("  Use terrain_preview.js first to check placement before filling.")


def main():
    parser = argparse.ArgumentParser(description="GSI elevation → Minecraft terrain")
    parser.add_argument(
        "--config", metavar="FILE",
        help="JSON config file (skips interactive prompts). "
             "Keys: lat_tl, lon_tl, lat_br, lon_br, spacing_m, "
             "mc_x, mc_z, mc_sea_level, real_sea_m, vertical_scale, out_dir",
    )
    args = parser.parse_args()

    if args.config:
        with open(args.config) as fh:
            cfg = json.load(fh)
        print(f"Loaded config from {args.config}")
    else:
        cfg = interactive_config()

    run(cfg)


if __name__ == "__main__":
    main()
