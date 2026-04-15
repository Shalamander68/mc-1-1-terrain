#!/usr/bin/env python3
"""
terrain_to_minecraft.py
=======================
Fetches real-world elevation data from Japan's Geospatial Information Authority
(GSI) for a rectangular region, resamples it to a target grid spacing, and
generates JsMacros-compatible JavaScript that pastes the terrain into Minecraft.

Data source: https://cyberjapandata.gsi.go.jp
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
TILE_SIZE    = 256

TILESETS = [
    ("dem5a", 15, 5),
    ("dem5b", 15, 5),
    ("dem",   14, 10),
]

# Vastly expanded limits for servers with build-height mods
MC_Y_MIN = -30000
MC_Y_MAX = 30000

BLOCK_SURFACE = "minecraft:grass_block"

# Seconds to wait between tile requests
REQUEST_DELAY = 0.05

# ── Coordinate helpers ─────────────────────────────────────────────────────────

def latlon_to_global_pixel(lat: float, lon: float, zoom: int):
    n   = 2.0 ** zoom
    px  = (lon + 180.0) / 360.0 * n * TILE_SIZE
    lat_r = math.radians(lat)
    py  = (1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n * TILE_SIZE
    return px, py

def global_pixel_to_latlon(px: float, py: float, zoom: int):
    n   = 2.0 ** zoom
    lon = px / (TILE_SIZE * n) * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * py / (TILE_SIZE * n)))))
    return lat, lon

def meters_per_pixel(lat: float, zoom: int) -> float:
    return 156_543.034 * math.cos(math.radians(lat)) / (2 ** zoom)

# ── GSI tile fetching ──────────────────────────────────────────────────────────

def fetch_tile(tileset: str, zoom: int, tx: int, ty: int) -> np.ndarray | None:
    url = GSI_TILE_URL.format(tileset=tileset, z=zoom, x=tx, y=ty)
    headers = {
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

        return np.array(rows, dtype=np.float32)
    except Exception as exc:
        return None

def fill_nans(arr: np.ndarray) -> np.ndarray:
    mask = np.isnan(arr)
    if not mask.any():
        return arr
    _, nearest = distance_transform_edt(mask, return_distances=True, return_indices=True)
    return arr[tuple(nearest)]

def fetch_region(lat_tl: float, lon_tl: float, lat_br: float, lon_br: float):
    center_lat = (lat_tl + lat_br) / 2

    for tileset, zoom, mesh_m in TILESETS:
        px_tl, py_tl = latlon_to_global_pixel(lat_tl, lon_tl, zoom)
        px_br, py_br = latlon_to_global_pixel(lat_br, lon_br, zoom)

        tx0, ty0 = int(px_tl // TILE_SIZE), int(py_tl // TILE_SIZE)
        tx1, ty1 = int(px_br // TILE_SIZE), int(py_br // TILE_SIZE)

        n_tiles = (tx1 - tx0 + 1) * (ty1 - ty0 + 1)
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
                
                # Light progress indicator
                print(f"  \r  [Downloading] {fetched}/{n_tiles} tiles fetched...", end="", flush=True)
                time.sleep(REQUEST_DELAY)
        print() 

        if fetched == 0:
            continue

        canvas = fill_nans(canvas)

        cx0 = int(px_tl) - tx0 * TILE_SIZE
        cy0 = int(py_tl) - ty0 * TILE_SIZE
        cx1 = math.ceil(px_br) - tx0 * TILE_SIZE
        cy1 = math.ceil(py_br) - ty0 * TILE_SIZE

        canvas = canvas[cy0 : cy1 + 1, cx0 : cx1 + 1]
        H, W   = canvas.shape

        lats = np.array([global_pixel_to_latlon(px_tl, py_tl + r, zoom)[0] for r in range(H)])
        lons = np.array([global_pixel_to_latlon(px_tl + c, py_tl,  zoom)[1] for c in range(W)])

        return canvas, lats, lons

    raise RuntimeError("Could not retrieve elevation data.")

# ── Resampling ─────────────────────────────────────────────────────────────────

def resample_grid(elev: np.ndarray, lats: np.ndarray, lons: np.ndarray, 
                  lat_tl: float, lon_tl: float, lat_br: float, lon_br: float, spacing_m: float):
    clat = (lat_tl + lat_br) / 2
    lat_per_m = 1.0 / 111_320.0
    lon_per_m = 1.0 / (111_320.0 * math.cos(math.radians(clat)))

    lat_step = lat_per_m * spacing_m
    lon_step = lon_per_m * spacing_m

    out_lats = np.arange(lat_tl, lat_br, -lat_step)
    out_lons = np.arange(lon_tl, lon_br, lon_step)

    interp = RegularGridInterpolator(
        (lats[::-1], lons),
        elev[::-1, :],
        method="linear",
        bounds_error=False,
        fill_value=None,
    )

    grid_lon, grid_lat = np.meshgrid(out_lons, out_lats)
    pts  = np.stack([grid_lat.ravel(), grid_lon.ravel()], axis=-1)
    out  = interp(pts).reshape(len(out_lats), len(out_lons)).astype(np.float32)

    return out, out_lats, out_lons

def elev_to_mc_y(elev: np.ndarray, mc_sea_level: int = 0, real_sea_m: float = 0.0, vertical_scale: float = 1.0) -> np.ndarray:
    mc_y = np.round(mc_sea_level + (elev - real_sea_m) * vertical_scale).astype(int)
    mc_y = np.clip(mc_y, MC_Y_MIN, MC_Y_MAX)
    return mc_y

# ── Command Generation ─────────────────────────────────────────────────────────

def build_commands(mc_y: np.ndarray, origin_x: int, origin_z: int):
    """
    Builds commands for a seamless surface. If an adjacent block is lower,
    it fills down to that neighbor's height to seal holes.
    """
    rows, cols = mc_y.shape
    cmds: list[str] = []

    for r in range(rows):
        for c in range(cols):
            mx = origin_x + c
            mz = origin_z + r
            sy = int(mc_y[r, c])

            # Find the lowest neighbor to seal gaps
            min_y = sy
            if r > 0: min_y = min(min_y, int(mc_y[r-1, c]))
            if r < rows - 1: min_y = min(min_y, int(mc_y[r+1, c]))
            if c > 0: min_y = min(min_y, int(mc_y[r, c-1]))
            if c < cols - 1: min_y = min(min_y, int(mc_y[r, c+1]))

            if sy == min_y:
                cmds.append(f"/setblock {mx} {sy} {mz} {BLOCK_SURFACE}")
            else:
                # Fills the vertical gap downwards so no holes are visible on slopes
                cmds.append(f"/fill {mx} {min_y} {mz} {mx} {sy} {mz} {BLOCK_SURFACE}")
                
    return cmds

def _jsm_header(region_w: int, region_h: int, origin_x: int, origin_z: int) -> list[str]:
    return [
        "// ============================================================",
        "// Terrain surface paste script — generated by terrain_to_minecraft.py",
        f"// Region : {region_w} × {region_h} blocks",
        f"// Origin : X={origin_x}  Z={origin_z}",
        "// ============================================================",
        "",
    ]

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

    if lat_br >= lat_tl or lon_br <= lon_tl:
        print("  ⚠ Invalid bounding box coordinates.")
        sys.exit(1)

    print("\n── Resolution & Coordinates ──")
    spacing = prompt_float("  Target grid spacing in metres (1–100)", 1.0)
    
    mc_x   = prompt_int  ("  Origin X (NW corner)", 0)
    mc_z   = prompt_int  ("  Origin Z (NW corner)", 0)
    
    # Pre-calculate the grid for verification
    clat = (lat_tl + lat_br) / 2
    lat_per_m = 1.0 / 111_320.0
    lon_per_m = 1.0 / (111_320.0 * math.cos(math.radians(clat)))

    lat_step = lat_per_m * spacing
    lon_step = lon_per_m * spacing

    out_lats = np.arange(lat_tl, lat_br, -lat_step)
    out_lons = np.arange(lon_tl, lon_br, lon_step)
    rows = len(out_lats)
    cols = len(out_lons)

    if rows == 0 or cols == 0:
        print("  ⚠ Grid resolves to 0 blocks. Check your coordinates and spacing.")
        sys.exit(1)

    print("\n── Pre-Flight Coordinate Verification ──")
    print("Please verify these corners in-game using /tpll to ensure alignment:")
    print(f"  Top-Left     | /tpll {lat_tl:.6f} {lon_tl:.6f}       ->  X: {mc_x}, Z: {mc_z}")
    print(f"  Top-Right    | /tpll {lat_tl:.6f} {out_lons[-1]:.6f}       ->  X: {mc_x + cols - 1}, Z: {mc_z}")
    print(f"  Bottom-Left  | /tpll {out_lats[-1]:.6f} {lon_tl:.6f}       ->  X: {mc_x}, Z: {mc_z + rows - 1}")
    print(f"  Bottom-Right | /tpll {out_lats[-1]:.6f} {out_lons[-1]:.6f}       ->  X: {mc_x + cols - 1}, Z: {mc_z + rows - 1}")
    
    while True:
        ans = input("\nDo these coordinates line up correctly? Proceed? (y/n): ").strip().lower()
        if ans == 'y': break
        elif ans == 'n': sys.exit(0)

    mc_sea = prompt_int  ("\n  Sea-level Y (Usually 0 for 1:1 mods)", 0)
    real_sea = prompt_float("  Real-world sea level (m)", 0.0)
    vscale   = prompt_float("  Vertical scale (1 = 1 block/m)", 1.0)
    out_dir = input("\n  Output directory [./output]: ").strip() or "./output"

    return dict(
        lat_tl=lat_tl, lon_tl=lon_tl, lat_br=lat_br, lon_br=lon_br,
        spacing_m=spacing, mc_x=mc_x, mc_z=mc_z,
        mc_sea_level=mc_sea, real_sea_m=real_sea, vertical_scale=vscale,
        out_dir=out_dir,
    )

# ── Main ───────────────────────────────────────────────────────────────────────

def run(cfg: dict):
    os.makedirs(cfg["out_dir"], exist_ok=True)

    print("\n[1/4] Fetching elevation data from GSI …")
    elev_raw, lats_raw, lons_raw = fetch_region(
        cfg["lat_tl"], cfg["lon_tl"],
        cfg["lat_br"], cfg["lon_br"],
    )

    print("\n[2/4] Resampling math …")
    elev, lats, lons = resample_grid(
        elev_raw, lats_raw, lons_raw, 
        cfg["lat_tl"], cfg["lon_tl"], cfg["lat_br"], cfg["lon_br"],
        cfg["spacing_m"]
    )

    print("[3/4] Mapping to Minecraft coordinates …")
    mc_y = elev_to_mc_y(
        elev,
        mc_sea_level=cfg["mc_sea_level"],
        real_sea_m=cfg["real_sea_m"],
        vertical_scale=cfg["vertical_scale"],
    )

    print("[4/4] Building commands in memory …")
    cmds = build_commands(mc_y, cfg["mc_x"], cfg["mc_z"])

    rows, cols = mc_y.shape
    blocks = rows * cols
    chunks = (cols / 16.0) * (rows / 16.0)

    print(f"\n┌─────────────────────────────────────────┐")
    print(f"│             GENERATION READY            │")
    print(f"└─────────────────────────────────────────┘")
    print(f"  Region Size    : {blocks:,} blocks ({chunks:,.2f} chunks)")
    print(f"  Total Commands : {len(cmds):,}")
    print(f"  Y Range        : {mc_y.min()} to {mc_y.max()}")

    ans = input("\nDo you want to generate the JsMacros script to send these commands to Minecraft? (y/n): ").strip().lower()
    
    if ans == 'y':
        js_path = os.path.join(cfg["out_dir"], "terrain_surface.js")
        lines = _jsm_header(cols, rows, cfg["mc_x"], cfg["mc_z"]) + [
            f"var cmds = {json.dumps(cmds)};",
            "",
            "var sent = 0;",
            "for (var i = 0; i < cmds.length; i++) {",
            "    Chat.say(cmds[i]);",
            "    sent++;",
            "}",
            "",
            "log('Terrain surface complete — sent ' + sent + ' commands.');",
        ]
        
        with open(js_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
            
        print(f"\n✓ Saved successfully to: {os.path.abspath(js_path)}")
        print("  Load terrain_surface.js in JsMacros to build the terrain.")
    else:
        print("\nAborted. Commands were not saved.")

def main():
    parser = argparse.ArgumentParser(description="GSI elevation → Minecraft terrain")
    parser.add_argument("--config", metavar="FILE", help="JSON config file")
    args = parser.parse_args()

    if args.config:
        with open(args.config) as fh:
            cfg = json.load(fh)
    else:
        cfg = interactive_config()

    run(cfg)

if __name__ == "__main__":
    main()