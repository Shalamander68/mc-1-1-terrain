#!/usr/bin/env python3
import argparse
import json
import os
import sys

# Local imports from our modules
import gsi_fetcher
import terrain_math
import mc_builder

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
    
    mc_x = prompt_int("  Origin X (NW corner)", 0)
    mc_z = prompt_int("  Origin Z (NW corner)", 0)
    
    # Calculate the grid to verify coordinates before downloading
    out_lats, out_lons = terrain_math.get_grid_arrays(lat_tl, lon_tl, lat_br, lon_br, spacing)
    rows, cols = len(out_lats), len(out_lons)

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

    mc_sea = prompt_int("\n  Sea-level Y (Usually 0 for 1:1 mods)", 0)
    real_sea = prompt_float("  Real-world sea level (m)", 0.0)
    vscale   = prompt_float("  Vertical scale (1 = 1 block/m)", 1.0)
    out_dir = input("\n  Output directory [./output]: ").strip() or "./output"

    return dict(
        lat_tl=lat_tl, lon_tl=lon_tl, lat_br=lat_br, lon_br=lon_br,
        spacing_m=spacing, mc_x=mc_x, mc_z=mc_z,
        mc_sea_level=mc_sea, real_sea_m=real_sea, vertical_scale=vscale,
        out_dir=out_dir,
    )

def run(cfg: dict):
    os.makedirs(cfg["out_dir"], exist_ok=True)

    print("\n[1/4] Fetching elevation data from GSI …")
    elev_raw, lats_raw, lons_raw = gsi_fetcher.fetch_region(
        cfg["lat_tl"], cfg["lon_tl"], cfg["lat_br"], cfg["lon_br"]
    )

    print("\n[2/4] Resampling math …")
    elev, lats, lons = terrain_math.resample_grid(
        elev_raw, lats_raw, lons_raw, 
        cfg["lat_tl"], cfg["lon_tl"], cfg["lat_br"], cfg["lon_br"],
        cfg["spacing_m"]
    )

    print("[3/4] Mapping to Minecraft coordinates …")
    mc_y = terrain_math.elev_to_mc_y(
        elev,
        mc_sea_level=cfg["mc_sea_level"],
        real_sea_m=cfg["real_sea_m"],
        vertical_scale=cfg["vertical_scale"],
    )

    print("[4/4] Building commands in memory …")
    cmds = mc_builder.build_commands(mc_y, cfg["mc_x"], cfg["mc_z"])

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
        macro_text = mc_builder.build_js_macro(cmds, cols, rows, cfg["mc_x"], cfg["mc_z"])
        
        with open(js_path, "w", encoding="utf-8") as fh:
            fh.write(macro_text)
            
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
    