"""Module for downloading and assembling elevation tiles from the GSI API."""

import math
import time
import numpy as np
import requests
from scipy.ndimage import distance_transform_edt

# GSI API endpoint pattern and standard Web Mercator tile size
GSI_TILE_URL = "https://cyberjapandata.gsi.go.jp/xyz/{tileset}/{z}/{x}/{y}.txt"
TILE_SIZE    = 256
REQUEST_DELAY = 0.05

# Fallback order for elevation data: (dataset_name, zoom_level, mesh_resolution_meters)
TILESETS = [
    ("dem5a", 15, 5), # Highest quality 5m mesh
    ("dem5b", 15, 5), # Alternative 5m mesh
    ("dem",   14, 10),# Fallback 10m mesh
]

def latlon_to_global_pixel(lat: float, lon: float, zoom: int):
    """Converts WGS-84 coordinates to absolute pixel coordinates at a specific zoom level."""
    n   = 2.0 ** zoom
    px  = (lon + 180.0) / 360.0 * n * TILE_SIZE
    lat_r = math.radians(lat)
    py  = (1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n * TILE_SIZE
    return px, py

def global_pixel_to_latlon(px: float, py: float, zoom: int):
    """Converts absolute pixel coordinates back to WGS-84 coordinates."""
    n   = 2.0 ** zoom
    lon = px / (TILE_SIZE * n) * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * py / (TILE_SIZE * n)))))
    return lat, lon

def _fetch_tile(tileset: str, zoom: int, tx: int, ty: int) -> np.ndarray | None:
    """Downloads a single elevation tile and parses its CSV text into a numpy array."""
    url = GSI_TILE_URL.format(tileset=tileset, z=zoom, x=tx, y=ty)
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; terrain_to_minecraft/1.0)",
        "Referer": "https://cyberjapandata.gsi.go.jp/",
        "Accept": "text/plain, */*",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        if resp.status_code == 404:
            return None # Tile doesn't exist in this dataset
        resp.raise_for_status()

        rows = []
        # Parse the comma-separated text file, converting "e" (error/nodata) to NaN
        for line in resp.text.strip().splitlines():
            row = []
            for cell in line.split(","):
                cell = cell.strip()
                row.append(float("nan") if cell in ("e", "") else float(cell))
            rows.append(row)

        return np.array(rows, dtype=np.float32)
    except Exception:
        return None

def _fill_nans(arr: np.ndarray) -> np.ndarray:
    """Fills missing data (NaNs) by copying the nearest valid elevation value."""
    mask = np.isnan(arr)
    if not mask.any():
        return arr
    # Find the indices of the nearest valid pixels using Euclidean distance
    _, nearest = distance_transform_edt(mask, return_distances=True, return_indices=True)
    return arr[tuple(nearest)]

def fetch_region(lat_tl: float, lon_tl: float, lat_br: float, lon_br: float):
    """Downloads and stitches together all tiles needed to cover the requested bounding box."""
    # Attempt to fetch using highest quality data first, falling back if missing
    for tileset, zoom, _ in TILESETS:
        # Calculate bounding box in pixel coordinates
        px_tl, py_tl = latlon_to_global_pixel(lat_tl, lon_tl, zoom)
        px_br, py_br = latlon_to_global_pixel(lat_br, lon_br, zoom)

        # Convert pixel coordinates to tile coordinates
        tx0, ty0 = int(px_tl // TILE_SIZE), int(py_tl // TILE_SIZE)
        tx1, ty1 = int(px_br // TILE_SIZE), int(py_br // TILE_SIZE)

        n_tiles = (tx1 - tx0 + 1) * (ty1 - ty0 + 1)
        canvas_w = (tx1 - tx0 + 1) * TILE_SIZE
        canvas_h = (ty1 - ty0 + 1) * TILE_SIZE
        
        # Create a blank canvas to hold the stitched tiles
        canvas   = np.full((canvas_h, canvas_w), np.nan, dtype=np.float32)
        fetched  = 0

        # Download and paste each tile into the canvas
        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                tile = _fetch_tile(tileset, zoom, tx, ty)
                if tile is not None:
                    oy = (ty - ty0) * TILE_SIZE
                    ox = (tx - tx0) * TILE_SIZE
                    canvas[oy : oy + TILE_SIZE, ox : ox + TILE_SIZE] = tile
                    fetched += 1
                
                # Print an inline progress indicator
                print(f"  \r  [Downloading] {fetched}/{n_tiles} tiles fetched...", end="", flush=True)
                time.sleep(REQUEST_DELAY) # Be polite to the API
        print() 

        if fetched == 0:
            continue # Try the next fallback dataset if this one had no tiles

        # Patch over any missing data points inside the canvas
        canvas = _fill_nans(canvas)

        # Crop the canvas down to the exact requested bounding box
        cx0 = int(px_tl) - tx0 * TILE_SIZE
        cy0 = int(py_tl) - ty0 * TILE_SIZE
        cx1 = math.ceil(px_br) - tx0 * TILE_SIZE
        cy1 = math.ceil(py_br) - ty0 * TILE_SIZE

        canvas = canvas[cy0 : cy1 + 1, cx0 : cx1 + 1]
        H, W   = canvas.shape

        # Generate corresponding latitude and longitude arrays for every pixel in the crop
        lats = np.array([global_pixel_to_latlon(px_tl, py_tl + r, zoom)[0] for r in range(H)])
        lons = np.array([global_pixel_to_latlon(px_tl + c, py_tl,  zoom)[1] for c in range(W)])

        return canvas, lats, lons

    raise RuntimeError("Could not retrieve elevation data.")