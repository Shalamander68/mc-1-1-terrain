"""Module for geographic resampling and Minecraft coordinate conversions."""

import math
import numpy as np
from scipy.interpolate import RegularGridInterpolator

# Limits adjusted to accommodate servers using build-height mods
MC_Y_MIN = -30000
MC_Y_MAX = 30000

def get_grid_arrays(lat_tl: float, lon_tl: float, lat_br: float, lon_br: float, spacing_m: float):
    """Calculates the target latitude and longitude arrays based on block spacing (meters)."""
    # Estimate lat/lon degrees per meter based on the region's center latitude
    clat = (lat_tl + lat_br) / 2
    lat_per_m = 1.0 / 111_320.0
    lon_per_m = 1.0 / (111_320.0 * math.cos(math.radians(clat)))

    lat_step = lat_per_m * spacing_m
    lon_step = lon_per_m * spacing_m

    # Create arrays stepping from top-left to bottom-right by the calculated block spacing
    out_lats = np.arange(lat_tl, lat_br, -lat_step)
    out_lons = np.arange(lon_tl, lon_br, lon_step)
    
    return out_lats, out_lons

def resample_grid(elev: np.ndarray, lats: np.ndarray, lons: np.ndarray, 
                  lat_tl: float, lon_tl: float, lat_br: float, lon_br: float, spacing_m: float):
    """Interpolates raw elevation data onto a uniform grid spaced by `spacing_m`."""
    out_lats, out_lons = get_grid_arrays(lat_tl, lon_tl, lat_br, lon_br, spacing_m)

    # Set up the linear interpolator with the raw data
    interp = RegularGridInterpolator(
        (lats[::-1], lons),
        elev[::-1, :],
        method="linear",
        bounds_error=False,
        fill_value=None,
    )

    # Generate a meshgrid of target coordinates and interpolate new elevation values
    grid_lon, grid_lat = np.meshgrid(out_lons, out_lats)
    pts  = np.stack([grid_lat.ravel(), grid_lon.ravel()], axis=-1)
    out  = interp(pts).reshape(len(out_lats), len(out_lons)).astype(np.float32)

    return out, out_lats, out_lons

def elev_to_mc_y(elev: np.ndarray, mc_sea_level: int = 0, real_sea_m: float = 0.0, vertical_scale: float = 1.0) -> np.ndarray:
    """Scales real-world elevation in meters to Minecraft block Y-coordinates."""
    # Apply sea level offset and vertical scaling
    mc_y = np.round(mc_sea_level + (elev - real_sea_m) * vertical_scale).astype(int)
    # Clamp to build limits to prevent impossible commands
    mc_y = np.clip(mc_y, MC_Y_MIN, MC_Y_MAX)
    return mc_y