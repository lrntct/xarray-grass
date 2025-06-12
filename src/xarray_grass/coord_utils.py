# coding=utf8
"""
Copyright (C) 2025 Laurent Courty

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
"""

from typing import Mapping
import numpy as np
import xarray as xr  # For type hinting xr.DataArray
from xarray_grass.grass_interface import GrassInterface


def get_coordinates(grass_i: GrassInterface, raster_3d: bool) -> dict:
    """return xarray coordinates from GRASS region."""
    current_region = grass_i.get_region()
    lim_e = current_region.e
    lim_w = current_region.w
    lim_n = current_region.n
    lim_s = current_region.s
    lim_t = current_region.t
    lim_b = current_region.b
    dz = current_region.tbres
    if raster_3d:
        dx = current_region.ewres3
        dy = current_region.nsres3
    else:
        dx = current_region.ewres
        dy = current_region.nsres
    # GRASS limits are at the edge of the region.
    # In the exported DataArray, coordinates are at the center of the cell
    # Stop not changed to include it in the range
    start_w = lim_w + dx / 2
    stop_e = lim_e
    start_s = lim_s + dy / 2
    stop_n = lim_n
    start_b = lim_b + dz / 2
    stop_t = lim_t
    x_coords = np.arange(start=start_w, stop=stop_e, step=dx, dtype=np.float32)
    y_coords = np.arange(start=start_s, stop=stop_n, step=dy, dtype=np.float32)
    z_coords = np.arange(start=start_b, stop=stop_t, step=dz, dtype=np.float32)
    return {"x": x_coords, "y": y_coords, "z": z_coords}


def get_region_from_xarray(data_array: xr.DataArray, dims: Mapping[str, str]) -> dict:
    """
    Calculates GRASS GIS region parameters from an xarray DataArray.

    This function analyzes the coordinates of the input xarray DataArray using the
    provided `dims` mapping to determine the bounding box (north, south, east,
    west, top, bottom) and resolutions (ewres, nsres, ewres3, nsres3, tbres).
    It's the inverse operation of `get_coordinates()`.

    GRASS limits are at the edge of the region, while xarray coordinates are
    typically at the center of the cell. This function accounts for this difference.
    The input DataArray is assumed to be pre-validated for correct spatial
    dimensions (2D or 3D). Time dimensions are ignored.

    Parameters
    ----------
    data_array : xr.DataArray
        The input xarray DataArray.
    dims : Mapping[str, str]
        A mapping from standard dimension names (e.g., 'latitude', 'longitude', 'z')
        to the actual dimension names used in the `data_array`. This mapping is
        expected to be complete and correct for the `data_array`.

    Returns
    -------
    dict
        A dictionary containing GRASS region parameters:
        'n', 's', 'e', 'w': float or None (geographical limits)
        't', 'b': float or None (top, bottom limits for 3D)
        'nsres', 'ewres': float or None (2D resolutions)
        'nsres3', 'ewres3': float or None (3D horizontal resolutions)
        'tbres': float or None (3D vertical resolution)
        Values can be None if not applicable or cannot be determined.
    """
    region = {
        "n": None,
        "s": None,
        "e": None,
        "w": None,
        "t": None,
        "b": None,
        "nsres": None,
        "ewres": None,
        "nsres3": None,
        "ewres3": None,
        "tbres": None,
    }

    def _calculate_res(coords_arr_np: np.ndarray) -> float | None:
        if coords_arr_np is not None and len(coords_arr_np) >= 2:
            # Ensure consistent dtype for subtraction, then convert to float
            res = np.abs(
                coords_arr_np[1].astype(float) - coords_arr_np[0].astype(float)
            )
            return float(res)
        return None

    # Determine if it's 3D based on presence of z-coordinate name in dims and data_array
    z_name = dims.get("z")
    is_3d = z_name and z_name in data_array.coords

    x_coords_np, y_coords_np, z_coords_np = None, None, None

    if is_3d:
        # Try 3D specific dim names first
        lon_name = dims.get("longitude_3d")
        lat_name = dims.get("latitude_3d")
        x_name = dims.get("x_3d")
        y_name = dims.get("y_3d")

        if (
            lon_name
            and lat_name
            and lon_name in data_array.coords
            and lat_name in data_array.coords
        ):
            x_coords_np = data_array.coords[lon_name].values
            y_coords_np = data_array.coords[lat_name].values
        elif (
            x_name
            and y_name
            and x_name in data_array.coords
            and y_name in data_array.coords
        ):
            x_coords_np = data_array.coords[x_name].values
            y_coords_np = data_array.coords[y_name].values

        if z_name in data_array.coords:  # Ensure z_name is valid before accessing
            z_coords_np = data_array.coords[z_name].values

        ewres_3d = _calculate_res(x_coords_np)
        nsres_3d = _calculate_res(y_coords_np)
        tbres_3d = _calculate_res(z_coords_np)
        region["ewres3"] = ewres_3d
        region["nsres3"] = nsres_3d
        region["tbres"] = tbres_3d
        # For 3D, also populate the 2D resolution fields.
        # GRASS often expects 2D res even for 3D regions.
        region["ewres"] = ewres_3d
        region["nsres"] = nsres_3d

    else:  # 2D case
        lon_name = dims.get("longitude")
        lat_name = dims.get("latitude")
        x_name = dims.get("x")
        y_name = dims.get("y")

        if (
            lon_name
            and lat_name
            and lon_name in data_array.coords
            and lat_name in data_array.coords
        ):
            x_coords_np = data_array.coords[lon_name].values
            y_coords_np = data_array.coords[lat_name].values
        elif (
            x_name
            and y_name
            and x_name in data_array.coords
            and y_name in data_array.coords
        ):
            x_coords_np = data_array.coords[x_name].values
            y_coords_np = data_array.coords[y_name].values

        ewres_2d = _calculate_res(x_coords_np)
        nsres_2d = _calculate_res(y_coords_np)
        region["ewres"] = ewres_2d
        region["nsres"] = nsres_2d

    # Calculate bounds (n, s, e, w)
    current_ewres = (
        region["ewres3"] if is_3d and region["ewres3"] is not None else region["ewres"]
    )
    current_nsres = (
        region["nsres3"] if is_3d and region["nsres3"] is not None else region["nsres"]
    )

    if x_coords_np is not None and current_ewres is not None and len(x_coords_np) > 0:
        region["w"] = float(x_coords_np[0] - current_ewres / 2)
        region["e"] = float(x_coords_np[-1] + current_ewres / 2)

    if y_coords_np is not None and current_nsres is not None and len(y_coords_np) > 0:
        y_ascending = True
        if len(y_coords_np) > 1:
            y_ascending = y_coords_np[0] < y_coords_np[-1]

        if y_ascending:
            region["s"] = float(y_coords_np[0] - current_nsres / 2)
            region["n"] = float(y_coords_np[-1] + current_nsres / 2)
        else:  # Descending
            region["s"] = float(y_coords_np[-1] - current_nsres / 2)
            region["n"] = float(y_coords_np[0] + current_nsres / 2)

    # Calculate 3D bounds (t, b) if applicable
    if (
        is_3d
        and z_coords_np is not None
        and region["tbres"] is not None
        and len(z_coords_np) > 0
    ):
        region["b"] = float(z_coords_np[0] - region["tbres"] / 2)
        region["t"] = float(z_coords_np[-1] + region["tbres"] / 2)

    return region
