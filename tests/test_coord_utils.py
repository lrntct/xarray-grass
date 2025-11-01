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

import xarray as xr
import numpy as np
import pytest

from xarray_grass.to_grass import DimensionsFormatter
from xarray_grass.coord_utils import get_region_from_xarray

default_dims = DimensionsFormatter.default_dims


def test_get_region_from_xarray_2d_xy():
    """Test get_region_from_xarray with a 2D DataArray using x, y."""
    data = np.arange(12).reshape(3, 4)
    x_coords = np.array([10, 20, 30, 40])
    y_coords = np.array([5, 15, 25])
    da = xr.DataArray(
        data,
        coords={"x": x_coords, "y": y_coords},
        dims=["y", "x"],
    )
    dims_map = default_dims.copy()
    region = get_region_from_xarray(da, dims_map)

    expected_ewres = 10.0
    expected_nsres = 10.0
    expected_w = x_coords[0] - expected_ewres / 2
    expected_e = x_coords[-1] + expected_ewres / 2
    expected_s = y_coords[0] - expected_nsres / 2
    expected_n = y_coords[-1] + expected_nsres / 2

    assert region.w == pytest.approx(expected_w)
    assert region.e == pytest.approx(expected_e)
    assert region.s == pytest.approx(expected_s)
    assert region.n == pytest.approx(expected_n)
    assert region.ewres == pytest.approx(expected_ewres)
    assert region.nsres == pytest.approx(expected_nsres)
    assert region.t is None
    assert region.b is None
    assert region.ewres3 is None
    assert region.nsres3 is None
    assert region.tbres is None


def test_get_region_from_xarray_2d_latlon():
    """Test get_region_from_xarray with a 2D DataArray using longitude, latitude with custom mapping."""
    data = np.arange(12).reshape(3, 4)
    lon_coords = np.array([-100.0, -90.0, -80.0, -70.0])
    lat_coords = np.array([30.0, 40.0, 50.0])
    da = xr.DataArray(
        data,
        coords={"longitude": lon_coords, "latitude": lat_coords},
        dims=["latitude", "longitude"],
    )
    dims_map = default_dims.copy()
    dims_map["x"] = "longitude"
    dims_map["y"] = "latitude"
    region = get_region_from_xarray(da, dims_map)

    expected_ewres = 10.0
    expected_nsres = 10.0
    expected_w = lon_coords[0] - expected_ewres / 2
    expected_e = lon_coords[-1] + expected_ewres / 2
    expected_s = lat_coords[0] - expected_nsres / 2
    expected_n = lat_coords[-1] + expected_nsres / 2

    assert region.w == pytest.approx(expected_w)
    assert region.e == pytest.approx(expected_e)
    assert region.s == pytest.approx(expected_s)
    assert region.n == pytest.approx(expected_n)
    assert region.ewres == pytest.approx(expected_ewres)
    assert region.nsres == pytest.approx(expected_nsres)
    assert region.t is None
    assert region.b is None
    assert region.ewres3 is None
    assert region.nsres3 is None
    assert region.tbres is None


def test_get_region_from_xarray_2d_latlon_descending_lat():
    """Test get_region_from_xarray with descending latitude using custom mapping."""
    data = np.arange(12).reshape(3, 4)
    lon_coords = np.array([-100.0, -90.0, -80.0, -70.0])
    lat_coords = np.array([50.0, 40.0, 30.0])  # Descending latitude
    da = xr.DataArray(
        data,
        coords={"longitude": lon_coords, "latitude": lat_coords},
        dims=["latitude", "longitude"],
    )
    dims_map = default_dims.copy()
    dims_map["x"] = "longitude"
    dims_map["y"] = "latitude"
    region = get_region_from_xarray(da, dims_map)

    expected_ewres = 10.0
    expected_nsres = 10.0  # abs(40-50)
    expected_w = lon_coords[0] - expected_ewres / 2
    expected_e = lon_coords[-1] + expected_ewres / 2
    # For descending lat, s is calculated from lat_coords[-1] and n from lat_coords[0]
    expected_s = lat_coords[-1] - expected_nsres / 2
    expected_n = lat_coords[0] + expected_nsres / 2

    assert region.w == pytest.approx(expected_w)
    assert region.e == pytest.approx(expected_e)
    assert region.s == pytest.approx(expected_s)
    assert region.n == pytest.approx(expected_n)
    assert region.ewres == pytest.approx(expected_ewres)
    assert region.nsres == pytest.approx(expected_nsres)


def test_get_region_from_xarray_3d_xyz():
    """Test get_region_from_xarray with a 3D DataArray using x_3d, y_3d, z."""
    data = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    x_coords = np.array([10, 20, 30, 40])
    y_coords = np.array([5, 15, 25])
    z_coords = np.array([100, 200])
    da = xr.DataArray(
        data,
        coords={"x_3d": x_coords, "y_3d": y_coords, "z": z_coords},
        dims=["z", "y_3d", "x_3d"],
    )
    dims_map = default_dims.copy()
    region = get_region_from_xarray(da, dims_map)

    expected_ewres3 = 10.0
    expected_nsres3 = 10.0
    expected_tbres = 100.0
    expected_w = x_coords[0] - expected_ewres3 / 2
    expected_e = x_coords[-1] + expected_ewres3 / 2
    expected_s = y_coords[0] - expected_nsres3 / 2
    expected_n = y_coords[-1] + expected_nsres3 / 2
    expected_b = z_coords[0] - expected_tbres / 2
    expected_t = z_coords[-1] + expected_tbres / 2

    assert region.w == pytest.approx(expected_w)
    assert region.e == pytest.approx(expected_e)
    assert region.s == pytest.approx(expected_s)
    assert region.n == pytest.approx(expected_n)
    assert region.b == pytest.approx(expected_b)
    assert region.t == pytest.approx(expected_t)
    assert region.ewres3 == pytest.approx(expected_ewres3)
    assert region.nsres3 == pytest.approx(expected_nsres3)
    assert region.tbres == pytest.approx(expected_tbres)


def test_get_region_from_xarray_3d_latlonz():
    """Test get_region_from_xarray with a 3D DataArray using longitude_3d, latitude_3d, z with custom mapping."""
    data = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    lon_coords = np.array([-100.0, -90.0, -80.0, -70.0])
    lat_coords = np.array([30.0, 40.0, 50.0])
    z_coords = np.array([1000.0, 2000.0])
    da = xr.DataArray(
        data,
        coords={
            "longitude_3d": lon_coords,
            "latitude_3d": lat_coords,
            "z": z_coords,
        },
        dims=["z", "latitude_3d", "longitude_3d"],
    )
    dims_map = default_dims.copy()
    dims_map["x_3d"] = "longitude_3d"
    dims_map["y_3d"] = "latitude_3d"
    region = get_region_from_xarray(da, dims_map)

    expected_ewres3 = 10.0
    expected_nsres3 = 10.0
    expected_tbres = 1000.0
    expected_w = lon_coords[0] - expected_ewres3 / 2
    expected_e = lon_coords[-1] + expected_ewres3 / 2
    expected_s = lat_coords[0] - expected_nsres3 / 2
    expected_n = lat_coords[-1] + expected_nsres3 / 2
    expected_b = z_coords[0] - expected_tbres / 2
    expected_t = z_coords[-1] + expected_tbres / 2

    assert region.w == pytest.approx(expected_w)
    assert region.e == pytest.approx(expected_e)
    assert region.s == pytest.approx(expected_s)
    assert region.n == pytest.approx(expected_n)
    assert region.b == pytest.approx(expected_b)
    assert region.t == pytest.approx(expected_t)
    assert region.ewres3 == pytest.approx(expected_ewres3)
    assert region.nsres3 == pytest.approx(expected_nsres3)
    assert region.tbres == pytest.approx(expected_tbres)


def test_get_region_from_xarray_missing_coord():
    """Test with a coordinate missing, expecting None for its resolution and bounds."""
    data = np.arange(12).reshape(3, 4)
    x_coords = np.array([10, 20, 30, 40])
    # y_coords is missing
    da = xr.DataArray(
        data,
        coords={"x": x_coords},  # y is intentionally missing
        dims=["dim_0", "x"],  # Using generic dim name for the one without coords
    )
    dims_map = default_dims.copy()  # 'y' will be in dims_map but not in da.coords
    region = get_region_from_xarray(da, dims_map)

    assert region.w is None
    assert region.e is None
    assert region.s is None
    assert region.n is None
    assert region.ewres is None
    assert region.nsres is None  # y-resolution should be None
    assert region.t is None
    assert region.b is None
    assert region.ewres3 is None
    assert region.nsres3 is None
    assert region.tbres is None


def test_get_region_from_xarray_single_point_coord():
    """Test with a coordinate having only a single point, expecting None for resolution."""
    data = np.arange(4).reshape(1, 4)  # Single point in y-dimension
    x_coords = np.array([10, 20, 30, 40])
    y_coords = np.array([5])  # Single y-coordinate
    da = xr.DataArray(
        data,
        coords={"x": x_coords, "y": y_coords},
        dims=["y", "x"],
    )
    dims_map = default_dims.copy()
    region = get_region_from_xarray(da, dims_map)

    expected_ewres = 10.0
    expected_w = x_coords[0] - expected_ewres / 2
    expected_e = x_coords[-1] + expected_ewres / 2
    # For single point, nsres is None, so s and n should also be None as per current logic
    # or they could be y_coords[0] +/- a default small value if resolution is None.
    # The function currently sets them to None if res is None.

    assert region.w == pytest.approx(expected_w)
    assert region.e == pytest.approx(expected_e)
    assert region.s is None  # Since nsres will be None
    assert region.n is None  # Since nsres will be None
    assert region.ewres == pytest.approx(expected_ewres)
    assert region.nsres is None  # Resolution for single point coord is None
    assert region.t is None
    assert region.b is None
    assert region.ewres3 is None
    assert region.nsres3 is None
    assert region.tbres is None


def test_get_region_from_xarray_custom_dim_names():
    """Test get_region_from_xarray with custom dimension names."""
    data = np.arange(12).reshape(3, 4)
    custom_x_coords = np.array([100, 200, 300, 400])
    custom_y_coords = np.array([50, 150, 250])
    da = xr.DataArray(
        data,
        coords={"my_x": custom_x_coords, "my_y": custom_y_coords},
        dims=["my_y", "my_x"],
    )
    dims_map = {
        "x": "my_x",
        "y": "my_y",
        **{k: default_dims[k] for k in default_dims if k not in ["x", "y"]},
    }
    region = get_region_from_xarray(da, dims_map)

    expected_ewres = 100.0
    expected_nsres = 100.0
    expected_w = custom_x_coords[0] - expected_ewres / 2
    expected_e = custom_x_coords[-1] + expected_ewres / 2
    expected_s = custom_y_coords[0] - expected_nsres / 2
    expected_n = custom_y_coords[-1] + expected_nsres / 2

    assert region.w == pytest.approx(expected_w)
    assert region.e == pytest.approx(expected_e)
    assert region.s == pytest.approx(expected_s)
    assert region.n == pytest.approx(expected_n)
    assert region.ewres == pytest.approx(expected_ewres)
    assert region.nsres == pytest.approx(expected_nsres)


def test_get_region_from_xarray_partial_custom_dim_names():
    """Test get_region_from_xarray with partial custom dimension names."""
    data = np.arange(12).reshape(3, 4)
    custom_x_coords = np.array([100, 200, 300, 400])
    custom_y_coords = np.array([50, 150, 250])
    da = xr.DataArray(
        data,
        coords={"x": custom_x_coords, "my_y": custom_y_coords},
        dims=["my_y", "x"],
    )
    dims_map = {
        "x": "x",
        "y": "my_y",
        **{k: default_dims[k] for k in default_dims if k not in ["x", "y"]},
    }
    region = get_region_from_xarray(da, dims_map)

    expected_ewres = 100.0
    expected_nsres = 100.0
    expected_w = custom_x_coords[0] - expected_ewres / 2
    expected_e = custom_x_coords[-1] + expected_ewres / 2
    expected_s = custom_y_coords[0] - expected_nsres / 2
    expected_n = custom_y_coords[-1] + expected_nsres / 2

    assert region.w == pytest.approx(expected_w)
    assert region.e == pytest.approx(expected_e)
    assert region.s == pytest.approx(expected_s)
    assert region.n == pytest.approx(expected_n)
    assert region.ewres == pytest.approx(expected_ewres)
    assert region.nsres == pytest.approx(expected_nsres)
