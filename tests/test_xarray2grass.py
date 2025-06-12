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

from pathlib import Path
import tempfile

import pandas as pd
import xarray as xr
import numpy as np
import pytest
from pyproj import CRS
import grass_session  # noqa: F401
import grass.script as gs


from xarray_grass import to_grass
from xarray_grass import GrassInterface  # For type hinting grass_i fixture


def create_sample_dataarray(
    dims_spec: dict,
    shape: tuple,
    crs_wkt: str,
    name: str = "test_data",
    use_latlon_dims: bool = False,
    time_dim_type: str = "absolute",  # "absolute", "relative", or "none"
    fill_value_generator=None,
) -> xr.DataArray:
    """
    Creates a sample xr.DataArray for testing.

    dims_spec: Dict mapping dimension names to their coordinate values.
               The keys should be the standard names ('time', 'z', 'y', 'x') and
               will be translated to 'latitude', 'longitude', 'y_3d', 'x_3d'
               based on use_latlon_dims and context (2D vs 3D).
               The order of keys in dims_spec must match the desired final
               dimension order of the DataArray and the order in `shape`.
               Example: {'time': pd.date_range(...), 'y': np.arange(...), ...}
    shape: Tuple defining the shape of the data array, matching the order of dims_spec.
    crs_wkt: WKT string for the crs_wkt attribute.
    name: Name of the xr.DataArray.
    use_latlon_dims: If True, spatial dimensions will be named 'latitude'/'longitude'
                     (and '_3d' versions). Otherwise 'x'/'y'.
    time_dim_type: If 'time' in dims_spec, specifies if time is 'absolute'
                   (datetime objects) or 'relative' (numeric).
    fill_value_generator: Function to generate data, e.g., lambda s: np.random.rand(*s).
                          If None, uses np.random.rand(*shape).
    """
    if fill_value_generator is None:
        data = np.random.rand(*shape)
    else:
        data = fill_value_generator(shape)

    coords = {}
    actual_dims_ordered = []  # This will store the final dimension names in the correct order

    # Define standard internal keys expected in dims_spec
    # These will be mapped to actual_dim_names based on context

    # Determine context for spatial dimension naming (2D or 3D)
    is_3d_spatial_context = "z" in dims_spec

    for dim_key in dims_spec.keys():  # Iterate in the order provided by dims_spec
        coord_values = dims_spec[dim_key]
        actual_dim_name = dim_key  # Default to key

        if dim_key == "time":
            if time_dim_type == "absolute":
                coords[actual_dim_name] = pd.to_datetime(coord_values)
            else:  # relative or none
                coords[actual_dim_name] = coord_values
        elif dim_key == "z":
            actual_dim_name = "z"  # Standard name
            coords[actual_dim_name] = coord_values
        elif dim_key == "y":
            if is_3d_spatial_context:
                actual_dim_name = "latitude_3d" if use_latlon_dims else "y_3d"
            else:  # 2D
                actual_dim_name = "latitude" if use_latlon_dims else "y"
            coords[actual_dim_name] = coord_values
        elif dim_key == "x":
            if is_3d_spatial_context:
                actual_dim_name = "longitude_3d" if use_latlon_dims else "x_3d"
            else:  # 2D
                actual_dim_name = "longitude" if use_latlon_dims else "x"
            coords[actual_dim_name] = coord_values
        else:  # Other dimensions (e.g., custom, non-spatial, non-temporal)
            coords[actual_dim_name] = coord_values

        actual_dims_ordered.append(actual_dim_name)

    if len(actual_dims_ordered) != len(shape):
        raise ValueError(
            f"Number of dimensions derived from dims_spec ({len(actual_dims_ordered)}) "
            f"does not match length of shape ({len(shape)}). "
            f"Ensure dims_spec keys are ordered correctly: {list(dims_spec.keys())} vs {actual_dims_ordered}"
        )

    da = xr.DataArray(
        data,
        coords=coords,
        dims=actual_dims_ordered,
        name=name,
    )
    da.attrs["crs_wkt"] = crs_wkt
    return da


def create_sample_dataset(
    data_vars_specs: dict,
    crs_wkt: str,
    global_use_latlon_dims: bool = False,
    global_time_dim_type: str = "absolute",
) -> xr.Dataset:
    """
    Creates a sample xr.Dataset for testing.

    data_vars_specs: Dict where keys are variable names and values are dicts
                     of parameters for create_sample_dataarray (dims_spec, shape, name,
                     optionally use_latlon_dims, time_dim_type, fill_value_generator).
                     The 'dims_spec' within each variable's spec should follow the
                     ordering and naming conventions for create_sample_dataarray.
    crs_wkt: WKT string for the crs_wkt attribute of the dataset and its DataArrays.
    global_use_latlon_dims: Default for use_latlon_dims if not in var_spec.
    global_time_dim_type: Default for time_dim_type if not in var_spec.
    """
    data_vars = {}
    for var_name, spec in data_vars_specs.items():
        # Ensure required keys are present in spec
        if not all(k in spec for k in ["dims_spec", "shape"]):
            raise ValueError(
                f"Variable spec for '{var_name}' is missing 'dims_spec' or 'shape'."
            )

        dims_spec = spec["dims_spec"]
        shape = spec["shape"]
        da_name = spec.get("name", var_name)
        use_latlon = spec.get("use_latlon_dims", global_use_latlon_dims)
        time_type = spec.get("time_dim_type", global_time_dim_type)
        fill_gen = spec.get("fill_value_generator", None)

        data_vars[var_name] = create_sample_dataarray(
            dims_spec=dims_spec,
            shape=shape,
            crs_wkt=crs_wkt,
            name=da_name,
            use_latlon_dims=use_latlon,
            time_dim_type=time_type,
            fill_value_generator=fill_gen,
        )
    ds = xr.Dataset(data_vars)
    ds.attrs["crs_wkt"] = crs_wkt
    return ds


@pytest.mark.usefixtures("grass_session_fixture")
class TestToGrassSuccess:
    @pytest.mark.parametrize("use_latlon_dims", [False, True])
    @pytest.mark.parametrize("mapset_is_path_obj", [False, True])
    def test_dataarray_2d_conversion(
        self,
        temp_gisdb,  # pytest.fixture from conftest.py
        grass_i: GrassInterface,  # pytest.fixture from conftest.py, type hinted
        use_latlon_dims: bool,
        mapset_is_path_obj: bool,
    ):
        """Test conversion of a 2D xr.DataArray to a GRASS Raster."""
        img_width = 10
        img_height = 12

        # Prepare sample DataArray
        if use_latlon_dims:
            # For lat/lon, use linspace to simulate geographic coordinates
            dims_spec = {
                "latitude": np.linspace(
                    50, 50.0 + (img_height - 1) * 0.01, img_height
                ),  # e.g. 50.00, 50.01 ...
                "longitude": np.linspace(
                    10, 10.0 + (img_width - 1) * 0.01, img_width
                ),  # e.g. 10.00, 10.01 ...
            }
            # The helper function `create_sample_dataarray` expects keys 'y', 'x' for spatial dims
            # and translates them based on `use_latlon_dims`.
            # So, the keys in dims_spec for helper should be 'y', 'x'.
            dims_spec_for_helper = {
                "y": dims_spec["latitude"],
                "x": dims_spec["longitude"],
            }
            expected_dims_order_in_da = ("latitude", "longitude")
        else:
            # For x/y, use arange to simulate projected or indexed coordinates
            dims_spec_for_helper = {
                "y": np.arange(img_height, dtype=float),  # 0.0, 1.0, ...
                "x": np.arange(img_width, dtype=float),  # 0.0, 1.0, ...
            }
            expected_dims_order_in_da = ("y", "x")

        shape = (img_height, img_width)  # (y, x) or (lat, lon)

        session_crs_wkt = grass_i.get_crs_wkt_str()

        sample_da = create_sample_dataarray(
            dims_spec=dims_spec_for_helper,  # Use the y,x keyed spec
            shape=shape,
            crs_wkt=session_crs_wkt,
            name="test_2d_raster",
            use_latlon_dims=use_latlon_dims,  # This controls final naming in DataArray
            fill_value_generator=lambda s: np.arange(s[0] * s[1])
            .reshape(s)
            .astype(float),
        )
        # Verify that the DataArray was created with the correct dimension names and order
        assert sample_da.dims == expected_dims_order_in_da, (
            f"DataArray dims {sample_da.dims} do not match expected {expected_dims_order_in_da}"
        )

        target_mapset_name = "test_2d_conv_mapset"
        mapset_path_obj = (
            Path(temp_gisdb.gisdb) / temp_gisdb.project / target_mapset_name
        )

        mapset_arg = mapset_path_obj if mapset_is_path_obj else str(mapset_path_obj)

        to_grass(
            dataset=sample_da,
            mapset=mapset_arg,
            create=True,
        )

        assert mapset_path_obj.exists() and mapset_path_obj.is_dir()

        current_mapsets_list = grass_i.get_accessible_mapsets()

        if target_mapset_name not in current_mapsets_list:
            gs.run_command(
                "g.mapsets", operation="add", mapset=target_mapset_name, quiet=True
            )

        grass_raster_name_full = f"{sample_da.name}@{target_mapset_name}"

        available_rasters = grass_i.list_raster(mapset=target_mapset_name)
        assert sample_da.name in available_rasters, (
            f"Raster '{sample_da.name}' not found in mapset '{target_mapset_name}'. Found: {available_rasters}"
        )

        info = gs.parse_command(
            "r.info", map=grass_raster_name_full, flags="g", quiet=True
        )

        assert int(info["rows"]) == img_height
        assert int(info["cols"]) == img_width

        # Check data statistics
        # Ensure data type of xarray DA is float for direct comparison with r.univar output
        sample_da_float = sample_da.astype(float)
        univar_stats = gs.parse_command(
            "r.univar", map=grass_raster_name_full, flags="g", quiet=True
        )

        # Convert univar stats from string to float, handling "none"
        for key, value in univar_stats.items():
            try:
                univar_stats[key] = float(value)
            except ValueError:  # Handles "none" or other non-float strings
                univar_stats[key] = np.nan

        assert np.isclose(
            univar_stats.get("min", np.nan),
            sample_da_float.min().item(),
            equal_nan=True,
        )
        assert np.isclose(
            univar_stats.get("max", np.nan),
            sample_da_float.max().item(),
            equal_nan=True,
        )
        assert np.isclose(
            univar_stats.get("mean", np.nan),
            sample_da_float.mean().item(),
            equal_nan=True,
        )

        # Clean up created mapset to keep tests isolated (optional, good practice)
        # First, change current mapset to PERMANENT to allow deletion of target_mapset_name
        # gs.run_command("g.mapset", mapset="PERMANENT", quiet=True)
        # gs.run_command("g.remove", flags="f", type="mapset", name=target_mapset_name, quiet=True)
        # The main gisdb is cleaned up by the temp_gisdb fixture.
        # Removing mapsets individually can be complex if GRASS holds locks.
        # For now, rely on temp_gisdb fixture for full cleanup.

    @pytest.mark.parametrize("use_latlon_dims", [False, True])
    @pytest.mark.parametrize("mapset_is_path_obj", [False, True])
    def test_dataarray_3d_conversion(
        self,
        temp_gisdb,
        grass_i: GrassInterface,
        use_latlon_dims: bool,
        mapset_is_path_obj: bool,
    ):
        """Test conversion of a 3D xr.DataArray to a GRASS 3D Raster."""
        img_depth = 5
        img_height = 8
        img_width = 6

        # Prepare sample DataArray
        # Helper `create_sample_dataarray` expects 'z', 'y', 'x' as keys in dims_spec
        # and translates y,x to y_3d/x_3d or latitude_3d/longitude_3d.
        if use_latlon_dims:
            dims_spec_for_helper = {
                "z": np.arange(img_depth, dtype=float),
                "y": np.linspace(50, 50.0 + (img_height - 1) * 0.01, img_height),
                "x": np.linspace(10, 10.0 + (img_width - 1) * 0.01, img_width),
            }
            expected_dims_order_in_da = ("z", "latitude_3d", "longitude_3d")
        else:
            dims_spec_for_helper = {
                "z": np.arange(img_depth, dtype=float),
                "y": np.arange(img_height, dtype=float),
                "x": np.arange(img_width, dtype=float),
            }
            expected_dims_order_in_da = ("z", "y_3d", "x_3d")

        shape = (img_depth, img_height, img_width)  # (z, y, x)

        session_crs_wkt = grass_i.get_crs_wkt_str()

        sample_da = create_sample_dataarray(
            dims_spec=dims_spec_for_helper,
            shape=shape,
            crs_wkt=session_crs_wkt,
            name="test_3d_raster",
            use_latlon_dims=use_latlon_dims,
            fill_value_generator=lambda s: np.arange(s[0] * s[1] * s[2])
            .reshape(s)
            .astype(float),
        )
        assert sample_da.dims == expected_dims_order_in_da, (
            f"DataArray dims {sample_da.dims} do not match expected {expected_dims_order_in_da}"
        )

        target_mapset_name = "test_3d_conv_mapset"
        mapset_path_obj = (
            Path(temp_gisdb.gisdb) / temp_gisdb.project / target_mapset_name
        )
        mapset_arg = mapset_path_obj if mapset_is_path_obj else str(mapset_path_obj)

        to_grass(
            dataset=sample_da,
            mapset=mapset_arg,
            create=True,
        )

        assert mapset_path_obj.exists() and mapset_path_obj.is_dir()

        current_mapsets_list = grass_i.get_accessible_mapsets()
        if target_mapset_name not in current_mapsets_list:
            gs.run_command(
                "g.mapsets", operation="add", mapset=target_mapset_name, quiet=True
            )

        grass_raster_name_full = f"{sample_da.name}@{target_mapset_name}"

        # Verification using list_grass_objects as per plan
        available_rasters_3d = grass_i.list_grass_objects(
            object_type="raster_3d", mapset_pattern=target_mapset_name
        )
        # list_grass_objects returns full names map@mapset
        assert grass_raster_name_full in available_rasters_3d, (
            f"3D Raster '{grass_raster_name_full}' not found. Found: {available_rasters_3d}"
        )

        info = gs.parse_command(
            "r3.info", map=grass_raster_name_full, flags="g", quiet=True
        )

        assert int(info["depth"]) == img_depth
        assert int(info["rows"]) == img_height
        assert int(info["cols"]) == img_width

        # Value checking for 3D rasters is more complex.
        # r3.univar exists but might be slow for large rasters in tests.
        # For now, metadata checks are primary.
        # Example: check min/max if r3.univar is fast enough or if a specific slice can be checked.
        # univar3_stats = gs.parse_command("r3.univar", map=grass_raster_name_full, quiet=True)
        # min_val_grass = float(univar3_stats['min']) # Key names might differ
        # max_val_grass = float(univar3_stats['max'])
        # assert np.isclose(min_val_grass, sample_da.min().item())
        # assert np.isclose(max_val_grass, sample_da.max().item())
        # This part is commented out as r3.univar might not output simple 'min=' 'max=' like r.univar -g
        # and full statistical comparison is deferred.

    @pytest.mark.parametrize("use_latlon_dims", [False, True])
    @pytest.mark.parametrize("mapset_is_path_obj", [False, True])
    @pytest.mark.parametrize("time_dim_type", ["absolute", "relative"])
    def test_dataarray_to_strds_conversion(
        self,
        temp_gisdb,
        grass_i: GrassInterface,
        use_latlon_dims: bool,
        mapset_is_path_obj: bool,
        time_dim_type: str,
    ):
        """Test conversion of a 3D xr.DataArray (time, space) to a GRASS STRDS."""
        num_times = 4
        img_height = 7
        img_width = 5

        time_coords = (
            None  # Initialize to avoid linter warning if conditions don't set it
        )
        if time_dim_type == "absolute":
            time_coords = pd.date_range(start="2023-01-01", periods=num_times, freq="D")
        elif time_dim_type == "relative":  # Ensure this is 'elif'
            time_coords = np.arange(1, num_times + 1)
        else:
            pytest.fail(f"Unsupported time_dim_type: {time_dim_type}")

        # Keys for dims_spec_for_helper should be 'time', 'y', 'x'
        if use_latlon_dims:
            dims_spec_for_helper = {
                "time": time_coords,
                "y": np.linspace(50, 50.0 + (img_height - 1) * 0.01, img_height),
                "x": np.linspace(10, 10.0 + (img_width - 1) * 0.01, img_width),
            }
            expected_dims_order_in_da = ("time", "latitude", "longitude")
        else:
            dims_spec_for_helper = {
                "time": time_coords,
                "y": np.arange(img_height, dtype=float),
                "x": np.arange(img_width, dtype=float),
            }
            expected_dims_order_in_da = ("time", "y", "x")

        shape = (num_times, img_height, img_width)
        session_crs_wkt = grass_i.get_crs_wkt_str()

        sample_da = create_sample_dataarray(
            dims_spec=dims_spec_for_helper,
            shape=shape,
            crs_wkt=session_crs_wkt,
            name="test_strds_raster",  # Base name for maps in STRDS
            use_latlon_dims=use_latlon_dims,
            time_dim_type=time_dim_type,
            fill_value_generator=lambda s: np.arange(s[0] * s[1] * s[2])
            .reshape(s)
            .astype(float),
        )
        assert sample_da.dims == expected_dims_order_in_da, (
            f"DataArray dims {sample_da.dims} do not match expected {expected_dims_order_in_da}"
        )

        target_mapset_name = "test_strds_conv_mapset"
        mapset_path_obj = (
            Path(temp_gisdb.gisdb) / temp_gisdb.project / target_mapset_name
        )
        mapset_arg = mapset_path_obj if mapset_is_path_obj else str(mapset_path_obj)

        to_grass(
            dataset=sample_da,
            mapset=mapset_arg,
            create=True,
        )

        assert mapset_path_obj.exists() and mapset_path_obj.is_dir()

        current_mapsets_list = grass_i.get_accessible_mapsets()
        if target_mapset_name not in current_mapsets_list:
            gs.run_command(
                "g.mapsets", operation="add", mapset=target_mapset_name, quiet=True
            )

        strds_name_full = f"{sample_da.name}@{target_mapset_name}"

        available_strds = grass_i.list_grass_objects(
            object_type="strds", mapset_pattern=target_mapset_name
        )
        assert strds_name_full in available_strds, (
            f"STRDS '{strds_name_full}' not found. Found: {available_strds}"
        )

        strds_maps_info_raw = gs.read_command(
            "t.rast.list", input=strds_name_full, columns="name,start_time", quiet=True
        )

        strds_map_names_in_grass = []
        if strds_maps_info_raw and strds_maps_info_raw.strip():
            for line in strds_maps_info_raw.strip().splitlines():
                parts = line.split()
                if parts:  # Ensure line is not empty after split
                    strds_map_names_in_grass.append(parts[0].split("@")[0])

        assert len(strds_map_names_in_grass) == num_times, (
            f"Expected {num_times} maps in STRDS '{strds_name_full}', found {len(strds_map_names_in_grass)}. Maps: {strds_map_names_in_grass}"
        )

        # Check statistics for the first and last time slices
        # This assumes t.rast.list output is ordered chronologically, which is typical.
        indices_to_check = [0, num_times - 1] if num_times > 0 else []

        for idx_in_da_time in indices_to_check:
            time_val = sample_da.time.values[idx_in_da_time]
            da_slice = sample_da.sel(time=time_val).astype(
                float
            )  # Ensure float for comparison

            # Assuming map order from t.rast.list corresponds to time order in DataArray
            map_to_check_name = strds_map_names_in_grass[idx_in_da_time]
            map_to_check_full = f"{map_to_check_name}@{target_mapset_name}"

            univar_stats = gs.parse_command(
                "r.univar", map=map_to_check_full, flags="g", quiet=True
            )
            for key, value in univar_stats.items():
                try:
                    univar_stats[key] = float(value)
                except ValueError:
                    univar_stats[key] = np.nan

            assert np.isclose(
                univar_stats.get("min", np.nan), da_slice.min().item(), equal_nan=True
            )
            assert np.isclose(
                univar_stats.get("max", np.nan), da_slice.max().item(), equal_nan=True
            )
            assert np.isclose(
                univar_stats.get("mean", np.nan), da_slice.mean().item(), equal_nan=True
            )

    @pytest.mark.parametrize("use_latlon_dims", [False, True])
    @pytest.mark.parametrize("mapset_is_path_obj", [False, True])
    @pytest.mark.parametrize("time_dim_type", ["absolute", "relative"])
    def test_dataarray_to_str3ds_conversion(
        self,
        temp_gisdb,
        grass_i: GrassInterface,
        use_latlon_dims: bool,
        mapset_is_path_obj: bool,
        time_dim_type: str,
    ):
        """Test conversion of a 4D xr.DataArray (time, z, space) to a GRASS STR3DS."""
        num_times = 3
        img_depth = 4
        img_height = 5
        img_width = 6

        time_coords = None
        if time_dim_type == "absolute":
            time_coords = pd.date_range(start="2024-01-01", periods=num_times, freq="H")
        elif time_dim_type == "relative":
            time_coords = np.arange(1, num_times + 1)
        else:
            pytest.fail(f"Unsupported time_dim_type: {time_dim_type}")

        # Keys for dims_spec_for_helper: 'time', 'z', 'y', 'x'
        if use_latlon_dims:
            dims_spec_for_helper = {
                "time": time_coords,
                "z": np.arange(img_depth, dtype=float),
                "y": np.linspace(50, 50.0 + (img_height - 1) * 0.001, img_height),
                "x": np.linspace(10, 10.0 + (img_width - 1) * 0.001, img_width),
            }
            expected_dims_order_in_da = ("time", "z", "latitude_3d", "longitude_3d")
        else:
            dims_spec_for_helper = {
                "time": time_coords,
                "z": np.arange(img_depth, dtype=float),
                "y": np.arange(img_height, dtype=float),
                "x": np.arange(img_width, dtype=float),
            }
            expected_dims_order_in_da = ("time", "z", "y_3d", "x_3d")

        shape = (num_times, img_depth, img_height, img_width)
        session_crs_wkt = grass_i.get_crs_wkt_str()

        sample_da = create_sample_dataarray(
            dims_spec=dims_spec_for_helper,
            shape=shape,
            crs_wkt=session_crs_wkt,
            name="test_str3ds_vol",  # Base name for volumes in STR3DS
            use_latlon_dims=use_latlon_dims,
            time_dim_type=time_dim_type,
            fill_value_generator=lambda s: np.arange(s[0] * s[1] * s[2] * s[3])
            .reshape(s)
            .astype(float),
        )
        assert sample_da.dims == expected_dims_order_in_da, (
            f"DataArray dims {sample_da.dims} do not match expected {expected_dims_order_in_da}"
        )

        target_mapset_name = "test_str3ds_conv_mapset"
        mapset_path_obj = (
            Path(temp_gisdb.gisdb) / temp_gisdb.project / target_mapset_name
        )
        mapset_arg = mapset_path_obj if mapset_is_path_obj else str(mapset_path_obj)

        to_grass(
            dataset=sample_da,
            mapset=mapset_arg,
            create=True,
        )

        assert mapset_path_obj.exists() and mapset_path_obj.is_dir()

        current_mapsets_list = grass_i.get_accessible_mapsets()
        if target_mapset_name not in current_mapsets_list:
            gs.run_command(
                "g.mapsets", operation="add", mapset=target_mapset_name, quiet=True
            )

        str3ds_name_full = f"{sample_da.name}@{target_mapset_name}"

        available_str3ds = grass_i.list_grass_objects(
            object_type="str3ds", mapset_pattern=target_mapset_name
        )
        assert str3ds_name_full in available_str3ds, (
            f"STR3DS '{str3ds_name_full}' not found. Found: {available_str3ds}"
        )

        str3ds_maps_info_raw = gs.read_command(
            "t.rast3d.list",
            input=str3ds_name_full,
            columns="name,start_time",
            quiet=True,
        )

        str3ds_map_names_in_grass = []
        if str3ds_maps_info_raw and str3ds_maps_info_raw.strip():
            for line in str3ds_maps_info_raw.strip().splitlines():
                parts = line.split()
                if parts:
                    str3ds_map_names_in_grass.append(parts[0].split("@")[0])

        assert len(str3ds_map_names_in_grass) == num_times, (
            f"Expected {num_times} maps in STR3DS '{str3ds_name_full}', found {len(str3ds_map_names_in_grass)}. Maps: {str3ds_map_names_in_grass}"
        )

        # Check metadata for the first and last time slices' 3D rasters
        indices_to_check = [0, num_times - 1] if num_times > 0 else []

        for idx_in_da_time in indices_to_check:
            # da_slice_3d = sample_da.sel(time=sample_da.time.values[idx_in_da_time]) # This is still 3D

            map_to_check_name = str3ds_map_names_in_grass[idx_in_da_time]
            map_to_check_full = f"{map_to_check_name}@{target_mapset_name}"

            info = gs.parse_command(
                "r3.info", map=map_to_check_full, flags="g", quiet=True
            )
            assert int(info["depth"]) == img_depth
            assert int(info["rows"]) == img_height
            assert int(info["cols"]) == img_width
            # Further statistical checks on r3.univar could be added if deemed necessary and performant.

    @pytest.mark.parametrize("mapset_is_path_obj", [False, True])
    def test_dataset_conversion_mixed_types(
        self,
        temp_gisdb,
        grass_i: GrassInterface,
        mapset_is_path_obj: bool,
    ):
        """Test conversion of an xr.Dataset with mixed DataArray types."""
        session_crs_wkt = grass_i.get_crs_wkt_str()
        target_mapset_name = "test_dataset_conv_mapset"
        mapset_path_obj = (
            Path(temp_gisdb.gisdb) / temp_gisdb.project / target_mapset_name
        )
        mapset_arg = mapset_path_obj if mapset_is_path_obj else str(mapset_path_obj)

        # Define specs for various DataArrays
        # Using simple x/y and relative time for brevity in this combined test
        # Individual type tests cover lat/lon and absolute time variations.

        # 2D Raster Spec
        da_2d_spec = {
            "dims_spec": {"y": np.arange(5.0), "x": np.arange(3.0)},
            "shape": (5, 3),
            "name": "ds_raster2d",
        }

        # 3D Raster Spec
        da_3d_spec = {
            "dims_spec": {
                "z": np.arange(2.0),
                "y": np.arange(4.0),
                "x": np.arange(3.0),
            },
            "shape": (2, 4, 3),
            "name": "ds_raster3d",
        }

        # STRDS Spec
        strds_spec = {
            "dims_spec": {
                "time": np.arange(1, 3),
                "y": np.arange(3.0),
                "x": np.arange(2.0),
            },
            "shape": (2, 3, 2),
            "name": "ds_strds",
        }

        # STR3DS Spec
        str3ds_spec = {
            "dims_spec": {
                "time": np.arange(1, 3),
                "z": np.arange(2.0),
                "y": np.arange(3.0),
                "x": np.arange(2.0),
            },
            "shape": (2, 2, 3, 2),
            "name": "ds_str3ds",
        }

        dataset_specs = {
            "raster2d_var": da_2d_spec,
            "raster3d_var": da_3d_spec,
            "strds_var": strds_spec,
            "str3ds_var": str3ds_spec,
        }

        sample_ds = create_sample_dataset(
            data_vars_specs=dataset_specs,
            crs_wkt=session_crs_wkt,
            global_use_latlon_dims=False,  # Use x,y for this test
            global_time_dim_type="relative",  # Use relative time
        )

        to_grass(
            dataset=sample_ds,
            mapset=mapset_arg,
            create=True,
        )

        assert mapset_path_obj.exists() and mapset_path_obj.is_dir()
        current_mapsets_list = grass_i.get_accessible_mapsets()
        if target_mapset_name not in current_mapsets_list:
            gs.run_command(
                "g.mapsets", operation="add", mapset=target_mapset_name, quiet=True
            )

        # Verification for each type
        # 2D Raster
        raster2d_name_full = f"{da_2d_spec['name']}@{target_mapset_name}"
        available_rasters = grass_i.list_raster(mapset=target_mapset_name)
        assert da_2d_spec["name"] in available_rasters
        info2d = gs.parse_command(
            "r.info", map=raster2d_name_full, flags="g", quiet=True
        )
        assert int(info2d["rows"]) == da_2d_spec["shape"][0]
        assert int(info2d["cols"]) == da_2d_spec["shape"][1]

        # 3D Raster
        raster3d_name_full = f"{da_3d_spec['name']}@{target_mapset_name}"
        available_rasters_3d = grass_i.list_grass_objects(
            object_type="raster_3d", mapset_pattern=target_mapset_name
        )
        assert raster3d_name_full in available_rasters_3d
        info3d = gs.parse_command(
            "r3.info", map=raster3d_name_full, flags="g", quiet=True
        )
        assert int(info3d["depth"]) == da_3d_spec["shape"][0]
        assert int(info3d["rows"]) == da_3d_spec["shape"][1]
        assert int(info3d["cols"]) == da_3d_spec["shape"][2]

        # STRDS
        strds_name_full = f"{strds_spec['name']}@{target_mapset_name}"
        available_strds = grass_i.list_grass_objects(
            object_type="strds", mapset_pattern=target_mapset_name
        )
        assert strds_name_full in available_strds
        num_strds_maps = len(grass_i.list_maps_in_strds())
        assert num_strds_maps == strds_spec["shape"][0]  # Number of time steps

        # STR3DS
        str3ds_name_full = f"{str3ds_spec['name']}@{target_mapset_name}"
        available_str3ds = grass_i.list_grass_objects(
            object_type="str3ds", mapset_pattern=target_mapset_name
        )
        assert str3ds_name_full in available_str3ds
        num_str3ds_maps = len(grass_i.list_maps_in_str3ds())
        assert num_str3ds_maps == str3ds_spec["shape"][0]  # Number of time steps

    def test_mapset_creation_true(self, temp_gisdb, grass_i: GrassInterface):
        """Test mapset creation when create=True."""
        session_crs_wkt = grass_i.get_crs_wkt_str()
        new_mapset_name = "mapset_created_by_test"
        # Construct full path for mapset creation and checking
        mapset_path = Path(temp_gisdb.gisdb) / temp_gisdb.project / new_mapset_name

        # Ensure mapset does not exist initially for a clean test
        if mapset_path.exists():
            try:
                # Attempt to remove it if it's a leftover.
                # Switch to PERMANENT mapset first if current is the one to be deleted.
                current_active_mapset = gs.read_command(
                    "g.mapset", flags="p", mapset="$"
                ).strip()
                if current_active_mapset == new_mapset_name:
                    gs.run_command("g.mapset", mapset="PERMANENT", quiet=True)

                # Try removing with GRASS command first
                gs.run_command(
                    "g.remove",
                    type="mapset",
                    name=new_mapset_name,
                    flags="f",
                    quiet=True,
                )

                # If still exists (e.g. GRASS couldn't remove non-empty), try rmtree
                if mapset_path.exists():
                    import shutil

                    shutil.rmtree(mapset_path)
            except Exception as e:
                # If cleanup fails, skip the test as its premise (mapset doesn't exist) is not met.
                pytest.skip(
                    f"Could not clean up pre-existing mapset {new_mapset_name} for test: {e}"
                )

        assert not mapset_path.exists(), (
            f"Mapset {new_mapset_name} still exists before test run."
        )

        sample_da = create_sample_dataarray(
            dims_spec={"y": np.arange(2.0), "x": np.arange(2.0)},
            shape=(2, 2),
            crs_wkt=session_crs_wkt,
            name="data_for_mapset_creation",
        )

        # Pass the string path to the mapset directory
        to_grass(dataset=sample_da, mapset=str(mapset_path), create=True)

        assert mapset_path.exists() and mapset_path.is_dir(), (
            f"Mapset directory {mapset_path} was not created."
        )

        # Add mapset to search path for verification, to_grass might not do this.
        current_mapsets_list = grass_i.get_accessible_mapsets()
        if new_mapset_name not in current_mapsets_list:
            gs.run_command(
                "g.mapsets", operation="add", mapset=new_mapset_name, quiet=True
            )

        available_rasters = grass_i.list_raster(mapset=new_mapset_name)
        assert sample_da.name in available_rasters, (
            f"Raster '{sample_da.name}' not found in newly created mapset '{new_mapset_name}'."
        )

    def test_mapset_creation_false_existing_mapset(
        self, temp_gisdb, grass_i: GrassInterface
    ):
        """Test using an existing mapset when create=False."""
        session_crs_wkt = grass_i.get_crs_wkt_str()
        existing_mapset_name = "existing_mapset_for_test"
        mapset_path = Path(temp_gisdb.gisdb) / temp_gisdb.project / existing_mapset_name

        # Create the mapset manually first
        if mapset_path.exists():  # Cleanup if exists
            try:
                current_active_mapset = gs.read_command(
                    "g.mapset", flags="p", mapset="$"
                ).strip()
                if current_active_mapset == existing_mapset_name:
                    gs.run_command("g.mapset", mapset="PERMANENT", quiet=True)
                gs.run_command(
                    "g.remove",
                    type="mapset",
                    name=existing_mapset_name,
                    flags="f",
                    quiet=True,
                )
                if mapset_path.exists():
                    import shutil

                    shutil.rmtree(mapset_path)
            except Exception as e:
                pytest.skip(
                    f"Could not clean up pre-existing mapset {existing_mapset_name} for test: {e}"
                )

        gs.run_command(
            "g.mapset",
            flags="c",
            mapset=existing_mapset_name,
            location=temp_gisdb.project,
            gisdbase=temp_gisdb.gisdb,
            quiet=True,
        )
        assert mapset_path.exists() and mapset_path.is_dir(), (
            f"Test setup failed: Mapset {existing_mapset_name} could not be created."
        )

        # Add to search path for subsequent operations if not already there
        current_mapsets_list = grass_i.get_accessible_mapsets()
        if existing_mapset_name not in current_mapsets_list:
            gs.run_command(
                "g.mapsets", operation="add", mapset=existing_mapset_name, quiet=True
            )

        sample_da = create_sample_dataarray(
            dims_spec={"y": np.arange(3.0), "x": np.arange(3.0)},
            shape=(3, 3),
            crs_wkt=session_crs_wkt,
            name="data_for_existing_mapset",
        )

        # Pass the string path to the mapset directory
        to_grass(dataset=sample_da, mapset=str(mapset_path), create=False)

        available_rasters = grass_i.list_raster(mapset=existing_mapset_name)
        assert sample_da.name in available_rasters, (
            f"Raster '{sample_da.name}' not found in existing mapset '{existing_mapset_name}'."
        )

    @pytest.mark.parametrize("mapset_as_path_object", [True, False])
    def test_mapset_argument_type(
        self, temp_gisdb, grass_i: GrassInterface, mapset_as_path_object: bool
    ):
        """Test that 'mapset' argument accepts both str and Path objects."""
        session_crs_wkt = grass_i.get_crs_wkt_str()
        mapset_name = "mapset_arg_type_test"
        mapset_path_obj = Path(temp_gisdb.gisdb) / temp_gisdb.project / mapset_name

        if mapset_path_obj.exists():  # Cleanup
            try:
                current_active_mapset = gs.read_command(
                    "g.mapset", flags="p", mapset="$"
                ).strip()
                if current_active_mapset == mapset_name:
                    gs.run_command("g.mapset", mapset="PERMANENT", quiet=True)
                gs.run_command(
                    "g.remove", type="mapset", name=mapset_name, flags="f", quiet=True
                )
                if mapset_path_obj.exists():
                    import shutil

                    shutil.rmtree(mapset_path_obj)
            except Exception as e:
                pytest.skip(f"Cleanup failed for {mapset_name}: {e}")

        assert not mapset_path_obj.exists()

        mapset_arg = mapset_path_obj if mapset_as_path_object else str(mapset_path_obj)

        sample_da = create_sample_dataarray(
            dims_spec={"y": np.arange(2.0), "x": np.arange(2.0)},
            shape=(2, 2),
            crs_wkt=session_crs_wkt,
            name="data_for_mapset_arg_type",
        )

        to_grass(dataset=sample_da, mapset=mapset_arg, create=True)

        assert mapset_path_obj.exists() and mapset_path_obj.is_dir()
        current_mapsets_list = grass_i.get_accessible_mapsets()
        if mapset_name not in current_mapsets_list:
            gs.run_command("g.mapsets", operation="add", mapset=mapset_name, quiet=True)
        available_rasters = grass_i.list_raster(mapset=mapset_name)
        assert sample_da.name in available_rasters

    @pytest.mark.parametrize(
        "use_latlon_dims_in_da, dims_param, expected_grass_dims_match_da_standard",
        [
            (False, None, True),  # DA uses y,x; no dims mapping -> GRASS uses y,x
            (
                True,
                None,
                True,
            ),  # DA uses lat,lon; no dims mapping -> GRASS uses lat,lon
            (False, {"y": "northing", "x": "easting"}, False),  # DA y,x; map y->N, x->E
            (
                True,
                {"latitude": "lat_custom", "longitude": "lon_custom"},
                False,
            ),  # DA lat,lon; map to custom
            (
                False,
                {"y": "custom_y"},
                True,
            ),  # Partial map: DA y,x; map y->custom_y, x should remain x
        ],
    )
    def test_dims_mapping(
        self,
        temp_gisdb,
        grass_i: GrassInterface,
        use_latlon_dims_in_da: bool,
        dims_param: dict,
        expected_grass_dims_match_da_standard: bool,
    ):
        """Test 'dims' mapping functionality."""
        session_crs_wkt = grass_i.get_crs_wkt_str()
        mapset_name = "dims_mapping_test_mapset"
        mapset_path = Path(temp_gisdb.gisdb) / temp_gisdb.project / mapset_name

        da_name = "dims_test_raster"
        img_height, img_width = 3, 2

        if use_latlon_dims_in_da:
            da_dims_spec = {
                "y": np.linspace(50, 50.01, img_height),  # 'y' key for helper
                "x": np.linspace(10, 10.01, img_width),  # 'x' key for helper
            }
            # expected_da_actual_dims = ("latitude", "longitude")
        else:
            da_dims_spec = {
                "y": np.arange(img_height, dtype=float),
                "x": np.arange(img_width, dtype=float),
            }
            # expected_da_actual_dims = ("y", "x")

        sample_da = create_sample_dataarray(
            dims_spec=da_dims_spec,
            shape=(img_height, img_width),
            crs_wkt=session_crs_wkt,
            name=da_name,
            use_latlon_dims=use_latlon_dims_in_da,  # This sets the actual dim names in DA
        )

        to_grass(
            dataset=sample_da, mapset=str(mapset_path), create=True, dims=dims_param
        )

        assert mapset_path.exists() and mapset_path.is_dir()
        current_mapsets_list = grass_i.get_accessible_mapsets()
        if mapset_name not in current_mapsets_list:
            gs.run_command("g.mapsets", operation="add", mapset=mapset_name, quiet=True)

        available_rasters = grass_i.list_raster(mapset=mapset_name)
        assert da_name in available_rasters

        # Verification of dims mapping is primarily by successful import.
        # The `to_grass` function internally uses these mappings to find the
        # coordinate data within the xarray object. If it imports successfully
        # with the given `dims` mapping, it implies the mapping was understood.
        # More detailed checks would involve inspecting GRASS metadata if it stored
        # original dimension names, which it typically doesn't directly.
        # So, successful creation is the main check here.
        # If `expected_grass_dims_match_da_standard` is True, it means we expect
        # GRASS to have used its standard interpretation of the DA's *original* standard dims.
        # If False, it means a custom mapping was applied, and GRASS still made a valid map.
        info = gs.parse_command(
            "r.info", map=f"{da_name}@{mapset_name}", flags="g", quiet=True
        )
        assert int(info["rows"]) == img_height
        assert int(info["cols"]) == img_width
        # No direct way to check if GRASS "knows" about "northing" vs "y" from r.info,
        # the key is that the data from the correct DA dimension was used.


@pytest.mark.usefixtures("grass_session_fixture")
class TestToGrassErrorHandling:
    def test_missing_crs_wkt_attribute(self, temp_gisdb, grass_i: GrassInterface):
        """Test error handling when input xarray object is missing 'crs_wkt' attribute."""
        mapset_name = "error_missing_crs_mapset"
        mapset_path = Path(temp_gisdb.gisdb) / temp_gisdb.project / mapset_name

        # Create a DataArray without crs_wkt
        # The helper function always adds it, so we create one manually here.
        sample_da_no_crs = xr.DataArray(
            np.random.rand(2, 2),
            coords={"y": [0, 1], "x": [0, 1]},
            dims=("y", "x"),
            name="data_no_crs",
        )
        # Intentionally do not set sample_da_no_crs.attrs["crs_wkt"]

        with pytest.raises((AttributeError, ValueError), match=r"CRS mismatch"):
            to_grass(dataset=sample_da_no_crs, mapset=str(mapset_path), create=True)

    def test_incompatible_crs_wkt(self, temp_gisdb, grass_i: GrassInterface):
        """Test error handling with an incompatible 'crs_wkt' attribute."""
        mapset_name = "error_incompatible_crs_mapset"
        mapset_path = Path(temp_gisdb.gisdb) / temp_gisdb.project / mapset_name

        session_crs_wkt = grass_i.get_crs_wkt_str()

        # Create an incompatible CRS WKT string
        incompatible_crs = CRS.from_epsg(7030)
        if CRS.from_wkt(session_crs_wkt).equals(incompatible_crs):
            # If by chance the session CRS is compatible, pick another one
            incompatible_crs = CRS.from_epsg(7008)
        incompatible_crs_wkt = incompatible_crs.to_wkt()

        sample_da = create_sample_dataarray(
            dims_spec={"y": np.arange(2.0), "x": np.arange(2.0)},
            shape=(2, 2),
            crs_wkt=incompatible_crs_wkt,  # Set the incompatible CRS
            name="data_incompatible_crs",
        )

        with pytest.raises(
            ValueError,
            match=r"(CRS does not match|Incompatible coordinate systems|Projection mismatch)",
        ):
            to_grass(dataset=sample_da, mapset=str(mapset_path), create=True)

    def test_invalid_mapset_path_non_existent_parent(
        self, temp_gisdb, grass_i: GrassInterface
    ):
        """Test error with mapset path having a non-existent parent directory."""
        session_crs_wkt = grass_i.get_crs_wkt_str()
        # Path to a non-existent directory, then the mapset
        mapset_path = Path(temp_gisdb.project) / "non_existent_parent_dir" / "my_mapset"

        sample_da = create_sample_dataarray(
            dims_spec={"y": np.arange(2.0), "x": np.arange(2.0)},
            shape=(2, 2),
            crs_wkt=session_crs_wkt,
            name="data_invalid_parent",
        )

        with pytest.raises(
            ValueError,
            match=r"(Invalid mapset path|Cannot create mapset|Parent directory.*does not exist)",
        ):
            to_grass(dataset=sample_da, mapset=str(mapset_path), create=True)

    def test_invalid_mapset_path_is_file(self, temp_gisdb, grass_i: GrassInterface):
        """Test error with mapset path being an existing file."""
        session_crs_wkt = grass_i.get_crs_wkt_str()

        # Create an empty file where the mapset directory would be
        file_as_mapset_path = Path(temp_gisdb.project) / "file_instead_of_mapset"
        with open(file_as_mapset_path, "w") as f:
            f.write("This is a file.")

        assert file_as_mapset_path.is_file()

        sample_da = create_sample_dataarray(
            dims_spec={"y": np.arange(2.0), "x": np.arange(2.0)},
            shape=(2, 2),
            crs_wkt=session_crs_wkt,
            name="data_mapset_is_file",
        )

        with pytest.raises(
            ValueError, match=r"(Invalid mapset path|Path.*is a file|Not a directory)"
        ):
            to_grass(dataset=sample_da, mapset=str(file_as_mapset_path), create=True)

        file_as_mapset_path.unlink()  # Clean up the created file

    def test_parent_dir_not_grass_location(self, grass_i: GrassInterface):
        """Test error when parent of mapset is not a GRASS Location (create=True)."""
        session_crs_wkt = grass_i.get_crs_wkt_str()

        with tempfile.TemporaryDirectory(
            prefix="not_a_grass_loc_"
        ) as tmp_non_grass_dir:
            mapset_path_in_non_grass_loc = Path(tmp_non_grass_dir) / "my_mapset_here"

            sample_da = create_sample_dataarray(
                dims_spec={"y": np.arange(2.0), "x": np.arange(2.0)},
                shape=(2, 2),
                crs_wkt=session_crs_wkt,
                name="data_non_grass_parent",
            )

            # This relies on to_grass checking if the parent is a GRASS location.
            # The exact error message might vary based on implementation.
            with pytest.raises(
                ValueError,
                match=r"(not a valid GRASS project|Parent directory.*not a GRASS location|Invalid GIS database)",
            ):
                to_grass(
                    dataset=sample_da,
                    mapset=str(mapset_path_in_non_grass_loc),
                    create=True,
                )

    def test_create_false_mapset_not_exists(self, temp_gisdb, grass_i: GrassInterface):
        """Test error when create=False and mapset does not exist."""
        session_crs_wkt = grass_i.get_crs_wkt_str()
        non_existent_mapset_path = (
            Path(temp_gisdb.project) / "mapset_does_not_exist_at_all"
        )

        assert not non_existent_mapset_path.exists()  # Ensure it really doesn't exist

        sample_da = create_sample_dataarray(
            dims_spec={"y": np.arange(2.0), "x": np.arange(2.0)},
            shape=(2, 2),
            crs_wkt=session_crs_wkt,
            name="data_create_false_no_mapset",
        )

        with pytest.raises(
            ValueError,
            match=r"(Mapset.*does not exist and create is False|Target mapset not found)",
        ):
            to_grass(
                dataset=sample_da, mapset=str(non_existent_mapset_path), create=False
            )

    def test_mapset_not_accessible_simplified(self, grass_i: GrassInterface):
        """Test simplified 'mapset not accessible' by providing a syntactically valid but unrelated path."""
        session_crs_wkt = grass_i.get_crs_wkt_str()

        # A path that is unlikely to be a GRASS mapset accessible to the current session
        # This doesn't create a separate GRASS session, just uses a bogus path.
        # The function should ideally detect this isn't a valid mapset within the current GISDB.
        unrelated_path = "/tmp/some_completely_random_unrelated_path_for_mapset_test"
        # Ensure it doesn't exist, or the error might be different (e.g. "is a file")
        if Path(unrelated_path).exists():
            try:
                if Path(unrelated_path).is_dir():
                    import shutil

                    shutil.rmtree(unrelated_path)
                else:
                    Path(unrelated_path).unlink()
            except OSError:
                pytest.skip(f"Could not clean up unrelated_path: {unrelated_path}")

        sample_da = create_sample_dataarray(
            dims_spec={"y": np.arange(2.0), "x": np.arange(2.0)},
            shape=(2, 2),
            crs_wkt=session_crs_wkt,
            name="data_unrelated_mapset_path",
        )

        # The expected error could be about invalid path, not a GRASS mapset, or not in current GISDB.
        with pytest.raises(
            ValueError,
            match=r"(Invalid mapset path|not a GRASS mapset|not accessible from current GRASS session|not part of a GRASS GIS database)",
        ):
            to_grass(dataset=sample_da, mapset=unrelated_path, create=True)
            to_grass(
                dataset=sample_da, mapset=unrelated_path, create=False
            )  # Also test with create=False


@pytest.mark.usefixtures("grass_session_fixture")
class TestToGrassInputValidation:
    def test_invalid_dataset_type(self, temp_gisdb, grass_i: GrassInterface):
        """Test error handling for invalid 'dataset' parameter type.
        That a first try. Let's see how it goes considering that the tested code uses duck typing."""
        mapset_path = Path(temp_gisdb.project) / "input_validation_mapset"

        # Ensure mapset exists for other params to be valid initially
        if not mapset_path.exists():
            gs.run_command(
                "g.mapset",
                flags="c",
                mapset=mapset_path.name,
                location=temp_gisdb.project,
                gisdbase=temp_gisdb.gisdb,
                quiet=True,
            )

        invalid_datasets = [123, "a string", [1, 2, 3], {"data": np.array([1])}, None]
        for invalid_ds in invalid_datasets:
            with pytest.raises(
                TypeError,
                match=r"(Dataset must be an xarray.Dataset or xarray.DataArray|is not an xarray.Dataset or xarray.DataArray)",
            ):
                to_grass(dataset=invalid_ds, mapset=str(mapset_path), create=False)

    def test_invalid_dims_parameter_type(self, temp_gisdb, grass_i: GrassInterface):
        """Test error handling for invalid 'dims' parameter type or content."""
        session_crs_wkt = grass_i.get_crs_wkt_str()
        mapset_path = Path(temp_gisdb.project) / "dims_validation_mapset"
        if not mapset_path.exists():
            gs.run_command(
                "g.mapset",
                flags="c",
                mapset=mapset_path.name,
                location=temp_gisdb.project,
                gisdbase=temp_gisdb.gisdb,
                quiet=True,
            )

        sample_da = create_sample_dataarray(
            dims_spec={"y": np.arange(2.0), "x": np.arange(2.0)},
            shape=(2, 2),
            crs_wkt=session_crs_wkt,
            name="data_for_dims_validation",
        )

        invalid_dims_params = [
            "not_a_dict",
            123,
            ["y", "x"],
            {"y": 123, "x": "longitude"},  # Value not a string
            {1: "y", 2: "x"},  # Key not a string
        ]
        for invalid_dims in invalid_dims_params:
            # Need to refind the error message match
            with pytest.raises(
                (TypeError, ValueError),
                match=r"(dims parameter must be a mapping|Invalid value in dims mapping|Invalid key in dims mapping)",
            ):
                to_grass(
                    dataset=sample_da,
                    mapset=str(mapset_path),
                    dims=invalid_dims,
                    create=False,
                )

    def test_invalid_mapset_parameter_type(self, temp_gisdb, grass_i: GrassInterface):
        """Test error handling for invalid 'mapset' parameter type."""
        session_crs_wkt = grass_i.get_crs_wkt_str()
        sample_da = create_sample_dataarray(
            dims_spec={"y": np.arange(2.0), "x": np.arange(2.0)},
            shape=(2, 2),
            crs_wkt=session_crs_wkt,
            name="data_for_mapset_type_validation",
        )

        invalid_mapset_params = [
            123,
            None,
            ["path", "to", "mapset"],
            {"path": "mapset_dir"},
        ]
        for invalid_mapset in invalid_mapset_params:
            with pytest.raises(
                TypeError,
                match=r"(mapset parameter must be a string or a Path|Invalid mapset type)",
            ):
                to_grass(dataset=sample_da, mapset=invalid_mapset, create=True)
