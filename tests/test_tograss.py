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

import pandas as pd
import numpy as np
import pytest
import grass_session  # noqa: F401
import grass.script as gs
from grass.exceptions import CalledModuleError


from xarray_grass import to_grass
from xarray_grass import GrassInterface
from .conftest import create_sample_dataarray, create_sample_dataset


@pytest.mark.usefixtures("grass_session_fixture", "grass_test_region")
class TestToGrassSuccess:
    @pytest.mark.parametrize("mapset_is_path_obj", [False, True])
    def test_dataarray_2d_conversion(
        self,
        temp_gisdb,
        grass_i: GrassInterface,
        mapset_is_path_obj: bool,
    ):
        """Test conversion of a 2D xr.DataArray to a GRASS Raster."""
        # If a mask is present, the stats comparison will not be accurate.
        assert not grass_i.has_mask()
        img_width = 10
        img_height = 12

        dims_spec_for_helper = {
            "y": np.arange(img_height, dtype=float),  # 0.0, 1.0, ...
            "x": np.arange(img_width, dtype=float),  # 0.0, 1.0, ...
        }
        expected_dims_order_in_da = ("y", "x")

        shape = (img_height, img_width)  # (y, x)

        session_crs_wkt = grass_i.get_crs_wkt_str()

        sample_da = create_sample_dataarray(
            dims_spec=dims_spec_for_helper,
            shape=shape,
            crs_wkt=session_crs_wkt,
            name="test_2d_raster",
            fill_value_generator=lambda s: np.arange(s[0] * s[1])
            .reshape(s)
            .astype(float),
        )
        # Verify that the DataArray was created with the correct dimension names and order
        assert sample_da.dims == expected_dims_order_in_da, (
            f"DataArray dims {sample_da.dims} do not match expected {expected_dims_order_in_da}"
        )

        target_mapset_name = temp_gisdb.mapset  # Use PERMANENT mapset
        mapset_path_obj = (
            Path(temp_gisdb.gisdb) / temp_gisdb.project / target_mapset_name
        )
        mapset_arg = mapset_path_obj if mapset_is_path_obj else str(mapset_path_obj)
        grass_raster_name_full = (
            f"{sample_da.name}@{target_mapset_name}"  # Moved and defined
        )

        try:
            to_grass(
                dataset=sample_da,
                mapset=mapset_arg,
                create=False,
            )
            available_rasters = grass_i.list_raster(mapset=target_mapset_name)
            assert grass_raster_name_full in available_rasters, (
                f"Raster '{grass_raster_name_full}' not found in mapset '{target_mapset_name}'. Found: {available_rasters}"
            )
            info = gs.parse_command(
                "r.info", map=grass_raster_name_full, flags="g", quiet=True
            )
            assert int(info["rows"]) == img_height
            assert int(info["cols"]) == img_width
            # Store current region (which is the original/default one from the test session)
            original_region_for_assertions = grass_i.get_region()
            try:
                gs.run_command(
                    "g.region", flags="o", raster=grass_raster_name_full, quiet=True
                )
                # Check data statistics
                # Ensure data type of xarray DA is float for direct comparison with r.univar output
                sample_da_float = sample_da.astype(float)
                univar_stats = gs.parse_command(
                    "r.univar", map=grass_raster_name_full, flags="g", quiet=True
                )
            finally:
                # Restore the original region for the test session
                grass_i.set_region(original_region_for_assertions)
            # Convert univar stats from string to float, handling "none"
            for key, value in univar_stats.items():
                try:
                    univar_stats[key] = float(value)
                except ValueError:
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

        finally:
            gs.run_command(
                "g.remove",
                flags="f",
                type="raster",
                name=grass_raster_name_full,
                quiet=True,
            )

    @pytest.mark.parametrize("mapset_is_path_obj", [False, True])
    def test_dataarray_3d_conversion(
        self,
        temp_gisdb,
        grass_i: GrassInterface,
        mapset_is_path_obj: bool,
    ):
        """Test conversion of a 3D xr.DataArray to a GRASS 3D Raster."""
        img_depth = 5
        img_height = 8
        img_width = 6

        # Use coordinates within valid range for NAD83(HARN) / North Carolina
        res3 = 1000
        dims_spec_for_helper = {
            "z": np.arange(img_depth, dtype=float),
            "y": np.linspace(220000, 220000 + (img_height - 1) * res3, img_height),
            "x": np.linspace(630000, 630000 + (img_width - 1) * res3, img_width),
        }
        expected_dims_order_in_da = ("z", "y_3d", "x_3d")

        shape = (img_depth, img_height, img_width)  # (z, y, x)
        session_crs_wkt = grass_i.get_crs_wkt_str()
        sample_da = create_sample_dataarray(
            dims_spec=dims_spec_for_helper,
            shape=shape,
            crs_wkt=session_crs_wkt,
            name="test_3d_raster",
            fill_value_generator=lambda s: np.arange(s[0] * s[1] * s[2])
            .reshape(s)
            .astype(float),
        )
        assert sample_da.dims == expected_dims_order_in_da, (
            f"DataArray dims {sample_da.dims} do not match expected {expected_dims_order_in_da}"
        )

        target_mapset_name = temp_gisdb.mapset
        mapset_path_obj = (
            Path(temp_gisdb.gisdb) / temp_gisdb.project / target_mapset_name
        )
        mapset_arg = mapset_path_obj if mapset_is_path_obj else str(mapset_path_obj)

        # Try statement for file cleanup
        try:
            to_grass(
                dataset=sample_da,
                mapset=mapset_arg,
                create=False,
            )
            grass_raster_name_full = f"{sample_da.name}@{target_mapset_name}"
            available_rasters_3d = grass_i.list_raster3d(mapset=target_mapset_name)

            assert grass_raster_name_full in available_rasters_3d, (
                f"3D Raster base name '{sample_da.name}' not found in mapset '{target_mapset_name}'. Found: {available_rasters_3d}"
            )
            info = gs.parse_command(
                "r3.info", map=grass_raster_name_full, flags="g", quiet=True
            )
            assert int(info["depths"]) == img_depth
            assert int(info["rows"]) == img_height
            assert int(info["cols"]) == img_width

            # Run univar in the adequate region
            old_region = grass_i.get_region()
            try:
                gs.run_command(
                    "g.region", flags="o", raster_3d=grass_raster_name_full, quiet=True
                )
                univar3_stats = gs.parse_command(
                    "r3.univar", map=grass_raster_name_full, flags="g", quiet=True
                )
            finally:
                grass_i.set_region(old_region)

            mean_val_grass = float(univar3_stats["mean"])
            min_val_grass = float(univar3_stats["min"])
            max_val_grass = float(univar3_stats["max"])
            assert np.isclose(mean_val_grass, sample_da.mean().item())
            assert np.isclose(min_val_grass, sample_da.min().item())
            assert np.isclose(max_val_grass, sample_da.max().item())

        finally:
            gs.run_command(
                "g.remove",
                flags="f",
                type="raster_3d",
                name=grass_raster_name_full,
                quiet=True,
            )

    @pytest.mark.parametrize("mapset_is_path_obj", [False, True])
    @pytest.mark.parametrize("time_dim_type", ["absolute", "relative"])
    def test_dataarray_to_strds_conversion(
        self,
        temp_gisdb,
        grass_i: GrassInterface,
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

        dims_spec_for_helper = {
            "start_time": time_coords,
            "y": np.arange(img_height, dtype=float),
            "x": np.arange(img_width, dtype=float),
        }
        expected_dims_order_in_da = ("start_time", "y", "x")

        shape = (num_times, img_height, img_width)
        session_crs_wkt = grass_i.get_crs_wkt_str()

        sample_da = create_sample_dataarray(
            dims_spec=dims_spec_for_helper,
            shape=shape,
            crs_wkt=session_crs_wkt,
            name="test_strds",
            time_dim_type=time_dim_type,
            fill_value_generator=lambda s: np.arange(s[0] * s[1] * s[2])
            .reshape(s)
            .astype(float),
        )
        assert sample_da.dims == expected_dims_order_in_da, (
            f"DataArray dims {sample_da.dims} do not match expected {expected_dims_order_in_da}"
        )

        target_mapset_name = temp_gisdb.mapset
        mapset_path_obj = (
            Path(temp_gisdb.gisdb) / temp_gisdb.project / target_mapset_name
        )
        mapset_arg = mapset_path_obj if mapset_is_path_obj else str(mapset_path_obj)

        to_grass(
            dataset=sample_da,
            mapset=mapset_arg,
            create=False,
        )
        strds_id = f"{sample_da.name}@{target_mapset_name}"

        # make sure to delete file
        try:
            available_strds = grass_i.list_strds()
            assert strds_id in available_strds, (
                f"STRDS '{strds_id}' not found. Found: {available_strds}"
            )

            strds_maps_in_grass = grass_i.list_maps_in_strds(strds_id)
            strds_map_names_in_grass = [m.id for m in strds_maps_in_grass]

            assert len(strds_map_names_in_grass) == num_times, (
                f"Expected {num_times} maps in STRDS '{strds_id}', found {len(strds_map_names_in_grass)}."
            )

            # Check statistics for the first and last time slices
            indices_to_check = [0, num_times - 1] if num_times > 0 else []
            for idx_in_da_time in indices_to_check:
                time_val = sample_da.start_time.values[idx_in_da_time]
                da_slice = sample_da.sel(start_time=time_val).astype(
                    float
                )  # Ensure float for comparison

                map_to_check_full = strds_map_names_in_grass[idx_in_da_time]

                old_region = grass_i.get_region()
                try:
                    gs.run_command(
                        "g.region", flags="o", raster=map_to_check_full, quiet=True
                    )
                    univar_stats = gs.parse_command(
                        "r.univar", map=map_to_check_full, flags="g", quiet=True
                    )
                finally:
                    grass_i.set_region(old_region)

                assert np.isclose(
                    float(univar_stats.get("min", np.nan)),
                    da_slice.min().item(),
                    equal_nan=True,
                )
                assert np.isclose(
                    float(univar_stats.get("max", np.nan)),
                    da_slice.max().item(),
                    equal_nan=True,
                )
                assert np.isclose(
                    float(univar_stats.get("mean", np.nan)),
                    da_slice.mean().item(),
                    equal_nan=True,
                )
        finally:
            gs.run_command("t.remove", inputs=strds_id, type="strds", flags="rfd")

    @pytest.mark.parametrize("mapset_is_path_obj", [False, True])
    @pytest.mark.parametrize("time_dim_type", ["absolute", "relative"])
    def test_dataarray_to_str3ds_conversion(
        self,
        temp_gisdb,
        grass_i: GrassInterface,
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
            time_coords = pd.date_range(start="2024-01-01", periods=num_times, freq="h")
        elif time_dim_type == "relative":
            time_coords = np.arange(1, num_times + 1)
        else:
            pytest.fail(f"Unsupported time_dim_type: {time_dim_type}")

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
            name="test_str3ds_vol",
            time_dim_type=time_dim_type,
            fill_value_generator=lambda s: np.arange(s[0] * s[1] * s[2] * s[3])
            .reshape(s)
            .astype(float),
        )
        assert sample_da.dims == expected_dims_order_in_da, (
            f"DataArray dims {sample_da.dims} do not match expected {expected_dims_order_in_da}"
        )

        target_mapset_name = temp_gisdb.mapset
        mapset_path_obj = (
            Path(temp_gisdb.gisdb) / temp_gisdb.project / target_mapset_name
        )
        mapset_arg = mapset_path_obj if mapset_is_path_obj else str(mapset_path_obj)

        to_grass(
            dataset=sample_da,
            mapset=mapset_arg,
            create=False,
            dims={"start_time": "time"},
        )
        try:
            available_str3ds = grass_i.list_str3ds()
            str3ds_id = grass_i.get_id_from_name(sample_da.name)
            assert str3ds_id in available_str3ds, (
                f"STR3DS '{str3ds_id}' not found. Found: {available_str3ds}"
            )

            str3ds_maps_in_grass = grass_i.list_maps_in_str3ds(str3ds_id)
            str3ds_map_names_in_grass = [m.id for m in str3ds_maps_in_grass]
            assert len(str3ds_map_names_in_grass) == num_times, (
                f"Expected {num_times} maps in STR3DS '{str3ds_id}', found {len(str3ds_maps_in_grass)}."
            )

            # Check statistics for the first and last time slices
            indices_to_check = [0, num_times - 1] if num_times > 0 else []
            for idx_in_da_time in indices_to_check:
                time_val = sample_da.time.values[idx_in_da_time]
                da_slice = sample_da.sel(time=time_val).astype(float)

                map_to_check_full = str3ds_map_names_in_grass[idx_in_da_time]

                old_region = grass_i.get_region()
                try:
                    gs.run_command(
                        "g.region", flags="o", raster_3d=map_to_check_full, quiet=True
                    )
                    univar_stats = gs.parse_command(
                        "r3.univar", map=map_to_check_full, flags="g", quiet=True
                    )
                finally:
                    grass_i.set_region(old_region)

                assert np.isclose(
                    float(univar_stats.get("min", np.nan)),
                    da_slice.min().item(),
                    equal_nan=True,
                )
                assert np.isclose(
                    float(univar_stats.get("max", np.nan)),
                    da_slice.max().item(),
                    equal_nan=True,
                )
                assert np.isclose(
                    float(univar_stats.get("mean", np.nan)),
                    da_slice.mean().item(),
                    equal_nan=True,
                )
        finally:
            gs.run_command("t.remove", inputs=str3ds_id, type="str3ds", flags="rfd")

    @pytest.mark.parametrize("mapset_is_path_obj", [False, True])
    def test_dataset_conversion_mixed_types(
        self,
        temp_gisdb,
        grass_i: GrassInterface,
        mapset_is_path_obj: bool,
    ):
        """Test conversion of an xr.Dataset with mixed DataArray types."""
        session_crs_wkt = grass_i.get_crs_wkt_str()
        target_mapset_name = temp_gisdb.mapset  # Use PERMANENT mapset
        mapset_path_obj = (
            Path(temp_gisdb.gisdb) / temp_gisdb.project / target_mapset_name
        )
        mapset_arg = mapset_path_obj if mapset_is_path_obj else str(mapset_path_obj)

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
        raster2d_id = grass_i.get_id_from_name("raster2d_var")
        raster3d_id = grass_i.get_id_from_name("raster3d_var")
        strds_id = grass_i.get_id_from_name("strds_var")
        str3ds_id = grass_i.get_id_from_name("str3ds_var")

        sample_ds = create_sample_dataset(
            data_vars_specs=dataset_specs,
            crs_wkt=session_crs_wkt,
            global_time_dim_type="relative",
        )

        to_grass(
            dataset=sample_ds,
            mapset=mapset_arg,
            create=False,
            dims={"start_time": "time"},
        )
        try:
            grass_objects = grass_i.list_grass_objects()

            # 2D Raster
            available_rasters = grass_objects["raster"]
            assert raster2d_id in available_rasters
            info2d = gs.parse_command("r.info", map=raster2d_id, flags="g", quiet=True)
            assert int(info2d["rows"]) == da_2d_spec["shape"][0]
            assert int(info2d["cols"]) == da_2d_spec["shape"][1]

            # 3D Raster
            available_rasters_3d = grass_objects["raster_3d"]
            assert raster3d_id in available_rasters_3d
            info3d = gs.parse_command("r3.info", map=raster3d_id, flags="g", quiet=True)
            assert int(info3d["depths"]) == da_3d_spec["shape"][0]
            assert int(info3d["rows"]) == da_3d_spec["shape"][1]
            assert int(info3d["cols"]) == da_3d_spec["shape"][2]

            # STRDS
            available_strds = grass_objects["strds"]
            assert strds_id in available_strds
            num_strds_maps = len(grass_i.list_maps_in_strds(strds_id))
            assert num_strds_maps == strds_spec["shape"][0]

            # STR3DS
            available_str3ds = grass_objects["str3ds"]
            assert str3ds_id in available_str3ds
            num_str3ds_maps = len(grass_i.list_maps_in_str3ds(str3ds_id))
            assert num_str3ds_maps == str3ds_spec["shape"][0]
        finally:
            try:
                gs.run_command(
                    "g.remove",
                    flags="f",
                    type="raster",
                    name=raster2d_id,
                    quiet=True,
                )
            except CalledModuleError:
                pass
            try:
                gs.run_command(
                    "g.remove",
                    flags="f",
                    type="raster_3d",
                    name=raster3d_id,
                    quiet=True,
                )
            except CalledModuleError:
                pass
            try:
                gs.run_command(
                    "t.remove", inputs=strds_id, type="strds", flags="rfd", quiet=True
                )
            except CalledModuleError:
                pass
            try:
                gs.run_command(
                    "t.remove", inputs=str3ds_id, type="str3ds", flags="rfd", quiet=True
                )
            except CalledModuleError:
                pass

    def test_mapset_creation_true(self, temp_gisdb, grass_i: GrassInterface):
        """Test mapset creation when create=True."""
        pytest.skip(
            "Skipping mapset creation test due to GRASS <8.5 tgis.init() bug with changing mapsets in active session."
        )
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

        to_grass(dataset=sample_da, mapset=str(mapset_path), create=True)

        assert mapset_path.exists() and mapset_path.is_dir(), (
            f"Mapset directory {mapset_path} was not created."
        )

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
        pytest.skip(
            "Skipping due to GRASS 8.4 tgis bug related to mapset switching/creation."
        )
        session_crs_wkt = grass_i.get_crs_wkt_str()
        existing_mapset_name = "existing_mapset_for_test"
        mapset_path = Path(temp_gisdb.gisdb) / temp_gisdb.project / existing_mapset_name
        if mapset_path.exists():
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

        to_grass(dataset=sample_da, mapset=str(mapset_path), create=False)

        available_rasters = grass_i.list_raster(mapset=existing_mapset_name)
        assert sample_da.name in available_rasters, (
            f"Raster '{sample_da.name}' not found in existing mapset '{existing_mapset_name}'."
        )

    @pytest.mark.parametrize(
        "rename_map", [None, {"y": "northing", "x": "easting"}, {"y": "custom_y"}]
    )
    def test_dims_mapping(
        self,
        temp_gisdb,
        grass_i: GrassInterface,
        rename_map: dict,
    ):
        """Test 'dims' mapping functionality."""
        session_crs_wkt = grass_i.get_crs_wkt_str()
        target_mapset_name = temp_gisdb.mapset
        mapset_path = Path(temp_gisdb.gisdb) / temp_gisdb.project / target_mapset_name
        da_name = "dims_test_raster"
        img_height, img_width = 3, 2

        da_dims_spec = {
            "y": np.arange(img_height, dtype=float),
            "x": np.arange(img_width, dtype=float),
        }

        sample_da = create_sample_dataarray(
            dims_spec=da_dims_spec,
            shape=(img_height, img_width),
            crs_wkt=session_crs_wkt,
            name=da_name,
        )

        if rename_map:
            sample_da = sample_da.rename(rename_map)
        to_grass(
            dataset=sample_da,
            mapset=str(mapset_path),
            create=False,
            dims=rename_map,
        )
        available_rasters = grass_i.list_raster(mapset=target_mapset_name)
        assert grass_i.get_id_from_name(da_name) in available_rasters

        info = gs.parse_command(
            "r.info", map=f"{da_name}@{target_mapset_name}", flags="g", quiet=True
        )
        assert int(info["rows"]) == img_height
        assert int(info["cols"]) == img_width
