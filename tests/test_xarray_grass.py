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
import os

import pytest
import xarray as xr

from xarray_grass.xarray_grass import dir_is_grass_mapset
from xarray_grass.xarray_grass import dir_is_grass_project

ACTUAL_STRDS = "LST_Day_monthly@modis_lst"
ACTUAL_RASTER_MAP = "elevation@PERMANENT"
ACTUAL_RASTER_MAP2 = "MOD11B3.A2015060.h11v05.single_LST_Day_6km@modis_lst"
ACTUAL_RASTER3D_MAP = "test3d_1@PERMANENT"
ACTUAL_RASTER3D_MAP2 = "test3d_3@PERMANENT"
RELATIVE_STR3DS = "test_str3ds_relative"
ABSOLUTE_STR3DS = "test_str3ds_absolute"


def test_dir_is_grass_project(grass_session_fixture, temp_gisdb):
    mapset_path = (
        Path(temp_gisdb.gisdb) / Path(temp_gisdb.project) / Path(temp_gisdb.mapset)
    )
    project_path = Path(temp_gisdb.gisdb) / Path(temp_gisdb.project)
    assert dir_is_grass_project(project_path)
    assert not dir_is_grass_project(mapset_path)
    assert not dir_is_grass_project("not a project")
    assert not dir_is_grass_project(Path("not a project"))
    assert not dir_is_grass_project([list, dict])  # Nonsensical input


def test_dir_is_grass_project_manual_structure(tmp_path: Path):
    # Case 1: Valid project structure
    valid_project = tmp_path / "my_grass_location"
    valid_project.mkdir()
    (valid_project / "PERMANENT").mkdir()
    assert dir_is_grass_project(valid_project)
    assert dir_is_grass_project(str(valid_project))

    # Case 2: Directory exists, but no PERMANENT subdir
    no_permanent_dir = tmp_path / "not_a_location"
    no_permanent_dir.mkdir()
    assert not dir_is_grass_project(no_permanent_dir)

    # Case 3: Directory exists, PERMANENT exists but is a file
    permanent_is_file_dir = tmp_path / "permanent_is_file"
    permanent_is_file_dir.mkdir()
    (permanent_is_file_dir / "PERMANENT").touch()
    assert not dir_is_grass_project(permanent_is_file_dir)

    # Case 4: Path is a file, not a directory
    file_path = tmp_path / "some_file.txt"
    file_path.touch()
    assert not dir_is_grass_project(file_path)

    # Case 5: Path does not exist
    non_existent_path = tmp_path / "does_not_exist"
    assert not dir_is_grass_project(non_existent_path)

    # Case 6: Path is None (dir_is_grass_project handles TypeError from Path(None) and returns False)
    assert not dir_is_grass_project(None)


def test_dir_is_grass_mapset(grass_session_fixture, temp_gisdb):
    mapset_path = (
        Path(temp_gisdb.gisdb) / Path(temp_gisdb.project) / Path(temp_gisdb.mapset)
    )
    assert dir_is_grass_mapset(mapset_path)
    project_path = Path(temp_gisdb.gisdb) / Path(temp_gisdb.project)
    assert not dir_is_grass_mapset(project_path)
    assert not dir_is_grass_mapset("not a mapset")
    assert not dir_is_grass_mapset(Path("not a mapset"))
    assert not dir_is_grass_mapset([list, dict])  # Nonsensical input


@pytest.mark.usefixtures("grass_session_fixture", "grass_test_region")
class TestXarrayGrass:
    def test_load_raster(self, grass_i, temp_gisdb) -> None:
        mapset_path = os.path.join(
            str(temp_gisdb.gisdb), str(temp_gisdb.project), str(temp_gisdb.mapset)
        )
        test_dataset = xr.open_dataset(mapset_path, raster=ACTUAL_RASTER_MAP)
        region = grass_i.get_region()
        assert isinstance(test_dataset, xr.Dataset)
        assert len(test_dataset.dims) == 2
        assert len(test_dataset.x) == region.cols
        assert len(test_dataset.y) == region.rows

    def test_load_raster3d(self, grass_i, temp_gisdb):
        mapset_path = os.path.join(
            str(temp_gisdb.gisdb), str(temp_gisdb.project), str(temp_gisdb.mapset)
        )
        test_dataset = xr.open_dataset(mapset_path, raster_3d=ACTUAL_RASTER3D_MAP)
        region = grass_i.get_region()
        assert isinstance(test_dataset, xr.Dataset)
        assert len(test_dataset.dims) == 3
        assert len(test_dataset.x_3d) == region.cols3
        assert len(test_dataset.y_3d) == region.rows3
        assert len(test_dataset.z) == region.depths
        assert True

    def test_load_strds(self, grass_i, temp_gisdb) -> None:
        mapset_path = (
            Path(temp_gisdb.gisdb) / Path(temp_gisdb.project) / Path(temp_gisdb.mapset)
        )
        test_dataset = xr.open_dataset(mapset_path, strds=ACTUAL_STRDS)
        region = grass_i.get_region()
        assert isinstance(test_dataset, xr.Dataset)
        assert len(test_dataset.dims) == 3
        assert len(test_dataset.x) == region.cols
        assert len(test_dataset.y) == region.rows

    def test_load_str3ds(self, grass_i, temp_gisdb) -> None:
        mapset_path = os.path.join(
            str(temp_gisdb.gisdb), str(temp_gisdb.project), str(temp_gisdb.mapset)
        )
        test_dataset = xr.open_dataset(mapset_path, str3ds=RELATIVE_STR3DS)
        region = grass_i.get_region()
        assert isinstance(test_dataset, xr.Dataset)
        assert len(test_dataset.dims) == 4
        assert len(test_dataset.x_3d) == region.cols3
        assert len(test_dataset.y_3d) == region.rows3
        assert len(test_dataset.z) == region.depths

    def test_load_multiple_rasters(self, grass_i, temp_gisdb) -> None:
        mapset_path = os.path.join(
            str(temp_gisdb.gisdb), str(temp_gisdb.project), str(temp_gisdb.mapset)
        )
        test_dataset = xr.open_dataset(
            mapset_path,
            raster=[ACTUAL_RASTER_MAP, ACTUAL_RASTER_MAP2],
            raster_3d=[ACTUAL_RASTER3D_MAP, ACTUAL_RASTER3D_MAP2],
            str3ds=[RELATIVE_STR3DS, ABSOLUTE_STR3DS],
            strds=ACTUAL_STRDS,
        )
        region = grass_i.get_region()
        assert isinstance(test_dataset, xr.Dataset)
        # z, y_3d, x_3d, y, x, and one time dimension for each stds
        assert len(test_dataset.dims) == 8
        assert len(test_dataset) == 7
        assert len(test_dataset.x_3d) == region.cols3
        assert len(test_dataset.y_3d) == region.rows3
        assert len(test_dataset.x) == region.cols
        assert len(test_dataset.y) == region.rows
        assert len(test_dataset.z) == region.depths

    def test_load_whole_mapset(self, grass_i, temp_gisdb) -> None:
        mapset_path = (
            Path(temp_gisdb.gisdb) / Path(temp_gisdb.project) / Path(temp_gisdb.mapset)
        )
        whole_mapset = xr.open_dataset(mapset_path)
        region = grass_i.get_region()
        dict_grass_objects = grass_i.list_grass_objects()

        # rasters and strds
        list_strds_id = dict_grass_objects["strds"]
        list_strds_name = [
            grass_i.get_name_from_id(strds_id) for strds_id in list_strds_id
        ]
        list_rasters = [
            grass_i.get_name_from_id(r) for r in dict_grass_objects["raster"]
        ]
        list_rasters_in_strds = []
        for strds_id in list_strds_id:
            list_rasters_in_strds.extend(
                [
                    grass_i.get_name_from_id(map_data.id)
                    for map_data in grass_i.list_maps_in_strds(strds_id)
                ]
            )
        list_rasters_not_in_strds = [
            r for r in list_rasters if r not in list_rasters_in_strds
        ]

        # raster_3d and str3ds
        list_str3ds_id = dict_grass_objects["str3ds"]
        list_str3ds_name = [
            grass_i.get_name_from_id(str3ds_id) for str3ds_id in list_str3ds_id
        ]
        list_raster3d = [
            grass_i.get_name_from_id(r) for r in dict_grass_objects["raster_3d"]
        ]
        list_raster3d_in_str3ds = []
        for str3ds_id in list_str3ds_id:
            list_raster3d_in_str3ds.extend(
                [
                    grass_i.get_name_from_id(map_data.id)
                    for map_data in grass_i.list_maps_in_str3ds(str3ds_id)
                ]
            )
        list_raster3d_not_in_str3ds = [
            r for r in list_raster3d if r not in list_raster3d_in_str3ds
        ]

        all_variables = (
            list_raster3d_not_in_str3ds
            + list_rasters_not_in_strds
            + list_strds_name
            + list_str3ds_name
        )
        assert isinstance(whole_mapset, xr.Dataset)
        dim_num = len(
            list_strds_name + list_str3ds_name + ["y", "x", "y_3d", "x_3d", "z"]
        )
        assert len(whole_mapset.dims) == dim_num
        assert len(whole_mapset) == len(all_variables)
        assert all(var in whole_mapset for var in all_variables)
        assert len(whole_mapset.x_3d) == region.cols3
        assert len(whole_mapset.y_3d) == region.rows3
        assert len(whole_mapset.x) == region.cols
        assert len(whole_mapset.y) == region.rows
        assert len(whole_mapset.z) == region.depths

    def test_load_bad_name(self, temp_gisdb) -> None:
        mapset_path = (
            Path(temp_gisdb.gisdb) / Path(temp_gisdb.project) / Path(temp_gisdb.mapset)
        )
        with pytest.raises(ValueError):
            xr.open_dataset(mapset_path, raster="not_a_real_map@PERMANENT")
            xr.open_dataset(mapset_path, raster="not_a_real_map")
            xr.open_dataset(mapset_path, str3ds=ACTUAL_RASTER_MAP)

    def test_drop_variables(self, grass_i, temp_gisdb) -> None:
        mapset_path = os.path.join(
            str(temp_gisdb.gisdb), str(temp_gisdb.project), str(temp_gisdb.mapset)
        )
        test_dataset = xr.open_dataset(
            mapset_path,
            raster=[ACTUAL_RASTER_MAP, ACTUAL_RASTER_MAP2],
            raster_3d=[ACTUAL_RASTER3D_MAP, ACTUAL_RASTER3D_MAP2],
            str3ds=[RELATIVE_STR3DS, ABSOLUTE_STR3DS],
            strds=ACTUAL_STRDS,
            drop_variables=[ABSOLUTE_STR3DS, ACTUAL_RASTER_MAP],
        )
        region = grass_i.get_region()
        assert isinstance(test_dataset, xr.Dataset)
        # Each stds has its own time dimension
        num_dims = len(
            ["z", "y_3d", "x_3d", "y", "x"] + [ACTUAL_STRDS, RELATIVE_STR3DS]
        )
        assert len(test_dataset.dims) == num_dims
        assert len(test_dataset) == 7 - 2  # dropped vars
        assert len(test_dataset.x_3d) == region.cols3
        assert len(test_dataset.y_3d) == region.rows3
        assert len(test_dataset.x) == region.cols
        assert len(test_dataset.y) == region.rows

    def test_attributes_separation(self, grass_i, temp_gisdb) -> None:
        """Test that DataArray attributes don't leak to Dataset level."""
        mapset_path = os.path.join(
            str(temp_gisdb.gisdb), str(temp_gisdb.project), str(temp_gisdb.mapset)
        )
        region = grass_i.get_region()

        # Test with multiple types of data to ensure attributes are handled correctly
        test_dataset = xr.open_dataset(
            mapset_path,
            raster=ACTUAL_RASTER_MAP,
            raster_3d=ACTUAL_RASTER3D_MAP,
            strds=ACTUAL_STRDS,
            str3ds=RELATIVE_STR3DS,
        )

        # Dataset-level attributes: only these should be present
        expected_dataset_attrs = {"crs_wkt", "Conventions", "source", "history"}
        actual_dataset_attrs = set(test_dataset.attrs.keys())
        assert actual_dataset_attrs == expected_dataset_attrs, (
            f"Dataset has unexpected attributes. "
            f"Expected: {expected_dataset_attrs}, "
            f"Got: {actual_dataset_attrs}"
        )

        # Verify Dataset attrs have correct values
        assert "Conventions" in test_dataset.attrs
        assert test_dataset.attrs["Conventions"] == "CF-1.13-draft"
        assert "crs_wkt" in test_dataset.attrs
        assert isinstance(test_dataset.attrs["crs_wkt"], str)

        # DataArray-level attributes that should NOT appear at Dataset level
        dataarray_only_attrs = {"long_name", "units", "comment"}

        # Check that DataArray attributes don't leak to Dataset level
        for attr in dataarray_only_attrs:
            assert attr not in test_dataset.attrs, (
                f"DataArray attribute '{attr}' should not appear at Dataset level"
            )

        # Verify each DataArray has its own attributes
        for var_name in test_dataset.data_vars:
            data_array = test_dataset[var_name]

            # Each DataArray should have at least some of these attributes
            # (depending on the data source, some might be empty strings)
            for attr in ["long_name", "source", "units"]:
                assert attr in data_array.attrs, (
                    f"DataArray '{var_name}' missing expected attribute '{attr}'"
                )

        # Verify coordinate attributes are set correctly (these are set by set_cf_coordinates)
        # Check x coordinate
        assert "axis" in test_dataset.x.attrs
        assert test_dataset.x.attrs["axis"] == "X"

        # Check y coordinate
        assert "axis" in test_dataset.y.attrs
        assert test_dataset.y.attrs["axis"] == "Y"

        # Check z coordinate for 3D data
        if "z" in test_dataset.coords:
            assert "axis" in test_dataset.z.attrs
            assert test_dataset.z.attrs["axis"] == "Z"
            assert len(test_dataset.z) == region.depths

        # Check time coordinate attributes (from STRDS)
        time_coords = [coord for coord in test_dataset.coords if "time" in coord]
        assert len(time_coords) > 0, "Expected at least one time coordinate from STRDS"

        for time_coord in time_coords:
            # Time coordinates should have axis="T"
            assert "axis" in test_dataset[time_coord].attrs
            assert test_dataset[time_coord].attrs["axis"] == "T"

            # Time coordinates should have standard_name="time"
            assert "standard_name" in test_dataset[time_coord].attrs
            assert test_dataset[time_coord].attrs["standard_name"] == "time"

            # For relative time coordinates, units should be present
            if time_coord.endswith(RELATIVE_STR3DS) or time_coord.endswith(
                RELATIVE_STR3DS
            ):
                # This is a relative time coordinate, should have units
                assert "units" in test_dataset[time_coord].attrs
        assert len(test_dataset.z) == region.depths
