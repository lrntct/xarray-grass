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


@pytest.mark.usefixtures("grass_session_fixture")
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
        # z, y_3d, x_3d, y, x, absolute and relative time
        assert len(test_dataset.dims) == 7
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
        assert len(whole_mapset.dims) == 7
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
            drop_variables=[ACTUAL_RASTER_MAP],
        )
        region = grass_i.get_region()
        assert isinstance(test_dataset, xr.Dataset)
        # z, y_3d, x_3d, y, x, absolute and relative time
        assert len(test_dataset.dims) == 7
        assert len(test_dataset) == 6  # 7 - 1 dropped variable
        assert len(test_dataset.x_3d) == region.cols3
        assert len(test_dataset.y_3d) == region.rows3
        assert len(test_dataset.x) == region.cols
        assert len(test_dataset.y) == region.rows
        assert len(test_dataset.z) == region.depths
