from pathlib import Path
import os

import pytest
import xarray as xr

from xarray_grass import GrassInterface
from xarray_grass.xarray_grass import dir_is_grass_mapset
from xarray_grass.xarray_grass import dir_is_grass_project

ACTUAL_STRDS = "LST_Day_monthly@modis_lst"
ACTUAL_RASTER_MAP = "elevation@PERMANENT"
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
        assert isinstance(test_dataset, xr.Dataset)
        assert len(test_dataset.dims) == 2
        assert len(test_dataset.x) == grass_i.cols
        assert len(test_dataset.y) == grass_i.rows

    def test_load_raster3d(self, grass_i, temp_gisdb):
        assert True

    def test_load_strds(self, grass_i, temp_gisdb) -> None:
        mapset_path = (
            Path(temp_gisdb.gisdb) / Path(temp_gisdb.project) / Path(temp_gisdb.mapset)
        )
        test_dataset = xr.open_dataset(mapset_path, strds=ACTUAL_STRDS)
        # print(test_dataset)
        assert isinstance(test_dataset, xr.Dataset)
        assert len(test_dataset.dims) == 3
        assert len(test_dataset.x) == grass_i.cols
        assert len(test_dataset.y) == grass_i.rows

    def test_load_str3ds(self, grass_i, temp_gisdb) -> None:
        pass

    def test_load_whole_mapset(self, grass_i, temp_gisdb):
        pass

    def test_load_bad_name(self, temp_gisdb) -> None:
        mapset_path = (
            Path(temp_gisdb.gisdb) / Path(temp_gisdb.project) / Path(temp_gisdb.mapset)
        )
        with pytest.raises(ValueError):
            xr.open_dataset(mapset_path, raster="not_a_real_map@PERMANENT")
            xr.open_dataset(mapset_path, raster="not_a_real_map")
            xr.open_dataset(mapset_path, str3ds=ACTUAL_RASTER_MAP)

    def test_drop_variables(self):
        pass
