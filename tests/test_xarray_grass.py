from pathlib import Path

import pytest
import xarray as xr

from xarray_grass import GrassInterface

ACTUAL_STRDS = "LST_Day_monthly@modis_lst"
ACTUAL_RASTER_MAP = "elevation@PERMANENT"


def test_load_strds(grass_session_fixture, temp_gisdb) -> None:
    grass_i = GrassInterface()
    mapset_path = (
        Path(temp_gisdb.gisdb) / Path(temp_gisdb.project) / Path(temp_gisdb.mapset)
    )
    test_dataset = xr.open_dataset(mapset_path, grass_object_name=ACTUAL_STRDS)
    # print(test_dataset)
    assert isinstance(test_dataset, xr.Dataset)
    assert len(test_dataset.dims) == 3
    assert len(test_dataset.x) == grass_i.xr
    assert len(test_dataset.y) == grass_i.yr


def test_load_bad_name(grass_session_fixture, temp_gisdb) -> None:
    mapset_path = (
        Path(temp_gisdb.gisdb) / Path(temp_gisdb.project) / Path(temp_gisdb.mapset)
    )
    with pytest.raises(ValueError):
        xr.open_dataset(mapset_path, grass_object_name="not_a_real_map@PERMANENT")
        xr.open_dataset(mapset_path, grass_object_name="not_a_real_map")
        # /!\ remove when implementing raster loading
        xr.open_dataset(mapset_path, grass_object_name=ACTUAL_RASTER_MAP)
