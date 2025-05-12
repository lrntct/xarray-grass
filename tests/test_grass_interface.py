from collections import namedtuple
from datetime import datetime

import pytest
import numpy as np
import grass_session
import grass.script as gscript
import grass.exceptions as gexceptions

from xarray_grass import GrassConfig, GrassInterface


ACTUAL_STRDS = "LST_Day_monthly@modis_lst"
ACTUAL_RASTER_MAP = "elevation@PERMANENT"


@pytest.fixture(scope="class")
def grass_session_fixture(temp_gisdb: GrassConfig):
    """Initialize a GRASS session for tests."""
    with grass_session.Session(
        gisdb=temp_gisdb.gisdb, location=temp_gisdb.project, mapset=temp_gisdb.mapset
    ) as session:
        # add the mapset to the session
        gscript.run_command("g.mapsets", mapset="modis_lst")
        yield session
        session.close()


def test_no_grass_session():
    with pytest.raises(RuntimeError):
        GrassInterface(region_id=None)


@pytest.mark.usefixtures("grass_session_fixture")
class TestGrassInterface:
    def test_grass_dtype(grass_session_fixture) -> None:
        """Test the dtype conversion frm numpy to GRASS."""
        grass_i = GrassInterface(region_id=None)
        assert grass_i.grass_dtype("bool_") == "CELL"
        assert grass_i.grass_dtype("int_") == "CELL"
        assert grass_i.grass_dtype("int8") == "CELL"
        assert grass_i.grass_dtype("int16") == "CELL"
        assert grass_i.grass_dtype("int32") == "CELL"
        assert grass_i.grass_dtype("intc") == "CELL"
        assert grass_i.grass_dtype("intp") == "CELL"
        assert grass_i.grass_dtype("uint8") == "CELL"
        assert grass_i.grass_dtype("uint16") == "CELL"
        assert grass_i.grass_dtype("uint32") == "CELL"
        assert grass_i.grass_dtype("float32") == "FCELL"
        assert grass_i.grass_dtype("float64") == "DCELL"
        with pytest.raises(ValueError):
            grass_i.grass_dtype("bool")
            grass_i.grass_dtype("int")
            grass_i.grass_dtype("float")

    def test_is_latlon(grass_session_fixture):
        assert GrassInterface.is_latlon() is False

    def test_format_id(grass_session_fixture):
        assert GrassInterface.format_id("test_map") == "test_map@PERMANENT"
        assert GrassInterface.format_id("test_map@PERMANENT") == "test_map@PERMANENT"
        assert GrassInterface.format_id("") == "@PERMANENT"
        with pytest.raises(TypeError):
            GrassInterface.format_id(False)
            GrassInterface.format_id(12.4)
            GrassInterface.format_id(4)

    def test_name_is_stds(grass_session_fixture):
        assert GrassInterface.name_is_stds(ACTUAL_STRDS) is True
        assert GrassInterface.name_is_stds(ACTUAL_RASTER_MAP) is False
        assert GrassInterface.name_is_stds("not_a_real_strds@PERMANENT") is False
        with pytest.raises(gexceptions.FatalError):
            GrassInterface.name_is_stds("not_a_real_strds@NOT_A_MAPSET")
            GrassInterface.name_is_stds("not_a_real_strds")

    def test_name_is_map(grass_session_fixture):
        assert GrassInterface.name_is_map(ACTUAL_RASTER_MAP) is True
        assert GrassInterface.name_is_map(ACTUAL_STRDS) is False
        assert GrassInterface.name_is_map("not_a_real_map@PERMANENT") is False
        assert GrassInterface.name_is_map("not_a_real_map@NOT_A_MAPSET") is False
        assert GrassInterface.name_is_map("not_a_real_map") is False

    def test_get_proj_as_dict(grass_session_fixture):
        proj_dict = GrassInterface.get_proj_as_dict()
        assert proj_dict["name"] == "Lambert Conformal Conic"
        assert proj_dict["proj"] == "lcc"
        assert proj_dict["datum"] == "nad83"
        assert proj_dict["es"] == "0.006694380022900787"
        assert proj_dict["lat_1"] == "36.16666666666666"
        assert proj_dict["lat_2"] == "34.33333333333334"
        assert proj_dict["lat_0"] == "33.75"
        assert proj_dict["lon_0"] == "-79"
        assert proj_dict["x_0"] == "609601.22"
        assert proj_dict["no_defs"] == "defined"
        assert proj_dict["srid"] == "EPSG:3358"
        assert proj_dict["unit"] == "Meter"
        assert proj_dict["units"] == "Meters"
        assert proj_dict["meters"] == "1"

    def test_has_mask(grass_session_fixture):
        assert GrassInterface.has_mask() is False
        gscript.run_command("r.mask", quiet=True, raster=ACTUAL_RASTER_MAP)
        assert GrassInterface.has_mask() is True
        gscript.run_command("r.mask", flags="r")
        assert GrassInterface.has_mask() is False

    def test_list_strds(grass_session_fixture):
        strds_list = GrassInterface.list_strds()
        assert strds_list == [ACTUAL_STRDS]

    def test_get_strds_infos(grass_session_fixture):
        grass_i = GrassInterface(region_id=None)
        strds_infos = grass_i.get_strds_infos(ACTUAL_STRDS)
        assert strds_infos.id == ACTUAL_STRDS
        assert strds_infos.temporal_type == "absolute"
        assert strds_infos.time_unit is None
        assert strds_infos.start_time == datetime(2015, 1, 1, 0, 0)
        assert strds_infos.end_time == datetime(2017, 1, 1, 0, 0)
        assert strds_infos.time_granularity == "1 month"
        assert strds_infos.north == pytest.approx(760180.12411493)
        assert strds_infos.south == pytest.approx(-415819.87588507)
        assert strds_infos.east == pytest.approx(1550934.46411531)
        assert strds_infos.west == pytest.approx(-448265.53588469)
        assert strds_infos.top == 0.0
        assert strds_infos.bottom == 0.0

    def test_list_maps_in_strds(grass_session_fixture):
        grass_i = GrassInterface(region_id=None)
        map_list = grass_i.list_maps_in_strds(ACTUAL_STRDS)
        assert len(map_list) == 24

    def test_read_raster_map(grass_session_fixture):
        np_map = GrassInterface.read_raster_map(ACTUAL_RASTER_MAP)
        assert np_map is not None
        assert np_map.shape == (1350, 1500)
        assert np_map.dtype == "float32"
        assert not np.isnan(np_map).any()

    def test_write_raster_map(grass_session_fixture):
        grass_i = GrassInterface(region_id=None)
        rng = np.random.default_rng()
        # tests cases
        TestCase = namedtuple("TestCase", ["np_dtype", "g_dtype", "map_name"])
        test_cases = [
            TestCase(np_dtype=np.uint8, g_dtype="CELL", map_name="test_write_int"),
            TestCase(np_dtype=np.float32, g_dtype="FCELL", map_name="test_write_f32"),
            TestCase(np_dtype=np.float64, g_dtype="DCELL", map_name="test_write_f64"),
        ]
        for test_case in test_cases:
            if test_case.g_dtype == "CELL":
                np_array = rng.integers(
                    0, 255, size=(1350, 1500), dtype=test_case.np_dtype
                )
            else:
                np_array = rng.random(size=(1350, 1500), dtype=test_case.np_dtype)
        grass_i.write_raster_map(np_array, test_case.map_name)
        map_info = gscript.parse_command(
            "r.info", flags="g", map=f"{test_case.map_name}@PERMANENT"
        )
        assert map_info["rows"] == "1350"
        assert map_info["cols"] == "1500"
        assert map_info["datatype"] == test_case.g_dtype
        # remove map
        gscript.run_command(
            "g.remove", flags="f", type="raster", name=test_case.map_name
        )

    def test_register_maps_in_stds(grass_session_fixture):
        grass_i = GrassInterface(region_id=None)
        rng = np.random.default_rng()
        np_array = rng.random(size=(1350, 1500), dtype="float32")
        grass_i.write_raster_map(np_array, "test_temporal_map1")
        grass_i.write_raster_map(np_array, "test_temporal_map2")
        maps_list = [
            ("test_temporal_map1", datetime(2023, 1, 1)),
            ("test_temporal_map2", datetime(2023, 2, 1)),
        ]
        grass_i.register_maps_in_stds(
            stds_title="test_stds_title",
            stds_name="test_stds",
            stds_desc="test description of a STRDS",
            semantic="mean",
            map_list=maps_list,
            t_type="absolute",
        )
        strds_info = gscript.parse_command(
            "t.info",
            flags="g",
            type="strds",
            input="test_stds@PERMANENT",
        )
        print(strds_info)
        assert strds_info["name"] == "test_stds"
        assert strds_info["mapset"] == "PERMANENT"
        assert strds_info["id"] == "test_stds@PERMANENT"
        assert strds_info["semantic_type"] == "mean"
        assert strds_info["temporal_type"] == "absolute"
        assert strds_info["number_of_maps"] == "2"
        # remove extra single quotes from the returned string
        assert strds_info["start_time"].strip("'") == str(datetime(2023, 1, 1))
        assert strds_info["end_time"].strip("'") == str(datetime(2023, 2, 1))
