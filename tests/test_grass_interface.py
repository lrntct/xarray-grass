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

from collections import namedtuple
from datetime import datetime
from copy import deepcopy

import pytest
import numpy as np

# Needed to import grass modules
import grass_session  # noqa: F401
import grass.script as gs
import grass.exceptions as gexceptions

from xarray_grass import GrassInterface, RegionData


ACTUAL_STRDS = "LST_Day_monthly@modis_lst"
ACTUAL_RASTER_MAP = "elevation@PERMANENT"
RELATIVE_STR3DS = "test_str3ds_relative"

TestCase = namedtuple("TestCase", ["np_dtype", "g_dtype", "map_name"])


def test_no_grass_session():
    """Test that GrassInterface raises RuntimeError when no GRASS session exists.
    This test must clear GISRC to ensure isolation from other tests that may have
    set up a GRASS session.
    """
    import os

    # Save and clear GISRC to ensure no session exists
    original_gisrc = os.environ.pop("GISRC", None)
    try:
        with pytest.raises(RuntimeError):
            GrassInterface()
    finally:
        # Restore GISRC if it was set
        if original_gisrc is not None:
            os.environ["GISRC"] = original_gisrc


@pytest.mark.usefixtures("grass_session_fixture")
class TestRegionInterface:
    def test_get_region(self, grass_i):
        """Test the get_region method."""
        region = grass_i.get_region()
        assert region is not None
        assert region.rows > 0
        assert region.cols > 0
        assert region.n > region.s
        assert region.e > region.w
        assert region.t > region.b

    @pytest.fixture
    def temp_2d_region(self, grass_i):
        region_before = deepcopy(grass_i.get_region())
        valid_data = RegionData(
            n=150.4,
            s=30.4,
            w=-12.3,
            e=150.9,
            nsres=20,
            ewres=23.3,
        )
        grass_i.set_region(valid_data)
        yield valid_data
        grass_i.set_region(region_before)

    def test_set_region_2D(self, grass_i, temp_2d_region):
        region = grass_i.get_region()
        assert region.n == pytest.approx(temp_2d_region.n)
        assert region.s == pytest.approx(temp_2d_region.s)
        assert region.w == pytest.approx(temp_2d_region.w)
        assert region.e == pytest.approx(temp_2d_region.e)
        assert region.nsres == pytest.approx(temp_2d_region.nsres, abs=1e-3)
        assert region.ewres == pytest.approx(temp_2d_region.ewres, abs=1e-1)

    @pytest.fixture
    def temp_3d_region(self, grass_i):
        region_before = deepcopy(grass_i.get_region())
        valid_data = RegionData(
            n=150.4,
            s=30.4,
            w=-12.3,
            e=150.9,
            nsres3=2.3,  # grass<8.5 requires nsres3==ewres3
            ewres3=2.3,
            t=1000.0,
            b=0.0,
            tbres=10.0,
        )
        grass_i.set_region(valid_data)
        yield valid_data
        grass_i.set_region(region_before)

    def test_set_region_3D(self, grass_i, temp_3d_region):
        region = grass_i.get_region()
        assert region.n == pytest.approx(temp_3d_region.n)
        assert region.s == pytest.approx(temp_3d_region.s)
        assert region.w == pytest.approx(temp_3d_region.w)
        assert region.e == pytest.approx(temp_3d_region.e)
        assert region.nsres3 == pytest.approx(temp_3d_region.nsres3, abs=1e-2)
        assert region.ewres3 == pytest.approx(temp_3d_region.ewres3, abs=1e-2)
        assert region.t == pytest.approx(temp_3d_region.t)
        assert region.b == pytest.approx(temp_3d_region.b)
        assert region.tbres == pytest.approx(temp_3d_region.tbres, abs=1e-3)


@pytest.mark.usefixtures("grass_session_fixture")
class TestGrassInterface:
    def test_grass_dtype(self, grass_i) -> None:
        """Test the dtype conversion frm numpy to GRASS."""
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

    def test_is_latlon(self):
        assert GrassInterface.is_latlon() is False

    def test_get_id_from_name(self):
        assert GrassInterface.get_id_from_name("test_map") == "test_map@PERMANENT"
        assert (
            GrassInterface.get_id_from_name("test_map@PERMANENT")
            == "test_map@PERMANENT"
        )
        assert GrassInterface.get_id_from_name("") == "@PERMANENT"
        with pytest.raises(TypeError):
            GrassInterface.get_id_from_name(False)
            GrassInterface.get_id_from_name(12.4)
            GrassInterface.get_id_from_name(4)

    def test_get_name_from_id(self):
        assert GrassInterface.get_name_from_id("test_map") == "test_map"
        assert GrassInterface.get_name_from_id("test_map@PERMANENT") == "test_map"
        assert GrassInterface.get_name_from_id("@PERMANENT") == ""
        assert GrassInterface.get_name_from_id("") == ""
        with pytest.raises(TypeError):
            GrassInterface.get_name_from_id(False)
            GrassInterface.get_name_from_id(2.4)
            GrassInterface.get_name_from_id(4)

    def test_name_is_stdrs(self, grass_i):
        assert grass_i.name_is_strds(ACTUAL_STRDS) is True
        assert grass_i.name_is_strds(ACTUAL_RASTER_MAP) is False
        assert grass_i.name_is_strds("not_a_real_strds@PERMANENT") is False
        with pytest.raises(gexceptions.FatalError):
            grass_i.name_is_strds("not_a_real_strds@NOT_A_MAPSET")
            grass_i.name_is_strds("not_a_real_strds")

    def test_name_is_raster(self, grass_i):
        assert grass_i.name_is_raster(ACTUAL_RASTER_MAP) is True
        assert grass_i.name_is_raster(ACTUAL_STRDS) is False
        assert grass_i.name_is_raster("not_a_real_map@PERMANENT") is False
        assert grass_i.name_is_raster("not_a_real_map@NOT_A_MAPSET") is False
        assert grass_i.name_is_raster("not_a_real_map") is False

    def test_get_crs_wkt_str(self):
        crs_str = GrassInterface.get_crs_wkt_str()
        ref_str = gs.read_command("g.proj", flags="wf")
        assert crs_str == ref_str.replace("\n", "")
        assert isinstance(crs_str, str)

    def test_has_mask(self):
        assert GrassInterface.has_mask() is False
        gs.run_command("r.mask", quiet=True, raster=ACTUAL_RASTER_MAP)
        assert GrassInterface.has_mask() is True
        gs.run_command("r.mask", flags="r")
        assert GrassInterface.has_mask() is False

    def test_list_strds(self):
        strds_list = GrassInterface.list_strds()
        assert ACTUAL_STRDS in strds_list
        assert len(strds_list) == 3  # modis + 2 synthetic ones

    def test_list_str3ds(self):
        str3ds_list = GrassInterface.list_str3ds()
        assert GrassInterface.get_id_from_name(RELATIVE_STR3DS) in str3ds_list
        assert len(str3ds_list) == 3  # 3 synthetic str3ds

    def test_get_stds_infos(self, grass_i):
        strds_infos = grass_i.get_stds_infos(ACTUAL_STRDS, stds_type="strds")
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

    def test_list_maps_in_strds(self, grass_i):
        map_list = grass_i.list_maps_in_strds(ACTUAL_STRDS)
        assert len(map_list) == 24

    def test_list_maps_in_str3ds(self, grass_i):
        map_list = grass_i.list_maps_in_str3ds(RELATIVE_STR3DS)
        assert len(map_list) == 3


@pytest.mark.usefixtures("grass_session_fixture", "grass_test_region")
class TestGrassInterfaceReadWrite:
    def test_read_raster_map(self, grass_i):
        np_map = grass_i.read_raster_map(ACTUAL_RASTER_MAP)
        region = grass_i.get_region()
        assert np_map is not None
        assert np_map.shape == (region.rows, region.cols)
        assert np_map.dtype == "float32"
        assert not np.isnan(np_map).any()

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCase(np_dtype=np.uint8, g_dtype="CELL", map_name="test_write_int"),
            TestCase(np_dtype=np.float32, g_dtype="FCELL", map_name="test_write_f32"),
            TestCase(np_dtype=np.float64, g_dtype="DCELL", map_name="test_write_f64"),
        ],
    )
    def test_write_raster_map(self, grass_i: GrassInterface, test_case: TestCase):
        rng = np.random.default_rng()
        region = grass_i.get_region()
        if test_case.g_dtype == "CELL":
            np_array_good = rng.integers(
                0,
                255,
                size=(region.rows, region.cols),
                dtype=test_case.np_dtype,
            )
            np_array_bad = rng.integers(0, 255, size=(5, 2), dtype=test_case.np_dtype)
        else:
            np_array_bad = rng.random(size=(20, 23), dtype=test_case.np_dtype)
            np_array_good = rng.random(
                size=(region.rows, region.cols), dtype=test_case.np_dtype
            )
        with pytest.raises(ValueError):
            grass_i.write_raster_map(np_array_bad, test_case.map_name)

        try:
            grass_i.write_raster_map(np_array_good, test_case.map_name)
            map_info = gs.parse_command(
                "r.info", flags="g", map=f"{test_case.map_name}@PERMANENT"
            )
            assert map_info["rows"] == str(region.rows)
            assert map_info["cols"] == str(region.cols)
            assert map_info["datatype"] == test_case.g_dtype
            map_univar = gs.parse_command("r.univar", flags="g", map=test_case.map_name)
            float(map_univar["mean"]) == np_array_good.mean()
            float(map_univar["min"]) == np_array_good.min()
            float(map_univar["max"]) == np_array_good.max()
        finally:
            gs.run_command(
                "g.remove", flags="f", type="raster", name=test_case.map_name
            )

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCase(np_dtype=np.uint8, g_dtype="CELL", map_name="test_write_int"),
            TestCase(np_dtype=np.float32, g_dtype="FCELL", map_name="test_write_f32"),
            TestCase(np_dtype=np.float64, g_dtype="DCELL", map_name="test_write_f64"),
        ],
    )
    def test_write_raster3d_map(self, grass_i: GrassInterface, test_case: TestCase):
        rng = np.random.default_rng()
        region = grass_i.get_region()
        if test_case.g_dtype == "CELL":
            np_array_good = rng.integers(
                0,
                255,
                size=(region.depths, region.rows3, region.cols3),
                dtype=test_case.np_dtype,
            )
            np_array_bad = rng.integers(0, 255, size=(5, 2), dtype=test_case.np_dtype)
        else:
            np_array_bad = rng.random(size=(20, 23), dtype=test_case.np_dtype)
            np_array_good = rng.random(
                size=(region.depths, region.rows3, region.cols3),
                dtype=test_case.np_dtype,
            )
        with pytest.raises(ValueError):
            grass_i.write_raster3d_map(np_array_bad, test_case.map_name)

        try:
            grass_i.write_raster3d_map(np_array_good, test_case.map_name)
            map_info = gs.parse_command(
                "r3.info", flags="g", map=f"{test_case.map_name}@PERMANENT"
            )
            assert map_info["rows"] == str(region.rows3)
            assert map_info["cols"] == str(region.cols3)
            assert map_info["depths"] == str(region.depths)
            map_univar = gs.parse_command(
                "r3.univar", flags="g", map=f"{test_case.map_name}@PERMANENT"
            )
            float(map_univar["mean"]) == np_array_good.mean()
            float(map_univar["min"]) == np_array_good.min()
            float(map_univar["max"]) == np_array_good.max()
        finally:
            gs.run_command(
                "g.remove", flags="f", type="raster_3d", name=test_case.map_name
            )

    def test_register_maps_in_stds(self, grass_i):
        rng = np.random.default_rng()
        region = grass_i.get_region()
        np_array = rng.random(size=(region.rows, region.cols), dtype="float32")
        grass_i.write_raster_map(np_array, "test_temporal_map1")
        grass_i.write_raster_map(np_array, "test_temporal_map2")
        maps_list = [
            ("test_temporal_map1", datetime(2023, 1, 1)),
            ("test_temporal_map2", datetime(2023, 2, 1)),
        ]
        stds_name = "test_stds"
        grass_i.register_maps_in_stds(
            stds_title="test_stds_title",
            stds_name=stds_name,
            stds_desc="test description of a STRDS",
            semantic="mean",
            map_list=maps_list,
            t_type="absolute",
            stds_type="strds",
        )
        strds_info = gs.parse_command(
            "t.info",
            flags="g",
            type="strds",
            input=f"{stds_name}@PERMANENT",
        )
        assert strds_info["name"] == stds_name
        assert strds_info["mapset"] == "PERMANENT"
        assert strds_info["id"] == f"{stds_name}@PERMANENT"
        assert strds_info["semantic_type"] == "mean"
        assert strds_info["temporal_type"] == "absolute"
        assert strds_info["number_of_maps"] == "2"
        # remove extra single quotes from the returned string
        assert strds_info["start_time"].strip("'") == str(datetime(2023, 1, 1))
        assert strds_info["end_time"].strip("'") == str(datetime(2023, 2, 1))
        # clean-up
        for map_name in ["test_temporal_map1", "test_temporal_map2"]:
            gs.run_command("g.remove", flags="f", type="raster", name=map_name)
        gs.run_command("t.remove", type="strds", input=stds_name, flags="f")
