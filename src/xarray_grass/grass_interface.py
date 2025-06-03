import os
from collections import namedtuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Self

import numpy as np

# Needed to import grass modules
import grass_session  # noqa: F401
import grass.script as gs
from grass.script import array as garray
import grass.pygrass.utils as gutils
from grass.pygrass import raster as graster
import grass.temporal as tgis


gs.core.set_raise_on_error(True)


@dataclass
class GrassConfig:
    gisdb: str | Path
    project: str | Path
    mapset: str | Path
    grassbin: str | Path


strds_cols = ["id", "start_time", "end_time"]
MapData = namedtuple("MapData", strds_cols)
strds_infos = [
    "id",
    "temporal_type",
    "time_unit",
    "start_time",
    "end_time",
    "time_granularity",
    "north",
    "south",
    "east",
    "west",
    "top",
    "bottom",
]
STRDSInfos = namedtuple("STRDSInfos", strds_infos)


class GrassInterface(object):
    """Interface to GRASS GIS for reading and writing raster data."""

    # datatype conversion between GRASS and numpy
    dtype_conv = {
        "FCELL": ("float16", "float32"),
        "DCELL": ("float_", "float64"),
        "CELL": (
            "bool_",
            "int_",
            "intc",
            "intp",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
        ),
    }

    def __init__(self, overwrite: bool = False):
        # Check if in a GRASS session
        if "GISRC" not in os.environ:
            raise RuntimeError("GRASS session not set.")
        self.overwrite = overwrite
        tgis.init()

    @staticmethod
    def get_gisenv() -> dict[str]:
        """Return the current GRASS environment."""
        return gs.gisenv()

    @staticmethod
    def get_accessible_mapsets() -> list[str]:
        """Return a list of accessible mapsets."""
        return gs.parse_command("g.mapsets", flags="p", format="json")["mapsets"]

    @staticmethod
    def get_region() -> namedtuple:
        """Return the current GRASS region."""
        region_raw = gs.parse_command("g.region", flags="g3")
        type_dict = {
            "projection": str,
            "zone": str,
            "n": float,
            "s": float,
            "w": float,
            "e": float,
            "t": float,
            "b": float,
            "nsres": float,
            "nsres3": float,
            "ewres": float,
            "ewres3": float,
            "tbres": float,
            "rows": int,
            "rows3": int,
            "cols": int,
            "cols3": int,
            "depths": int,
            "cells": int,
            "cells3": int,
        }
        RegionData = namedtuple("RegionData", region_raw.keys())
        region = {k: type_dict[k](v) for k, v in region_raw.items() if v is not None}
        region = RegionData(**region)
        return region

    @staticmethod
    def is_latlon():
        return gs.locn_is_latlong()

    @staticmethod
    def get_id_from_name(name: str) -> str:
        """Take a map or stds name as input
        and return a fully qualified name, i.e. including mapset
        """
        if "@" in name:
            return name
        else:
            return "@".join((name, gutils.getenv("MAPSET")))

    @staticmethod
    def get_name_from_id(input_string: str) -> str:
        """Take a map id and return a base name, i.e without mapset"""
        try:
            at_index = input_string.find("@")
        except AttributeError:
            raise TypeError(f"{input_string} not a string")
        if at_index != -1:
            return input_string[:at_index]
        else:
            return input_string

    def name_is_strds(self, name: str) -> bool:
        """return True if the name given as input is a registered strds
        False if not
        """
        # make sure temporal module is initialized
        tgis.init()
        strds_id = self.get_id_from_name(name)
        return bool(tgis.SpaceTimeRasterDataset(strds_id).is_in_db())

    def name_is_str3ds(self, name: str) -> bool:
        """return True if the name given as input is a registered str3ds
        False if not
        """
        # make sure temporal module is initialized
        tgis.init()
        str3ds_id = self.get_id_from_name(name)
        return bool(tgis.SpaceTimeRaster3DDataset(str3ds_id).is_in_db())

    def name_is_raster(self, raster_name: str) -> bool:
        """return True if the given name is a map in the grass database
        False if not
        """
        map_id = self.get_id_from_name(raster_name)
        return bool(gs.find_file(name=map_id, element="raster").get("file"))

    def name_is_raster_3d(self, raster3d_name: str) -> bool:
        """return True if the given name is a 3D raster in the grass database."""
        map_id = self.get_id_from_name(raster3d_name)
        return bool(gs.find_file(name=map_id, element="raster_3d").get("file"))

    @staticmethod
    def get_proj_str() -> str:
        return gs.read_command("g.proj", flags="jf").replace("\n", "")

    @staticmethod
    def get_crs_wkt_str() -> str:
        return gs.read_command("g.proj", flags="wf").replace("\n", "")

    def grass_dtype(self, dtype: str) -> str:
        if dtype in self.dtype_conv["DCELL"]:
            mtype = "DCELL"
        elif dtype in self.dtype_conv["CELL"]:
            mtype = "CELL"
        elif dtype in self.dtype_conv["FCELL"]:
            mtype = "FCELL"
        else:
            raise ValueError("datatype incompatible with GRASS!")
        return mtype

    @staticmethod
    def has_mask() -> bool:
        """Return True if the mapset has a mask, False otherwise."""
        return bool(gs.read_command("g.list", type="raster", pattern="MASK"))

    @staticmethod
    def list_strds(mapset: str = None) -> list[str]:
        if mapset:
            return tgis.tlist_grouped("strds")[mapset]
        else:
            return tgis.tlist("strds")

    @staticmethod
    def list_str3ds(mapset: str = None) -> list[str]:
        if mapset:
            return tgis.tlist_grouped("str3ds")[mapset]
        else:
            return tgis.tlist("str3ds")

    @staticmethod
    def list_raster(mapset: str = None) -> list[str]:
        """List raster maps in the given mapset"""
        return gs.list_strings("raster", mapset=mapset)

    @staticmethod
    def list_raster3d(mapset: str = None) -> list[str]:
        return gs.list_strings("raster_3d", mapset=mapset)

    def list_grass_objects(self, mapset: str = None) -> dict[list[str]]:
        """Return all GRASS objects in a given mapset."""
        objects_dict = {}
        objects_dict["raster"] = self.list_raster(mapset)
        objects_dict["raster3d"] = self.list_raster3d(mapset)
        objects_dict["strds"] = self.list_strds(mapset)
        objects_dict["str3ds"] = self.list_str3ds(mapset)
        return objects_dict

    def get_strds_infos(self, strds_name) -> STRDSInfos:
        strds_id = self.get_id_from_name(strds_name)
        strds = tgis.open_stds.open_old_stds(strds_id, "strds")
        temporal_type = strds.get_temporal_type()
        if temporal_type == "relative":
            start_time, end_time, time_unit = strds.get_relative_time()
        elif temporal_type == "absolute":
            start_time, end_time = strds.get_absolute_time()
            time_unit = None
        else:
            raise ValueError(f"Unknown temporal type for {strds_id}: {temporal_type}")
        granularity = strds.get_granularity()
        spatial_extent = strds.get_spatial_extent_as_tuple()
        infos = STRDSInfos(
            id=strds_id,
            temporal_type=temporal_type,
            time_unit=time_unit,
            start_time=start_time,
            end_time=end_time,
            time_granularity=granularity,
            north=spatial_extent[0],
            south=spatial_extent[1],
            east=spatial_extent[2],
            west=spatial_extent[3],
            top=spatial_extent[4],
            bottom=spatial_extent[5],
        )
        return infos

    def list_maps_in_str3ds(self, strds_name: str) -> list[MapData]:
        strds = tgis.open_stds.open_old_stds(strds_name, "str3ds")
        maplist = strds.get_registered_maps(
            columns=",".join(strds_cols), order="start_time"
        )
        # check if every map exist
        maps_not_found = [m[0] for m in maplist if not self.name_is_raster_3d(m[0])]
        if any(maps_not_found):
            err_msg = "STR3DS <{}>: Can't find following maps: {}"
            str_lst = ",".join(maps_not_found)
            raise RuntimeError(err_msg.format(strds_name, str_lst))
        return [MapData(*i) for i in maplist]

    def list_maps_in_strds(self, strds_name: str) -> list[MapData]:
        strds = tgis.open_stds.open_old_stds(strds_name, "strds")
        maplist = strds.get_registered_maps(
            columns=",".join(strds_cols), order="start_time"
        )
        # check if every map exist
        maps_not_found = [m[0] for m in maplist if not self.name_is_raster(m[0])]
        if any(maps_not_found):
            err_msg = "STRDS <{}>: Can't find following maps: {}"
            str_lst = ",".join(maps_not_found)
            raise RuntimeError(err_msg.format(strds_name, str_lst))
        return [MapData(*i) for i in maplist]

    @staticmethod
    def read_raster_map(rast_name: str) -> np.ndarray:
        """Read a GRASS raster and return a numpy array"""
        with graster.RasterRow(rast_name, mode="r") as rast:
            array = np.array(rast)
        return array

    @staticmethod
    def read_raster3d_map(rast3d_name: str) -> np.ndarray:
        """Read a GRASS 3D raster and return a numpy array"""
        array = garray.array3d(mapname=rast3d_name)
        return array

    def write_raster_map(self, arr: np.ndarray, rast_name: str) -> Self:
        mtype: str = self.grass_dtype(arr.dtype)
        region = self.get_region()
        region_shape = (region.rows, region.cols)
        if region_shape != arr.shape:
            raise ValueError(
                f"Cannot write an array of shape {arr.shape} into "
                f"a GRASS region of size {region_shape}"
            )
        with graster.RasterRow(
            rast_name, mode="w", mtype=mtype, overwrite=self.overwrite
        ) as newraster:
            newrow = graster.Buffer((arr.shape[1],), mtype=mtype)
            for row in arr:
                newrow[:] = row[:]
                newraster.put_row(newrow)

        return self

    def register_maps_in_stds(
        self,
        stds_title: str,
        stds_name: str,
        stds_desc: str,
        map_list: list[tuple[str, datetime | timedelta]],
        semantic: str,
        t_type: str,
    ) -> Self:
        """Create a STDS, create one mapdataset for each map and
        register them in the temporal database.
        TODO: add support for units other than seconds
        """
        # create stds
        stds_id = self.get_id_from_name(stds_name)
        stds_desc = ""
        stds = tgis.open_new_stds(
            name=stds_id,
            type="strds",
            temporaltype=t_type,
            title=stds_title,
            descr=stds_desc,
            semantic=semantic,
            dbif=None,
            overwrite=self.overwrite,
        )

        # create MapDataset objects list
        map_dts_lst = []
        for map_name, map_time in map_list:
            # create MapDataset
            map_id = self.get_id_from_name(map_name)
            map_dts = tgis.RasterDataset(map_id)
            # load spatial data from map
            map_dts.load()
            # set time
            if t_type == "relative":
                if not isinstance(map_time, timedelta):
                    raise TypeError("relative time requires a timedelta object.")
                rel_time = map_time.total_seconds()
                map_dts.set_relative_time(rel_time, None, "seconds")
            elif t_type == "absolute":
                if not isinstance(map_time, datetime):
                    raise TypeError("absolute time requires a datetime object.")
                map_dts.set_absolute_time(start_time=map_time)
            else:
                raise ValueError(
                    f"Invalid temporal type {t_type}, must be 'relative' or 'absolute'"
                )
            # populate the list of MapDataset objects
            map_dts_lst.append(map_dts)
        # Finally register the maps
        t_unit = {"relative": "seconds", "absolute": ""}
        tgis.register.register_map_object_list(
            type="raster",
            map_list=map_dts_lst,
            output_stds=stds,
            delete_empty=True,
            unit=t_unit[t_type],
        )
        return self
