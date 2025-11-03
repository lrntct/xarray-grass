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

import os
import math
from collections import namedtuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Self, Optional

import numpy as np
import pandas as pd


# Needed to import grass modules
import grass.script as gs
from grass.script import array as garray
import grass.pygrass.utils as gutils
from grass.pygrass import raster as graster
from grass.pygrass.raster.abstract import Info, RasterAbstractBase
import grass.temporal as tgis

from xarray_grass.coord_utils import (
    region_type_dict,
    RegionData,
)

gs.core.set_raise_on_error(True)


@dataclass
class GrassConfig:
    gisdb: str | Path
    project: str | Path
    mapset: str | Path
    grassbin: str | Path


strds_cols = ["id", "start_time", "end_time"]
MapData = namedtuple("MapData", strds_cols + ["dtype"])

strds_infos = [
    "id",
    "title",
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
        if self.overwrite:
            os.environ["GRASS_OVERWRITE"] = "1"
        else:
            os.environ["GRASS_OVERWRITE"] = "0"
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

        region = {
            k: region_type_dict[k](v) for k, v in region_raw.items() if v is not None
        }
        region = RegionData(**region)
        return region

    @staticmethod
    def set_region(region_data: RegionData) -> None:
        # 2D region
        if region_data.tbres is None:
            if not all(
                [
                    region_data.n,
                    region_data.s,
                    region_data.e,
                    region_data.w,
                    region_data.nsres,
                    region_data.ewres,
                ]
            ):
                raise ValueError(
                    "n, s, e, w, nsres and ewres must be set for 2D regions."
                )
            gs.run_command(
                "g.region",
                flags="o",
                n=region_data.n,
                s=region_data.s,
                e=region_data.e,
                w=region_data.w,
                nsres=region_data.nsres,
                ewres=region_data.ewres,
            )
        # 3D region
        else:
            # TODO: remove when grass 8.5 is released
            tolerance = 1e-9
            if not math.isclose(
                region_data.nsres3, region_data.ewres3, rel_tol=tolerance
            ):
                raise ValueError(
                    f"ewres3 and nsres3 must be equal (within {tolerance} relative tolerance)."
                )
            gs.run_command(
                "g.region",
                flags="o",
                n=region_data.n,
                s=region_data.s,
                e=region_data.e,
                w=region_data.w,
                t=region_data.t,
                b=region_data.b,
                res3=region_data.nsres3,
                # nsres3=region_data.nsres3,
                # ewres3=region_data.ewres3,
                tbres=region_data.tbres,
            )

    @staticmethod
    def is_latlon():
        return gs.locn_is_latlong()

    def is_xy(self):
        """return True if the location is neither projected or latlon"""
        proj_code = gs.parse_command("g.region", flags="pug")["projection"]
        if int(proj_code) == 0:
            return True
        else:
            return False

    def get_spatial_units(self):
        if self.is_xy:
            return None
        else:
            return gs.parse_command("g.proj", flags="g")["units"]

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
        """return True if the given name is a raster map in the grass database."""
        # Using pygrass instead of gscript is at least 40x faster
        map_id = self.get_id_from_name(raster_name)
        map_object = RasterAbstractBase(map_id)
        return map_object.exist()

    def name_is_raster_3d(self, raster3d_name: str) -> bool:
        """return True if the given name is a 3D raster in the grass database."""
        map_id = self.get_id_from_name(raster3d_name)
        return bool(gs.find_file(name=map_id, element="raster_3d").get("file"))

    @staticmethod
    def get_crs_wkt_str() -> str:
        return gs.read_command("g.proj", flags="wf").replace("\n", "")

    def grass_dtype(self, dtype: str) -> str:
        """Takes a numpy-style data-type description string,
        and return a GRASS data type string."""
        if dtype in self.dtype_conv["DCELL"]:
            mtype = "DCELL"
        elif dtype in self.dtype_conv["CELL"]:
            mtype = "CELL"
        elif dtype in self.dtype_conv["FCELL"]:
            mtype = "FCELL"
        else:
            raise ValueError(f"datatype '{dtype}' incompatible with GRASS!")
        return mtype

    @staticmethod
    def numpy_dtype(mtype: str) -> np.dtype:
        if mtype == "CELL":
            dtype = np.dtype("int64")
        elif mtype == "FCELL":
            dtype = np.dtype("float32")
        elif mtype == "DCELL":
            dtype = np.dtype("float64")
        else:
            raise ValueError(f"Unknown GRASS data type: {mtype}")
        return dtype

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
        objects_dict["raster_3d"] = self.list_raster3d(mapset)
        objects_dict["strds"] = self.list_strds(mapset)
        objects_dict["str3ds"] = self.list_str3ds(mapset)
        return objects_dict

    @staticmethod
    def get_raster_info(raster_id: str) -> Info:
        result = gs.parse_command("r.info", map=raster_id, flags="ge")
        # Strip quotes from string values (r.info returns quoted strings)
        return {k: v.strip('"') if isinstance(v, str) else v for k, v in result.items()}

    @staticmethod
    def get_raster3d_info(raster3d_id):
        result = gs.parse_command("r3.info", map=raster3d_id, flags="gh")
        # Strip quotes from string values (r3.info -gh returns quoted strings)
        return {k: v.strip('"') if isinstance(v, str) else v for k, v in result.items()}

    def get_stds_infos(self, strds_name, stds_type) -> STRDSInfos:
        strds_id = self.get_id_from_name(strds_name)
        if stds_type not in ["strds", "str3ds"]:
            raise ValueError(
                f"Invalid strds type: {stds_type}. Must be 'strds' or 'str3ds'."
            )
        strds = tgis.open_stds.open_old_stds(strds_id, stds_type)
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
            title=strds.metadata.get_title(),
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
        tuple_list = []
        for i in maplist:
            mtype = self.get_raster3d_info(i[0])["datatype"]
            dtype = self.numpy_dtype(mtype)
            tuple_list.append(MapData(*i, dtype=dtype))
        return tuple_list

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
        tuple_list = []
        for i in maplist:
            dtype = self.numpy_dtype(Info(i[0]).mtype)
            tuple_list.append(MapData(*i, dtype=dtype))
        return tuple_list

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
        region = self.get_region()
        region_shape = (region.rows, region.cols)
        if region_shape != arr.shape:
            raise ValueError(
                f"Cannot write an array of shape {arr.shape} into "
                f"a GRASS region of size {region_shape}"
            )
        # Write with array interface
        map2d = garray.array(dtype=arr.dtype)
        map2d[:] = arr
        map2d.write(mapname=rast_name, overwrite=self.overwrite)
        return self

    def write_raster3d_map(self, arr: np.ndarray, rast_name: str) -> Self:
        region = self.get_region()
        region_shape = (region.depths, region.rows3, region.cols3)
        if region_shape != arr.shape:
            raise ValueError(
                f"Cannot write an array of shape {arr.shape} into "
                f"a GRASS region of size {region_shape}"
            )
        # Write with array interface
        map3d = garray.array3d(dtype=arr.dtype)
        map3d[:] = arr
        map3d.write(mapname=rast_name, overwrite=self.overwrite)
        return self

    def register_maps_in_stds(
        self,
        stds_title: str,
        stds_name: str,
        stds_desc: str,
        map_list: list[tuple[str, datetime | timedelta]],
        semantic: str,
        t_type: str,
        stds_type: str,
        time_unit: Optional[str] = None,
    ) -> Self:
        """Create a STDS, create one mapdataset for each map and
        register them in the temporal database.
        """
        # create stds
        stds_id = self.get_id_from_name(stds_name)
        stds_desc = ""
        stds = tgis.open_new_stds(
            name=stds_id,
            type=stds_type,
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
            if stds_type == "str3ds":
                map_dts = tgis.Raster3DDataset(map_id)
            else:
                map_dts = tgis.RasterDataset(map_id)
            # load spatial data from map
            map_dts.load()
            # set time
            if t_type == "relative":
                if not isinstance(map_time, timedelta):
                    raise TypeError("relative time requires a timedelta object.")
                if not time_unit:
                    raise TypeError("relative time requires a time_unit.")
                # Convert timedelta to numeric value in the specified unit
                rel_time = map_time / pd.Timedelta(1, unit=time_unit)
                map_dts.set_relative_time(rel_time, None, time_unit)
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
        # Use provided unit for relative time, empty string for absolute
        if t_type == "relative":
            t_unit = time_unit
        else:
            t_unit = ""

        map_type = "raster"
        if stds_type == "str3ds":
            map_type = "raster_3d"
        tgis.register.register_map_object_list(
            type=map_type,
            map_list=map_dts_lst,
            output_stds=stds,
            delete_empty=True,
            unit=t_unit,
        )
        return self

    def get_coordinates(self, raster_3d: bool) -> dict[str : np.ndarray]:
        """return np.ndarray of coordinates from the GRASS region."""
        current_region = self.get_region()
        lim_e = current_region.e
        lim_w = current_region.w
        lim_n = current_region.n
        lim_s = current_region.s
        lim_t = current_region.t
        lim_b = current_region.b
        dz = current_region.tbres
        if raster_3d:
            dx = current_region.ewres3
            dy = current_region.nsres3
        else:
            dx = current_region.ewres
            dy = current_region.nsres
        # GRASS limits are at the edge of the region.
        # In the exported arrays, coordinates are at the center of the cell
        # Stop not changed to include it in the range
        start_w = lim_w + dx / 2
        stop_e = lim_e
        start_s = lim_s + dy / 2
        stop_n = lim_n
        start_b = lim_b + dz / 2
        stop_t = lim_t
        x_coords = np.arange(start=start_w, stop=stop_e, step=dx, dtype=np.float32)
        y_coords = np.arange(start=start_s, stop=stop_n, step=dy, dtype=np.float32)
        z_coords = np.arange(start=start_b, stop=stop_t, step=dz, dtype=np.float32)
        return {"x": x_coords, "y": y_coords, "z": z_coords}
