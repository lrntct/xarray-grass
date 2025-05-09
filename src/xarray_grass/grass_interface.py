import os
from collections import namedtuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Self

import numpy as np

# Needed to be able to import grass modules
import grass_session  # noqa: F401
import grass.script as gscript
import grass.pygrass.utils as gutils
from grass.pygrass.gis.region import Region
from grass.pygrass import raster as graster
import grass.temporal as tgis


gscript.core.set_raise_on_error(True)


@dataclass
class GrassConfig:
    gisdb: str | Path
    project: str | Path
    mapset: str | Path
    grassbin: str | Path


class GrassInterface(object):
    """Interface to GRASS GIS for reading and writing raster data."""

    strds_cols = ["id", "start_time", "end_time"]
    MapData = namedtuple("MapData", strds_cols)
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

    def __init__(self, region_id: str | None = None, overwrite: bool = False):
        # Check if in a GRASS session
        if "GISRC" not in os.environ:
            raise RuntimeError("GRASS session not set.")
        self.overwrite = overwrite
        # Set region
        self.region_id = region_id
        if self.region_id:
            gscript.use_temp_region()
            gscript.run_command("g.region", region=region_id)
        self.region = Region()
        self.xr = self.region.cols
        self.yr = self.region.rows

    @staticmethod
    def format_id(name: str) -> str:
        """Take a map or stds name as input
        and return a fully qualified name, i.e. including mapset
        """
        if "@" in name:
            return name
        else:
            return "@".join((name, gutils.getenv("MAPSET")))

    @staticmethod
    def name_is_stds(name: str) -> bool:
        """return True if the name given as input is a registered strds
        False if not
        """
        # make sure temporal module is initialized
        tgis.init()
        return bool(tgis.SpaceTimeRasterDataset(name).is_in_db())

    @staticmethod
    def name_is_map(map_id):
        """return True if the given name is a map in the grass database
        False if not
        """
        return bool(gscript.find_file(name=map_id, element="cell").get("file"))

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

    def has_mask(self) -> bool:
        """Return True if the mapset has a mask, False otherwise."""
        return bool(gscript.read_command("g.list", type="raster", pattern="MASK"))

    def read_raster_map(self, rast_name: str) -> np.ndarray:
        """Read a GRASS raster and return a numpy array"""
        with graster.RasterRow(rast_name, mode="r") as rast:
            array = np.array(rast)
        return array

    def write_raster_map(self, arr: np.ndarray, rast_name: str) -> Self:
        mtype: str = self.grass_dtype(arr.dtype)
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
        stds_id = self.format_id(stds_name)
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
            map_id = self.format_id(map_name)
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
