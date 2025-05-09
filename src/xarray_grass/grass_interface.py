import os
from collections import namedtuple
from dataclasses import dataclass
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
