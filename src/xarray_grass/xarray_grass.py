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

import numpy as np
from xarray.backends import BackendEntrypoint
from xarray.backends import BackendArray
import xarray as xr
import grass_session
from xarray_grass.grass_interface import GrassInterface


class GrassBackendEntrypoint(BackendEntrypoint):
    """
    Backend entry point for GRASS mapset."""

    open_dataset_parameters = ["filename_or_obj", "grass_object_name", "drop_variables"]
    description = "Open GRASS mapset in Xarray"
    url = "https://link_to/your_backend/documentation"  # TODO

    def open_dataset(
        self,
        filename_or_obj,
        *,
        grass_object_name,
        drop_variables=None,
        # other backend specific keyword arguments
        # `chunks` and `cache` DO NOT go here, they are handled by xarray
    ) -> xr.Dataset:
        return open_grass_mapset(
            filename_or_obj, grass_object_name, drop_variables=drop_variables
        )

    def guess_can_open(self, filename_or_obj) -> bool:
        """infer if the path is a GRASS mapset by searching for WIND and VAR files
        in the directory.
        """
        return is_grass_mapset(filename_or_obj)


def is_grass_mapset(filename_or_obj) -> bool:
    """
    Check if the given path is a GRASS mapset.
    """
    try:
        dirpath = Path(filename_or_obj)
    except TypeError:
        return False
    if dirpath.is_dir():
        wind_file = dirpath / Path("WIND")
        var_file = dirpath / Path("VAR")
        if wind_file.exists() and var_file.exists():
            return True
    return False


def open_grass_mapset(
    filename_or_obj, grass_object_name, drop_variables=None
) -> xr.Dataset:
    """
    Open a GRASS mapset and return an xarray dataset.
    TODO: add support for single map
    TODO: add support for whole mapset
    TODO: add support for 3D STRDS
    TODO: add proj4 string as an attribute
    """
    dirpath = Path(filename_or_obj)
    if not is_grass_mapset(dirpath):
        raise ValueError(f"{filename_or_obj} is not a GRASS mapset")
    mapset = dirpath.stem
    project = dirpath.parent.stem
    gisdb = dirpath.parent.parent
    with grass_session.Session(
        gisdb=str(gisdb), location=str(project), mapset=str(mapset)
    ):
        grass_i = GrassInterface()
        if grass_object_name not in grass_i.list_strds():
            raise ValueError(f"{grass_object_name} not an STRDS")
        else:
            data_array = open_grass_strds(grass_object_name, grass_i)
    return data_array.to_dataset()


def open_grass_strds(strds_name: str, grass_i: GrassInterface):
    """must be called from within a grass session
    TODO: add unit, description etc. as attributes
    TODO: lazy loading
    TODO: Make sure the coordinate represents what it should
    """
    lim_e = grass_i.reg_bbox["e"]
    lim_w = grass_i.reg_bbox["w"]
    lim_n = grass_i.reg_bbox["n"]
    lim_s = grass_i.reg_bbox["s"]
    dx = grass_i.dx
    dy = grass_i.dy
    x_coords = np.arange(start=lim_w, stop=lim_e, step=dx, dtype=np.float32)
    y_coords = np.arange(start=lim_s, stop=lim_n, step=dy, dtype=np.float32)
    is_latlon = grass_i.is_latlon()
    if is_latlon:
        dims = ["time", "latitude", "longitude"]
        coordinates = dict.fromkeys(dims)
        coordinates["longitude"] = x_coords
        coordinates["latitude"] = y_coords
    else:
        dims = ["time", "y", "x"]
        coordinates = dict.fromkeys(dims)
        coordinates["x"] = x_coords
        coordinates["y"] = y_coords
    map_list = grass_i.list_maps_in_strds(strds_name)
    array_list = []
    for map_data in map_list:
        coordinates["time"] = [map_data.start_time]
        ndarray = grass_i.read_raster_map(map_data.id)
        # add time dimension at the beginning
        ndarray = np.expand_dims(ndarray, axis=0)
        data_array = xr.DataArray(
            ndarray,
            coords=coordinates,
            dims=dims,
            name=grass_i.get_name_from_id(strds_name),
        )
        array_list.append(data_array)
    return xr.concat(array_list, dim="time")


class GrassBackendArray(BackendArray):
    """To implement lazy loading"""

    def __init__(
        self,
        shape,
        dtype,
        lock,
        # other backend specific keyword arguments
    ):
        self.shape = shape
        self.dtype = dtype
        self.lock = lock

    def __getitem__(self, key: xr.core.indexing.ExplicitIndexer) -> np.typing.ArrayLike:
        """takes in input an index and returns a NumPy array"""
        pass
