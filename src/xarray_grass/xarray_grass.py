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
from typing import Iterable

import numpy as np
from xarray.backends import BackendEntrypoint
from xarray.backends import BackendArray
import xarray as xr
import grass_session  # noqa: F401
from xarray_grass.grass_interface import GrassInterface


class GrassBackendEntrypoint(BackendEntrypoint):
    """
    Backend entry point for GRASS mapset."""

    open_dataset_parameters = [
        "filename_or_obj",
        "raster",
        "raster_3d",
        "strds",
        "str3ds",
        "drop_variables",
    ]
    description = "Open a GRASS mapset in Xarray"
    url = "https://link_to/your_backend/documentation"  # TODO

    def open_dataset(
        self,
        filename_or_obj,
        *,
        raster: str | Iterable[str] = [],
        raster_3d: str | Iterable[str] = [],
        strds: str | Iterable[str] = [],
        str3ds: str | Iterable[str] = [],
        drop_variables: Iterable[str],
        # other backend specific keyword arguments
        # `chunks` and `cache` DO NOT go here, they are handled by xarray
    ) -> xr.Dataset:
        """Open GRASS project or mapset as an xarray.Dataset.
        TODO: add support for whole project.
        """
        open_func_params = dict(
            raster_list=raster,
            raster_3d_list=raster_3d,
            strds_list=strds,
            str3ds_list=str3ds,
        )
        if not any([raster, raster_3d, strds, str3ds]):
            # list all the maps in the mapset / project
            pass
        else:
            # Format str inputs into list
            for object_type, elem in open_func_params.items():
                if isinstance(elem, str):
                    open_func_params[object_type] = [elem]
                elif elem is None:
                    open_func_params[object_type] = []
                else:
                    open_func_params[object_type] = list(elem)
        # drop requested variables
        if drop_variables is not None:
            for object_type, grass_obj_name_list in open_func_params.items():
                open_func_params[object_type] = [
                    name for name in grass_obj_name_list if name not in drop_variables
                ]

        return open_grass_maps(filename_or_obj, **open_func_params)

    def guess_can_open(self, filename_or_obj) -> bool:
        """infer if the path is a GRASS mapset."""
        return dir_is_grass_mapset(filename_or_obj) or dir_is_grass_project(
            filename_or_obj
        )


def dir_is_grass_mapset(filename_or_obj: str | Path) -> bool:
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


def dir_is_grass_project(filename_or_obj: str | Path) -> bool:
    """Return True if a subdir named PERMANENT is present."""
    try:
        dirpath = Path(filename_or_obj)
    except TypeError:
        return False
    if dirpath.is_dir():
        return (dirpath / Path("PERMANENT")).is_dir()
    else:
        return False


def open_grass_maps(
    filename_or_obj: str | Path,
    raster_list: Iterable[str] = None,
    raster_3d_list: Iterable[str] = None,
    strds_list: Iterable[str] = None,
    str3ds_list: Iterable[str] = None,
    raise_on_not_found: bool = True,
) -> xr.Dataset:
    """
    Open a GRASS mapset and return an xarray dataset.
    TODO: add support for single map
    TODO: add support for whole mapset
    TODO: add support for 3D STRDS
    """
    dirpath = Path(filename_or_obj)
    if not dir_is_grass_mapset(dirpath):
        raise ValueError(f"{filename_or_obj} is not a GRASS mapset")
    mapset = dirpath.stem
    project = dirpath.parent.stem
    gisdb = dirpath.parent.parent
    with grass_session.Session(
        gisdb=str(gisdb), location=str(project), mapset=str(mapset)
    ):
        gi = GrassInterface()
        # Open all given maps and identify non-existent data
        # Need refactoring
        not_found = {k: [] for k in ["raster", "raster_3d", "strds", "str3ds"]}
        data_array_list = []
        for raster_map_name in raster_list:
            if not gi.name_is_raster(raster_map_name):
                not_found["raster"].append(raster_map_name)
                continue
            data_array = open_grass_raster(raster_map_name, gi)
            data_array_list.append(data_array)
        for raster_3d_map_name in raster_3d_list:
            if not gi.name_is_raster_3d(raster_3d_map_name):
                not_found["raster_3d"].append(raster_3d_map_name)
                continue
            data_array = open_grass_raster_3d(raster_3d_map_name, gi)
            data_array_list.append(data_array)
        for strds_name in strds_list:
            if not gi.name_is_strds(strds_name):
                not_found["strds"].append(strds_name)
                continue
            data_array = open_grass_strds(strds_name, gi)
            data_array_list.append(data_array)
        for str3ds_name in str3ds_list:
            if not gi.name_is_str3ds(str3ds_name):
                not_found["str3ds"].append(str3ds_name)
                continue
            data_array = open_grass_str3ds(str3ds_name, gi)
            data_array_list.append(data_array)
        if raise_on_not_found and any(not_found.values()):
            raise ValueError(f"Objects not found: {not_found}")
        data_array_list = [da for da in data_array_list if isinstance(da, xr.DataArray)]
        dataset = xr.merge(data_array_list)
        dataset.attrs["crs"] = gi.get_proj_str()
    return dataset


def get_coordinates(grass_i: GrassInterface, raster_3d: bool) -> dict:
    """return xarray coordinates from GRASS region."""
    current_region = grass_i.get_region()
    print(current_region)
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
    # In the exported DataArray, coordinates are at the center of the cell
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


def open_grass_raster(raster_name: str, grass_i: GrassInterface) -> xr.DataArray:
    """Open a single raster map."""
    x_coords, y_coords, _ = get_coordinates(grass_i, raster_3d=False).values()
    is_latlon = grass_i.is_latlon()
    if is_latlon:
        dims = ["latitude", "longitude"]
        coordinates = dict.fromkeys(dims)
        coordinates["longitude"] = x_coords
        coordinates["latitude"] = y_coords
    else:
        dims = ["y", "x"]
        coordinates = dict.fromkeys(dims)
        coordinates["x"] = x_coords
        coordinates["y"] = y_coords
    raster_array = grass_i.read_raster_map(raster_name)
    data_array = xr.DataArray(
        raster_array,
        coords=coordinates,
        dims=dims,
        name=grass_i.get_name_from_id(raster_name),
    )
    return data_array


def open_grass_raster_3d(raster_3d_name: str, grass_i: GrassInterface) -> xr.DataArray:
    """Open a single 3D raster map."""
    x_coords, y_coords, z_coords = get_coordinates(grass_i, raster_3d=True).values()
    is_latlon = grass_i.is_latlon()
    if is_latlon:
        dims = ["z", "latitude", "longitude"]
        coordinates = dict.fromkeys(dims)
        coordinates["longitude"] = x_coords
        coordinates["latitude"] = y_coords
    else:
        dims = ["z", "y", "x"]
        coordinates = dict.fromkeys(dims)
        coordinates["x"] = x_coords
        coordinates["y"] = y_coords
    coordinates["z"] = z_coords
    raster_array = grass_i.read_raster3d_map(raster_3d_name)
    print(f"Raster3D shape: {raster_array.shape}")

    data_array = xr.DataArray(
        raster_array,
        coords=coordinates,
        dims=dims,
        name=grass_i.get_name_from_id(raster_3d_name),
    )
    return data_array


def open_grass_str3ds(str3ds_name: str, grass_i: GrassInterface) -> xr.DataArray:
    """Open a series of 3D raster maps.
    TODO: Figure out what to do when the z value of the maps is time."""
    pass


def open_grass_strds(strds_name: str, grass_i: GrassInterface) -> xr.DataArray:
    """must be called from within a grass session
    TODO: add unit, description etc. as attributes
    TODO: lazy loading
    """
    x_coords, y_coords, _ = get_coordinates(grass_i, raster_3d=False).values()
    is_latlon = grass_i.is_latlon()
    if is_latlon:
        dims = ["start_time", "latitude", "longitude"]
        coordinates = dict.fromkeys(dims)
        coordinates["longitude"] = x_coords
        coordinates["latitude"] = y_coords
    else:
        dims = ["start_time", "y", "x"]
        coordinates = dict.fromkeys(dims)
        coordinates["x"] = x_coords
        coordinates["y"] = y_coords
    map_list = grass_i.list_maps_in_strds(strds_name)
    array_list = []
    for map_data in map_list:
        coordinates["start_time"] = [map_data.start_time]
        coordinates["end_time"] = ("start_time", [map_data.end_time])
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
    return xr.concat(array_list, dim="start_time")


class GrassBackendArray(BackendArray):
    """For lazy loading"""

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
