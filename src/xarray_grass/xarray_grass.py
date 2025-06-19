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

import os
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
    url = "https://github.com/lrntct/xarray-grass"

    def open_dataset(
        self,
        filename_or_obj,
        *,
        raster: str | Iterable[str] = [],
        raster_3d: str | Iterable[str] = [],
        strds: str | Iterable[str] = [],
        str3ds: str | Iterable[str] = [],
        drop_variables: Iterable[str],
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
            # Load the whole mapset.
            # If a map is in a STDS, do not load it as a single map.
            gi = GrassInterface()
            grass_objects = gi.list_grass_objects()
            # strds
            rasters_in_strds = []
            for strds_name in grass_objects["strds"]:
                maps_in_strds = gi.list_maps_in_strds(strds_name)
                rasters_in_strds.extend([map_data.id for map_data in maps_in_strds])
                open_func_params["strds_list"].append(strds_name)
            raster3ds_in_str3ds = []
            # str3ds
            for str3ds_name in grass_objects["str3ds"]:
                maps_in_str3ds = gi.list_maps_in_str3ds(str3ds_name)
                raster3ds_in_str3ds.extend([map_data.id for map_data in maps_in_str3ds])
                open_func_params["str3ds_list"].append(str3ds_name)
            # rasters not in strds
            open_func_params["raster_list"] = [
                name for name in grass_objects["raster"] if name not in rasters_in_strds
            ]
            # rasters 3d not in str3ds
            open_func_params["raster_3d_list"] = [
                name
                for name in grass_objects["raster_3d"]
                if name not in raster3ds_in_str3ds
            ]
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
        """infer if the path is a GRASS mapset.
        TODO: add support for whole project."""
        return dir_is_grass_mapset(filename_or_obj)


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
        # A newly created mapset might only have WIND, VAR appears later.
        if wind_file.exists():
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


def get_coordinates(grass_i: GrassInterface, raster_3d: bool) -> dict:
    """return xarray coordinates from GRASS region."""
    current_region = grass_i.get_region()
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
    """
    dirpath = Path(filename_or_obj)
    if not dir_is_grass_mapset(dirpath):
        raise ValueError(f"{filename_or_obj} is not a GRASS mapset")
    mapset = dirpath.stem
    project = dirpath.parent.stem
    gisdb = dirpath.parent.parent

    # Check if we're already in a GRASS session
    session = None
    if "GISRC" not in os.environ:
        # No existing session, create a new one
        session = grass_session.Session(
            gisdb=str(gisdb), location=str(project), mapset=str(mapset)
        )
        session.__enter__()
        gi = GrassInterface()

    else:
        # We're in an existing session, check if it matches the requested path
        gi = GrassInterface()
        gisenv = gi.get_gisenv()
        current_gisdb = gisenv["GISDBASE"]
        current_location = gisenv["LOCATION_NAME"]
        accessible_mapsets = gi.get_accessible_mapsets()

        requested_path = Path(gisdb) / Path(project)
        current_path = Path(current_gisdb) / Path(current_location)

        if requested_path != current_path or str(mapset) not in accessible_mapsets:
            raise ValueError(
                f"Cannot access {gisdb}/{project}/{mapset} "
                f"from current GRASS session in project "
                f"{current_gisdb}/{current_location}. "
                f"Accessible mapsets: {accessible_mapsets}."
            )
    try:
        # Configuration for processing different GRASS map types
        map_processing_configs = [
            {
                "input_list": raster_list,
                "existence_check_method": gi.name_is_raster,
                "open_function": open_grass_raster,
                "not_found_key": "raster",
            },
            {
                "input_list": raster_3d_list,
                "existence_check_method": gi.name_is_raster_3d,
                "open_function": open_grass_raster_3d,
                "not_found_key": "raster_3d",
            },
            {
                "input_list": strds_list,
                "existence_check_method": gi.name_is_strds,
                "open_function": open_grass_strds,
                "not_found_key": "strds",
            },
            {
                "input_list": str3ds_list,
                "existence_check_method": gi.name_is_str3ds,
                "open_function": open_grass_str3ds,
                "not_found_key": "str3ds",
            },
        ]
        # Open all given maps and identify non-existent data
        not_found = {config["not_found_key"]: [] for config in map_processing_configs}
        data_array_list = []
        for config in map_processing_configs:
            for map_name in config["input_list"]:
                if not config["existence_check_method"](map_name):
                    not_found[config["not_found_key"]].append(map_name)
                    continue
                data_array = config["open_function"](map_name, gi)
                data_array_list.append(data_array)
        if raise_on_not_found and any(not_found.values()):
            raise ValueError(f"Objects not found: {not_found}")
    finally:
        if session is not None:
            session.__exit__(None, None, None)

    dataset = xr.merge(data_array_list)
    dataset.attrs["crs_wkt"] = gi.get_crs_wkt_str()
    dataset.attrs["Conventions"] = "CF-1.13-draft"
    # dataset.attrs["title"] = ""
    # dataset.attrs["history"] = ""
    # dataset.attrs["source"] = ""
    # dataset.attrs["references"] = ""
    # dataset.attrs["institution"] = ""
    # dataset.attrs["comment"] = ""
    return dataset


def open_grass_raster(raster_name: str, grass_i: GrassInterface) -> xr.DataArray:
    """Open a single raster map."""
    x_coords, y_coords, _ = get_coordinates(grass_i, raster_3d=False).values()
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
    # Add CF attributes
    r_infos = grass_i.get_raster_info(raster_name)
    print(f"{r_infos=}")
    da_with_attrs = set_cf_coordinates(data_array, gi=grass_i, is_3d=False)
    da_with_attrs.attrs["long_name"] = r_infos.get("title", "")
    da_with_attrs.attrs["source"] = ",".join([r_infos["source1"], r_infos["source2"]])
    da_with_attrs.attrs["units"] = r_infos.get("units", "")
    da_with_attrs.attrs["comment"] = r_infos.get("comments", "")
    # CF attributes "institution" and "references"
    # Do not correspond to a direct GRASS value.
    return da_with_attrs


def open_grass_raster_3d(raster_3d_name: str, grass_i: GrassInterface) -> xr.DataArray:
    """Open a single 3D raster map."""
    x_coords, y_coords, z_coords = get_coordinates(grass_i, raster_3d=True).values()
    dims = ["z", "y_3d", "x_3d"]
    coordinates = dict.fromkeys(dims)
    coordinates["x_3d"] = x_coords
    coordinates["y_3d"] = y_coords
    coordinates["z"] = z_coords
    raster_array = grass_i.read_raster3d_map(raster_3d_name)

    data_array = xr.DataArray(
        raster_array,
        coords=coordinates,
        dims=dims,
        name=grass_i.get_name_from_id(raster_3d_name),
    )
    # Add CF attributes
    r3_infos = grass_i.get_raster3d_info(raster_3d_name)
    da_with_attrs = set_cf_coordinates(
        data_array, gi=grass_i, is_3d=True, z_unit=r3_infos["vertical_units"]
    )
    da_with_attrs.attrs["long_name"] = r3_infos.get("title", "")
    da_with_attrs.attrs["source"] = ",".join([r3_infos["source1"], r3_infos["source2"]])
    da_with_attrs.attrs["units"] = r3_infos.get("units", "")
    da_with_attrs.attrs["comment"] = r3_infos.get("comments", "")
    # CF attributes "institution" and "references"
    # Do not correspond to a direct GRASS value.
    return da_with_attrs


def open_grass_strds(strds_name: str, grass_i: GrassInterface) -> xr.DataArray:
    """must be called from within a grass session
    TODO: add unit, description etc. as attributes
    TODO: lazy loading
    """
    x_coords, y_coords, _ = get_coordinates(grass_i, raster_3d=False).values()
    strds_infos = grass_i.get_stds_infos(strds_name, stds_type="strds")
    if strds_infos.temporal_type == "absolute":
        start_time_dim = "start_time"
        end_time_dim = "end_time"
        time_unit = None
    else:
        time_unit = strds_infos.time_unit
        start_time_dim = f"start_time_{time_unit}"
        end_time_dim = f"end_time_{time_unit}"
    dims = [start_time_dim, "y", "x"]
    coordinates = dict.fromkeys(dims)
    coordinates["x"] = x_coords
    coordinates["y"] = y_coords
    map_list = grass_i.list_maps_in_strds(strds_name)
    array_list = []
    for map_data in map_list:
        coordinates[start_time_dim] = [map_data.start_time]
        coordinates[end_time_dim] = (start_time_dim, [map_data.end_time])
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
    da_concat = xr.concat(array_list, dim=start_time_dim)
    # Add CF attributes
    r_infos = grass_i.get_raster_info(map_list[0].id)
    da_with_attrs = set_cf_coordinates(
        da_concat, gi=grass_i, is_3d=False, time_dim=start_time_dim, time_unit=time_unit
    )
    da_with_attrs.attrs["long_name"] = strds_infos.title
    da_with_attrs.attrs["source"] = ",".join([r_infos["source1"], r_infos["source2"]])
    da_with_attrs.attrs["units"] = r_infos.get("units", "")
    da_with_attrs.attrs["comment"] = r_infos.get("comments", "")
    # CF attributes "institution" and "references"
    # Do not correspond to a direct GRASS value.
    return da_with_attrs


def open_grass_str3ds(str3ds_name: str, grass_i: GrassInterface) -> xr.DataArray:
    """Open a series of 3D raster maps.
    TODO: Figure out what to do when the z value of the maps is time."""
    x_coords, y_coords, z_coords = get_coordinates(grass_i, raster_3d=True).values()
    strds_infos = grass_i.get_stds_infos(str3ds_name, stds_type="str3ds")
    if strds_infos.temporal_type == "absolute":
        start_time_dim = "start_time"
        end_time_dim = "end_time"
        time_unit = None
    else:
        time_unit = strds_infos.time_unit
        start_time_dim = f"start_time_{time_unit}"
        end_time_dim = f"end_time_{time_unit}"
    dims = [start_time_dim, "z", "y_3d", "x_3d"]
    coordinates = dict.fromkeys(dims)
    coordinates["x_3d"] = x_coords
    coordinates["y_3d"] = y_coords
    coordinates["z"] = z_coords
    map_list = grass_i.list_maps_in_str3ds(str3ds_name)
    array_list = []
    for map_data in map_list:
        coordinates[start_time_dim] = [map_data.start_time]
        coordinates[end_time_dim] = (start_time_dim, [map_data.end_time])
        ndarray = grass_i.read_raster3d_map(map_data.id)
        # add time dimension at the beginning
        ndarray = np.expand_dims(ndarray, axis=0)
        data_array = xr.DataArray(
            ndarray,
            coords=coordinates,
            dims=dims,
            name=grass_i.get_name_from_id(str3ds_name),
        )
        array_list.append(data_array)

    da_concat = xr.concat(array_list, dim=start_time_dim)
    # Add CF attributes
    r3_infos = grass_i.get_raster3d_info(map_list[0].id)
    da_with_attrs = set_cf_coordinates(
        da_concat,
        gi=grass_i,
        is_3d=True,
        z_unit=r3_infos["vertical_units"],
        time_dim=start_time_dim,
        time_unit=time_unit,
    )
    da_with_attrs.attrs["long_name"] = strds_infos.title
    da_with_attrs.attrs["source"] = ",".join([r3_infos["source1"], r3_infos["source2"]])
    da_with_attrs.attrs["units"] = r3_infos.get("units", "")
    da_with_attrs.attrs["comment"] = r3_infos.get("comments", "")
    # CF attributes "institution" and "references"
    # Do not correspond to a direct GRASS value.
    return da_with_attrs


def set_cf_coordinates(
    da: xr.DataArray,
    gi: GrassInterface,
    is_3d: bool,
    z_unit: str = "",
    time_dim: str = "",
    time_unit: str = "",
):
    """Set coordinate attributes according to CF conventions"""
    spatial_unit = gi.get_spatial_units()
    if is_3d:
        da["z"].attrs["positive"] = "up"
        da["z"].attrs["axis"] = "Z"
        da["z"].attrs["units"] = z_unit
        y_coord = "y_3d"
        x_coord = "x_3d"
    else:
        y_coord = "y"
        x_coord = "x"
    if time_dim:
        da[time_dim].attrs["axis"] = "T"
        da[time_dim].attrs["standard_name"] = "time"
    if time_unit:
        da[time_dim].attrs["units"] = time_unit
    da[x_coord].attrs["axis"] = "X"
    da[y_coord].attrs["axis"] = "Y"
    if gi.is_latlon():
        da[x_coord].attrs["long_name"] = "longitude"
        da[x_coord].attrs["units"] = "degrees_east"
        da[x_coord].attrs["standard_name"] = "longitude"
        da[y_coord].attrs["long_name"] = "latitude"
        da[y_coord].attrs["units"] = "degrees_north"
        da[y_coord].attrs["standard_name"] = "latitude"
    else:
        da[x_coord].attrs["long_name"] = "x-coordinate in Cartesian system"
        da[y_coord].attrs["long_name"] = "y-coordinate in Cartesian system"
        if gi.is_xy():
            da[x_coord].attrs["standard_name"] = "x_coordinate"
            da[y_coord].attrs["standard_name"] = "y_coordinate"
        else:
            da[x_coord].attrs["standard_name"] = "projection_x_coordinate"
            da[y_coord].attrs["standard_name"] = "projection_y_coordinate"
            da[x_coord].attrs["units"] = spatial_unit
            da[y_coord].attrs["units"] = spatial_unit
    return da


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
