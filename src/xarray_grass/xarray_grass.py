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
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from xarray.backends import BackendEntrypoint
import xarray as xr

import xarray_grass
from xarray_grass.grass_interface import GrassInterface
from xarray_grass.grass_backend_array import GrassSTDSBackendArray


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
        raster: Optional[str | Iterable[str]] = None,
        raster_3d: Optional[str | Iterable[str]] = None,
        strds: Optional[str | Iterable[str]] = None,
        str3ds: Optional[str | Iterable[str]] = None,
        drop_variables: Iterable[str],
    ) -> xr.Dataset:
        """Open GRASS project or mapset as an xarray.Dataset.
        Requires an active GRASS session.
        TODO: add support for whole project.
        """
        if "GISRC" not in os.environ:
            raise RuntimeError(
                "GRASS session not detected. "
                "Please setup a GRASS session before trying to access GRASS data."
            )

        if filename_or_obj:
            dirpath = Path(filename_or_obj)
            if not dir_is_grass_mapset(dirpath):
                raise ValueError(f"{filename_or_obj} is not a GRASS mapset")
                self.check_accessible_mapset(filename_or_obj)

        self.grass_interface = GrassInterface()

        open_func_params = dict(
            raster_list=raster,
            raster_3d_list=raster_3d,
            strds_list=strds,
            str3ds_list=str3ds,
        )
        if not any([raster, raster_3d, strds, str3ds]):
            self._list_all_mapset(open_func_params)
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

        return self._open_grass_maps(filename_or_obj, **open_func_params)

    def guess_can_open(self, filename_or_obj) -> bool:
        """infer if the path is a GRASS mapset.
        TODO: add support for whole project."""
        return dir_is_grass_mapset(filename_or_obj)

    def _list_all_mapset(self, open_func_params):
        """List map objects in the whole mapset.
        If a map is part of a STDS, do not list it as a single map.
        """
        grass_objects = self.grass_interface.list_grass_objects()
        # strds
        rasters_in_strds = []
        for strds_name in grass_objects["strds"]:
            maps_in_strds = self.grass_interface.list_maps_in_strds(strds_name)
            rasters_in_strds.extend([map_data.id for map_data in maps_in_strds])
            if open_func_params["strds_list"] is None:
                open_func_params["strds_list"] = [strds_name]
            else:
                open_func_params["strds_list"].append(strds_name)
        raster3ds_in_str3ds = []
        # str3ds
        for str3ds_name in grass_objects["str3ds"]:
            maps_in_str3ds = self.grass_interface.list_maps_in_str3ds(str3ds_name)
            raster3ds_in_str3ds.extend([map_data.id for map_data in maps_in_str3ds])
            if open_func_params["str3ds_list"] is None:
                open_func_params["str3ds_list"] = [str3ds_name]
            else:
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

    def check_accessible_mapset(self, filename_or_obj):
        dirpath = Path(filename_or_obj)
        mapset = dirpath.stem
        project_path = dirpath.parent
        gisdb_path = project_path.parent
        project = project_path.stem
        gisdb = gisdb_path
        gisenv = self.grass_interface.get_gisenv()
        current_gisdb = gisenv["GISDBASE"]
        current_location = gisenv["LOCATION_NAME"]
        accessible_mapsets = self.grass_interface.get_accessible_mapsets()

        requested_path = Path(gisdb) / Path(project)
        current_path = Path(current_gisdb) / Path(current_location)

        if requested_path != current_path or str(mapset) not in accessible_mapsets:
            raise ValueError(
                f"Cannot access {gisdb}/{project}/{mapset} "
                f"from current GRASS session in project "
                f"{current_gisdb}/{current_location}. "
                f"Accessible mapsets: {accessible_mapsets}."
            )

    def _open_grass_maps(
        self,
        filename_or_obj: str | Path,
        raster_list: Iterable[str] = None,
        raster_3d_list: Iterable[str] = None,
        strds_list: Iterable[str] = None,
        str3ds_list: Iterable[str] = None,
    ) -> xr.Dataset:
        """
        Open a GRASS mapset and return an xarray dataset.
        """
        # Configuration for processing different GRASS map types
        map_processing_configs = [
            {
                "input_list": raster_list,
                "existence_check_method": self.grass_interface.name_is_raster,
                "open_function": self._open_grass_raster,
                "not_found_key": "raster",
            },
            {
                "input_list": raster_3d_list,
                "existence_check_method": self.grass_interface.name_is_raster_3d,
                "open_function": self._open_grass_raster_3d,
                "not_found_key": "raster_3d",
            },
            {
                "input_list": strds_list,
                "existence_check_method": self.grass_interface.name_is_strds,
                "open_function": self._open_grass_strds,
                "not_found_key": "strds",
            },
            {
                "input_list": str3ds_list,
                "existence_check_method": self.grass_interface.name_is_str3ds,
                "open_function": self._open_grass_str3ds,
                "not_found_key": "str3ds",
            },
        ]
        # Open all given maps and identify non-existent data
        not_found = {config["not_found_key"]: [] for config in map_processing_configs}
        data_array_list = []
        raw_coords_list = []
        for config in map_processing_configs:
            for map_name in config["input_list"]:
                if not config["existence_check_method"](map_name):
                    not_found[config["not_found_key"]].append(map_name)
                    continue
                data_array = config["open_function"](map_name)
                raw_coords_list.append(data_array.coords)
                data_array_list.append(data_array)
        if any(not_found.values()):
            raise ValueError(f"Objects not found: {not_found}")

        crs_wkt = self.grass_interface.get_crs_wkt_str()

        coords_dict = {}
        for coords in raw_coords_list:
            for k, v in coords.items():
                coords_dict[k] = v

        data_array_dict = {da.name: da for da in data_array_list}

        # Set attributes
        dirpath = Path(filename_or_obj)
        mapset = dirpath.stem
        project = dirpath.parent.stem
        attrs = {
            "crs_wkt": crs_wkt,
            "Conventions": "CF-1.13-draft",
            # "title": "",
            "history": f"{datetime.now(timezone.utc)}: Created with xarray-grass version {xarray_grass.__version__}",
            "source": f"GRASS database. project: {Path(project).name}, mapset: {Path(mapset).name}",
            # "references": "",
            # "institution": "",
            # "comment": "",
        }
        dataset = xr.Dataset(data_vars=data_array_dict, coords=coords_dict, attrs=attrs)
        return dataset

    def _set_cf_coordinates_attributes(
        self,
        da: xr.DataArray,
        is_3d: bool,
        z_unit: str = "",
        time_dims: Optional[list[str, str]] = None,
        time_unit: str = "",
    ):
        """Set coordinate attributes according to CF conventions"""
        spatial_unit = self.grass_interface.get_spatial_units()
        if is_3d:
            # da["z"].attrs["positive"] = "up"  # Not defined by grass
            da["z"].attrs["axis"] = "Z"
            da["z"].attrs["units"] = z_unit
            y_coord = "y_3d"
            x_coord = "x_3d"
        else:
            y_coord = "y"
            x_coord = "x"
        if time_dims is not None:
            for time_dim in time_dims:
                da[time_dim].attrs["axis"] = "T"
                da[time_dim].attrs["standard_name"] = "time"
                if time_unit:  # NetCDF does not accept units for absolute time
                    da[time_dim].attrs["units"] = time_unit
        da[x_coord].attrs["axis"] = "X"
        da[y_coord].attrs["axis"] = "Y"
        if self.grass_interface.is_latlon():
            da[x_coord].attrs["long_name"] = "longitude"
            da[x_coord].attrs["units"] = "degrees_east"
            da[x_coord].attrs["standard_name"] = "longitude"
            da[y_coord].attrs["long_name"] = "latitude"
            da[y_coord].attrs["units"] = "degrees_north"
            da[y_coord].attrs["standard_name"] = "latitude"
        else:
            da[x_coord].attrs["long_name"] = "x-coordinate in Cartesian system"
            da[y_coord].attrs["long_name"] = "y-coordinate in Cartesian system"
            if self.grass_interface.is_xy():
                da[x_coord].attrs["standard_name"] = "x_coordinate"
                da[y_coord].attrs["standard_name"] = "y_coordinate"
            else:
                da[x_coord].attrs["standard_name"] = "projection_x_coordinate"
                da[y_coord].attrs["standard_name"] = "projection_y_coordinate"
                da[x_coord].attrs["units"] = str(spatial_unit)
                da[y_coord].attrs["units"] = str(spatial_unit)
        return da

    def _open_grass_raster(self, raster_name: str) -> xr.DataArray:
        """Open a single raster map."""
        x_coords, y_coords, _ = self.grass_interface.get_coordinates(
            raster_3d=False
        ).values()
        dims = ["y", "x"]
        coordinates = dict.fromkeys(dims)
        coordinates["x"] = x_coords
        coordinates["y"] = y_coords
        raster_array = self.grass_interface.read_raster_map(raster_name)
        data_array = xr.DataArray(
            raster_array,
            coords=coordinates,
            dims=dims,
            name=self.grass_interface.get_name_from_id(raster_name),
        )
        # Add CF attributes
        r_infos = self.grass_interface.get_raster_info(raster_name)
        da_with_attrs = self._set_cf_coordinates_attributes(data_array, is_3d=False)
        da_with_attrs.attrs["long_name"] = r_infos.get("title", "")
        da_with_attrs.attrs["source"] = ",".join(
            [r_infos["source1"], r_infos["source2"]]
        )
        da_with_attrs.attrs["units"] = str(r_infos.get("units", ""))
        da_with_attrs.attrs["comment"] = str(r_infos.get("comments", ""))
        # CF attributes "institution" and "references"
        # Do not correspond to a direct GRASS value.
        return da_with_attrs

    def _open_grass_raster_3d(self, raster_3d_name: str) -> xr.DataArray:
        """Open a single 3D raster map."""
        x_coords, y_coords, z_coords = self.grass_interface.get_coordinates(
            raster_3d=True
        ).values()
        dims = ["z", "y_3d", "x_3d"]
        coordinates = dict.fromkeys(dims)
        coordinates["x_3d"] = x_coords
        coordinates["y_3d"] = y_coords
        coordinates["z"] = z_coords
        raster_array = self.grass_interface.read_raster3d_map(raster_3d_name)

        data_array = xr.DataArray(
            raster_array,
            coords=coordinates,
            dims=dims,
            name=self.grass_interface.get_name_from_id(raster_3d_name),
        )
        # Add CF attributes
        r3_infos = self.grass_interface.get_raster3d_info(raster_3d_name)
        da_with_attrs = self._set_cf_coordinates_attributes(
            data_array, is_3d=True, z_unit=r3_infos["vertical_units"]
        )
        da_with_attrs.attrs["long_name"] = r3_infos.get("title", "")
        da_with_attrs.attrs["source"] = ",".join(
            [r3_infos["source1"], r3_infos["source2"]]
        )
        da_with_attrs.attrs["units"] = r3_infos.get("units", "")
        da_with_attrs.attrs["comment"] = r3_infos.get("comments", "")
        # CF attributes "institution" and "references"
        # Do not correspond to a direct GRASS value.
        return da_with_attrs

    def _open_grass_strds(self, strds_name: str) -> xr.DataArray:
        """Open a STRDS with lazy loading - data is only loaded when accessed"""
        strds_id = self.grass_interface.get_id_from_name(strds_name)
        strds_name = self.grass_interface.get_name_from_id(strds_id)
        x_coords, y_coords, _ = self.grass_interface.get_coordinates(
            raster_3d=False
        ).values()
        strds_infos = self.grass_interface.get_stds_infos(strds_id, stds_type="strds")
        if strds_infos.temporal_type == "absolute":
            time_unit = ""
        else:
            time_unit = strds_infos.time_unit
        start_time_dim = f"start_time_{strds_name}"
        end_time_dim = f"end_time_{strds_name}"

        map_list = self.grass_interface.list_maps_in_strds(strds_id)
        region = self.grass_interface.get_region()

        # Create a single backend array for the entire STRDS
        backend_array = GrassSTDSBackendArray(
            shape=(len(map_list), region.rows, region.cols),
            dtype=map_list[0].dtype,
            map_list=map_list,
            map_type="raster",
            grass_interface=self.grass_interface,
        )
        lazy_array = xr.core.indexing.LazilyIndexedArray(backend_array)

        # Create Variable with lazy array
        var = xr.Variable(dims=[start_time_dim, "y", "x"], data=lazy_array)

        # Extract time coordinates
        start_times = [map_data.start_time for map_data in map_list]
        end_times = [map_data.end_time for map_data in map_list]

        # Create coordinates
        coordinates = {
            "x": x_coords,
            "y": y_coords,
            start_time_dim: start_times,
            end_time_dim: (start_time_dim, end_times),
        }

        # Convert to DataArray
        data_array = xr.DataArray(
            var,
            coords=coordinates,
            name=strds_name,
        )

        # Add CF attributes
        r_infos = self.grass_interface.get_raster_info(map_list[0].id)
        da_with_attrs = self._set_cf_coordinates_attributes(
            data_array,
            is_3d=False,
            time_dims=[start_time_dim, end_time_dim],
            time_unit=time_unit,
        )
        da_with_attrs.attrs["long_name"] = strds_infos.title
        da_with_attrs.attrs["source"] = ",".join(
            [r_infos["source1"], r_infos["source2"]]
        )
        da_with_attrs.attrs["units"] = r_infos.get("units", "")
        da_with_attrs.attrs["comment"] = r_infos.get("comments", "")
        # CF attributes "institution" and "references"
        # Do not correspond to a direct GRASS value.
        return da_with_attrs

    def _open_grass_str3ds(self, str3ds_name: str) -> xr.DataArray:
        """Open a STR3DS with lazy loading - data is only loaded when accessed
        TODO: Figure out what to do when the z value of the maps is time."""
        str3ds_id = self.grass_interface.get_id_from_name(str3ds_name)
        str3ds_name = self.grass_interface.get_name_from_id(str3ds_id)
        x_coords, y_coords, z_coords = self.grass_interface.get_coordinates(
            raster_3d=True
        ).values()
        strds_infos = self.grass_interface.get_stds_infos(str3ds_id, stds_type="str3ds")
        if strds_infos.temporal_type == "absolute":
            time_unit = ""
        else:
            time_unit = strds_infos.time_unit
        start_time_dim = f"start_time_{str3ds_name}"
        end_time_dim = f"end_time_{str3ds_name}"

        map_list = self.grass_interface.list_maps_in_str3ds(str3ds_id)
        region = self.grass_interface.get_region()

        # Create a single backend array for the entire STR3DS
        backend_array = GrassSTDSBackendArray(
            shape=(len(map_list), region.depths, region.rows3, region.cols3),
            dtype=map_list[0].dtype,
            map_list=map_list,
            map_type="raster3d",
            grass_interface=self.grass_interface,
        )
        lazy_array = xr.core.indexing.LazilyIndexedArray(backend_array)

        # Create Variable with lazy array
        var = xr.Variable(dims=[start_time_dim, "z", "y_3d", "x_3d"], data=lazy_array)

        # Extract time coordinates
        start_times = [map_data.start_time for map_data in map_list]
        end_times = [map_data.end_time for map_data in map_list]

        # Create coordinates
        coordinates = {
            "x_3d": x_coords,
            "y_3d": y_coords,
            "z": z_coords,
            start_time_dim: start_times,
            end_time_dim: (start_time_dim, end_times),
        }

        # Convert to DataArray
        data_array = xr.DataArray(
            var,
            coords=coordinates,
            name=str3ds_name,
        )

        # Add CF attributes
        r3_infos = self.grass_interface.get_raster3d_info(map_list[0].id)
        da_with_attrs = self._set_cf_coordinates_attributes(
            data_array,
            is_3d=True,
            z_unit=r3_infos["vertical_units"],
            time_dims=[start_time_dim, end_time_dim],
            time_unit=time_unit,
        )
        da_with_attrs.attrs["long_name"] = strds_infos.title
        da_with_attrs.attrs["source"] = ",".join(
            [r3_infos["source1"], r3_infos["source2"]]
        )
        da_with_attrs.attrs["units"] = r3_infos.get("units", "")
        da_with_attrs.attrs["comment"] = r3_infos.get("comments", "")
        # CF attributes "institution" and "references"
        # Do not correspond to a direct GRASS value.
        return da_with_attrs


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
