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

from __future__ import annotations
import os
from typing import TYPE_CHECKING, Mapping, Optional

from pyproj import CRS
import xarray as xr
import numpy as np
import pandas as pd

from xarray_grass.coord_utils import get_region_from_xarray

if TYPE_CHECKING:
    from xarray_grass.grass_interface import GrassInterface


def to_grass(
    dataset: xr.Dataset | xr.DataArray,
    dims: Optional[Mapping[str, Mapping[str, str]]] = None,
    overwrite: bool = False,
) -> None:
    """Convert an xarray.Dataset or xarray.DataArray to GRASS GIS maps.

    Perform the conversion of the xarray object's data variables into GRASS
    raster, raster 3D, STRDS, or STR3DS object.


    Parameters
    ----------
    dataset : xr.Dataset | xr.DataArray
        The xarray object to convert. If a Dataset, each data variable
        will be converted.
    dims : Mapping[str, Mapping[str, str]], optional
        A mapping from standard dimension names
        ('start_time', 'end_time', 'x', 'y', 'x_3d', 'y_3d', 'z',)
        to the actual dimension names in the dataset. For example,
        if your 3D variable named "pressure" has an east-west coordinate named 'lon',
        you would pass `dims={'pressure': {'x_3d': 'lon'}}`.
        Defaults to None, which implies standard dimension names are used.
    create : bool, optional
        If True (default), the mapset will be created if it does not exist.
        The parent directory of the mapset path must be a valid GRASS project
        (location).

    Returns
    -------
    None
    """
    if "GISRC" not in os.environ:
        raise RuntimeError(
            "GRASS session not detected. "
            "Please setup a GRASS session before trying to access GRASS data."
        )

    from xarray_grass.grass_interface import GrassInterface

    if isinstance(dataset, xr.Dataset):
        input_var_names = [var_name for var_name, _ in dataset.data_vars.items()]
    elif isinstance(dataset, xr.DataArray):
        input_var_names = [dataset.name]
    else:
        raise TypeError(
            f"'dataset must be either an Xarray DataArray or Dataset, not {type(dataset)}"
        )

    # set dimensions Mapping
    dim_formatter = DimensionsFormatter(input_var_names, dims)
    dim_dataset = dim_formatter.get_formatted_dims()

    # Write to grass
    gi = GrassInterface(overwrite)
    xarray_to_grass = XarrayToGrass(dataset, gi, dim_dataset)
    xarray_to_grass.to_grass()


class DimensionsFormatter:
    """Populate the dimension mapping based on default values and user-provided ones"""

    # Default dimension names
    default_dims = {
        "start_time": "start_time",
        "end_time": "end_time",
        "x": "x",
        "y": "y",
        "x_3d": "x_3d",
        "y_3d": "y_3d",
        "z": "z",
    }

    def __init__(self, input_var_names, input_dims):
        self.input_var_names = input_var_names
        self.input_dims = input_dims
        self._dataset_dims = {}

        # Instantiate the dimensions with default values
        for var_name in input_var_names:
            self._dataset_dims[var_name] = (
                self.default_dims.copy()
            )  # copy avoids shared state

        if self.input_dims is not None:
            self.check_input_dims()

    def check_input_dims(self):
        """Check conformity of provided dims Mapping"""
        if not isinstance(self.input_dims, Mapping):
            raise TypeError(
                "The 'dims' parameter must be of type Mapping[str, Mapping[str, str]]."
            )
        for var_name, var_dims in self.input_dims.items():
            if not isinstance(var_dims, Mapping):
                raise TypeError(
                    "The 'dims' parameter must be of type Mapping[str, Mapping[str, str]]."
                )
            if var_name not in self.input_var_names:
                raise ValueError(
                    f"Variable {var_name} not found in the input dataset. "
                    f"Variables found: {self.input_var_names}"
                )

    def fill_dims(self):
        """Replace the default values with those given by the user."""
        for var_name, dims in self.input_dims.items():
            for dim_key, dim_value in dims.items():
                self._dataset_dims[var_name][dim_key] = dim_value

    def get_formatted_dims(self):
        if self.input_dims is not None:
            self.fill_dims()
        return self._dataset_dims


class XarrayToGrass:
    def __init__(
        self,
        dataset: xr.Dataset | xr.DataArray,
        grass_interface: GrassInterface,
        dims: Mapping[str, str] = None,
    ):
        self.dataset = dataset
        self.grass_interface = grass_interface
        self.dataset_dims = dims

    def to_grass(self) -> None:
        """Convert an xarray Dataset or DataArray to GRASS maps.
        This function validates the CRS and pass the individual DataArrays to the
        `datarray_to_grass` function"""
        grass_crs = CRS(self.grass_interface.get_crs_wkt_str())
        dataset_crs = CRS(self.dataset.attrs["crs_wkt"])
        # TODO: reproj if not same crs
        # TODO: handle no CRS for xy locations
        if grass_crs != dataset_crs:
            raise ValueError(
                f"CRS mismatch: GRASS project CRS is {grass_crs}, "
                f"but dataset CRS is {dataset_crs}."
            )
        try:
            for var_name, data in self.dataset.data_vars.items():
                self._datarray_to_grass(data, self.dataset_dims[var_name])
        except AttributeError:  # DataArray
            self._datarray_to_grass(self.dataset, self.dataset_dims[self.dataset.name])

    def _datarray_to_grass(
        self,
        data: xr.DataArray,
        dims: Mapping[str, str],
    ) -> None:
        """Convert an xarray DataArray to GRASS maps."""
        if len(data.dims) > 4 or len(data.dims) < 2:
            raise ValueError(
                f"Only DataArray with 2 to 4 dimensions are supported. "
                f"Found {len(data.dims)} dimension(s)."
            )

        # Check for 2D spatial dimensions
        is_spatial_2d = dims["x"] in data.dims and dims["y"] in data.dims

        # Check for 3D spatial dimensions
        is_spatial_3d = (
            dims["x_3d"] in data.dims
            and dims["y_3d"] in data.dims
            and dims["z"] in data.dims
        )

        # Check for time dimension
        has_time = dims["start_time"] in data.dims

        # Note: 'end_time' is also a potential temporal dimension but GRASS STRDS/STR3DS
        # are typically defined by a start time.
        # For simplicity 'start_time' is the primary indicator here.

        # Determine dataset type based on number of dimensions and identified dimension types
        is_raster = len(data.dims) == 2 and is_spatial_2d
        is_raster_3d = len(data.dims) == 3 and is_spatial_3d
        is_strds = len(data.dims) == 3 and has_time and is_spatial_2d
        is_str3ds = len(data.dims) == 4 and has_time and is_spatial_3d

        # Set temp region
        current_region = self.grass_interface.get_region()
        temp_region = get_region_from_xarray(data, dims)
        self.grass_interface.set_region(temp_region)
        try:
            if is_raster:
                data = self.transpose(data, dims, arr_type="raster")
                self.grass_interface.write_raster_map(data, data.name)
            elif is_strds:
                self._write_stds(data, dims)
            elif is_raster_3d:
                data = self.transpose(data, dims, arr_type="raster3d")
                self.grass_interface.write_raster3d_map(data, data.name)
            elif is_str3ds:
                self._write_stds(data, dims)
            else:
                raise ValueError(
                    f"DataArray '{data.name}' does not match any supported GRASS dataset type. "
                    f"Expected 2D, 3D, STRDS, or STR3DS."
                )
        finally:
            # Restore the original region
            self.grass_interface.set_region(current_region)

    def transpose(
        self, da: xr.DataArray, dims, arr_type: str = "raster"
    ) -> xr.DataArray:
        """Force dimension order to conform with grass expectation."""
        if "raster" == arr_type:
            return da.transpose(dims["y"], dims["x"])
        elif "raster3d" == arr_type:
            return da.transpose(dims["z"], dims["y_3d"], dims["x_3d"])
        else:
            raise ValueError(
                f"Unknown array type: {arr_type}. Must be 'raster' or 'raster3d'."
            )

    def _write_stds(self, data: xr.DataArray, dims: Mapping):
        # 1. Determine the temporal coordinate and type
        time_coord = data[dims["start_time"]]
        time_dtype = time_coord.dtype
        time_unit = ""  # Initialize for absolute time case
        if isinstance(time_dtype, np.dtypes.DateTime64DType):
            temporal_type = "absolute"
        elif np.issubdtype(time_dtype, np.integer):
            temporal_type = "relative"
            time_unit = time_coord.attrs.get("units", None)
            if not time_unit:
                raise ValueError(
                    f"Relative time coordinate '{dims['start_time']}' in DataArray '{data.name}' "
                    "requires a 'units' attribute. "
                    "Accepted values: 'days', 'hours', 'minutes', 'seconds'."
                )
            # Validate that the unit is supported by both pandas and GRASS
            supported_units = ["days", "hours", "minutes", "seconds"]
            if time_unit not in supported_units:
                raise ValueError(
                    f"Unsupported time unit '{time_unit}' for relative time in DataArray '{data.name}'. "
                    f"Supported units are: {', '.join(supported_units)}. "
                )
        else:
            raise ValueError(f"Temporal type not supported: {time_dtype}")
        # 2. Determine the semantic type
        # TODO: find actual type
        semantic_type = "mean"
        # 2.5 determine if 2D or 3D
        is_3d = False
        stds_type = "strds"
        arr_type = "raster"
        if len(data.isel({dims["start_time"]: 0}).dims) == 3:
            is_3d = True
            stds_type = "str3ds"
            arr_type = "raster3d"

        # Check if exists
        if "strds" == stds_type:
            if (
                not self.grass_interface.overwrite
                and self.grass_interface.name_is_strds(data.name)
            ):
                raise RuntimeError(
                    f"STRDS {data.name} already exists and will not be overwritten."
                )
        elif "str3ds" == stds_type:
            if (
                not self.grass_interface.overwrite
                and self.grass_interface.name_is_str3ds(data.name)
            ):
                raise RuntimeError(
                    f"STR3DS {data.name} already exists and will not be overwritten."
                )
        else:
            raise ValueError(f"Unknown STDS type '{stds_type}'.")

        # 3. Loop through the time dim:
        map_list = []
        for index, time in enumerate(time_coord):
            darray = data.sel({dims["start_time"]: time})
            darray = self.transpose(darray, dims, arr_type=arr_type)
            nd_array = darray.values
            # 3.1 Write each map individually
            raster_name = f"{data.name}_{temporal_type}_{index}"
            if not is_3d:
                self.grass_interface.write_raster_map(
                    arr=nd_array, rast_name=raster_name
                )
            else:
                self.grass_interface.write_raster3d_map(
                    arr=nd_array, rast_name=raster_name
                )
            # 3.2 populate an iterable[tuple[str, datetime | timedelta]]
            time_value = time.values.item()
            if temporal_type == "absolute":
                absolute_time = pd.Timestamp(time_value)
                map_list.append((raster_name, absolute_time.to_pydatetime()))
            else:
                relative_time = pd.Timedelta(time_value, unit=time_unit)
                map_list.append((raster_name, relative_time.to_pytimedelta()))
        # 4. Create STDS and register the maps in it
        self.grass_interface.register_maps_in_stds(
            stds_title="",
            stds_name=data.name,
            stds_desc="",
            map_list=map_list,
            semantic=semantic_type,
            t_type=temporal_type,
            stds_type=stds_type,
            time_unit=time_unit,
        )
