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


class GrassBackendEntrypoint(BackendEntrypoint):
    """
    Backend entry point for GRASS GIS."""

    open_dataset_parameters = ["filename_or_obj", "drop_variables"]
    description = "Open GRASS mapset in Xarray"
    url = "https://link_to/your_backend/documentation"  # TODO

    def open_dataset(
        self,
        filename_or_obj,
        *,
        drop_variables=None,
        # other backend specific keyword arguments
        # `chunks` and `cache` DO NOT go here, they are handled by xarray
    ) -> xr.Dataset:
        return open_grass_dataset(filename_or_obj, drop_variables=drop_variables)

    def guess_can_open(self, filename_or_obj) -> bool:
        """infer if the path is a GRASS mapset by searching for WIND and VAR files
        in the directory.
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
        else:
            return False


def open_grass_dataset(filename_or_obj, drop_variables=None) -> xr.Dataset:
    """
    Open a GRASS mapset and return an xarray dataset.
    """
    pass


class GrassBackendArray(BackendArray):
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
