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
from typing import TYPE_CHECKING
import threading

import numpy as np
import xarray as xr

from xarray.backends import BackendArray

if TYPE_CHECKING:
    from xarray_grass.grass_interface import GrassInterface


class GrassBackendArray(BackendArray):
    """Lazy loading of grass arrays"""

    def __init__(
        self,
        shape,
        dtype,
        # lock,
        map_id: str,
        map_type: str,
        grass_interface: GrassInterface,
    ):
        self.shape = shape
        self.dtype = dtype
        self._lock = threading.Lock()
        self.map_id = map_id
        self.map_type = map_type  # "raster" or "raster3d"
        self.grass_interface = grass_interface
        self._array: np.ndarray = None

    def __getitem__(self, key: xr.core.indexing.ExplicitIndexer) -> np.typing.ArrayLike:
        """takes in input an index and returns a NumPy array"""
        return xr.core.indexing.explicit_indexing_adapter(
            key,
            self.shape,
            xr.core.indexing.IndexingSupport.BASIC,
            self._raw_indexing_method,
        )

    def _raw_indexing_method(self, key: tuple):
        with self._lock:
            if self._array is None:
                self._array = self._load_map()
            return self._array[key]

    def _load_map(self):
        if self.map_type == "raster":
            return self.grass_interface.read_raster_map(self.map_id)
        else:  # 'raster3d'
            return self.grass_interface.read_raster3d_map(self.map_id)
