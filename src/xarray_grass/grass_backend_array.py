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

import threading
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from xarray.backends import BackendArray
from xarray.core import indexing

if TYPE_CHECKING:
    from xarray_grass.grass_interface import GrassInterface, MapData


class GrassSTDSBackendArray(BackendArray):
    """Lazy loading of grass Space-Time DataSets (multiple maps in time series)"""

    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: np.dtype[Any],
        map_list: list[MapData],
        map_type: Literal["raster", "raster3d"],
        grass_interface: GrassInterface,
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self._lock = threading.Lock()
        self.map_list = map_list
        self.map_type = map_type
        self.grass_interface = grass_interface
        self._cached_maps: dict[int, np.ndarray] = {}

    def __getitem__(self, key: indexing.ExplicitIndexer) -> np.ndarray:
        """takes in input an index and returns a NumPy array"""
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.BASIC,
            self._raw_indexing_method,
        )

    def _raw_indexing_method(self, key: tuple[Any, ...]) -> np.ndarray:
        """Load only the maps needed for the requested slice"""
        with self._lock:
            # key is a tuple of slices/indices for each dimension
            # First dimension is time
            time_key = key[0] if key else slice(None)
            spatial_key = key[1:] if len(key) > 1 else ()

            # Determine which time indices are needed
            if isinstance(time_key, slice):
                time_indices = range(*time_key.indices(self.shape[0]))
            elif isinstance(time_key, int):
                time_indices = [time_key]
            else:
                time_indices = list(time_key)

            # Load only the needed maps
            result_list: list[np.ndarray] = []
            for t_idx in time_indices:
                if t_idx not in self._cached_maps:
                    map_data = self.map_list[t_idx]
                    if self.map_type == "raster":
                        self._cached_maps[t_idx] = self.grass_interface.read_raster_map(
                            map_data.id
                        )
                    else:  # 'raster3d'
                        self._cached_maps[t_idx] = (
                            self.grass_interface.read_raster3d_map(map_data.id)
                        )

                # Apply spatial indexing
                if spatial_key:
                    result_list.append(self._cached_maps[t_idx][spatial_key])
                else:
                    result_list.append(self._cached_maps[t_idx])

            # Stack results along time dimension
            if len(result_list) == 1 and isinstance(time_key, int):
                # Single time slice requested as integer index
                return result_list[0]
            else:
                return np.stack(result_list, axis=0)
