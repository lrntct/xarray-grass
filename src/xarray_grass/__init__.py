from xarray_grass.grass_interface import GrassConfig as GrassConfig
from xarray_grass.grass_interface import GrassInterface as GrassInterface
from xarray_grass.xarray_grass import GrassBackendEntrypoint as GrassBackendEntrypoint
from xarray_grass.grass_backend_array import (
    GrassSTDSBackendArray as GrassSTDSBackendArray,
)
from xarray_grass.to_grass import to_grass as to_grass
from xarray_grass.coord_utils import RegionData as RegionData

__version__ = "0.4.0"
