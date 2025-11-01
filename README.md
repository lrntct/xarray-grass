# xarray-grass

[![PyPI - Version](https://img.shields.io/pypi/v/xarray-grass?label=pypi%20package)](https://pypi.org/project/xarray-grass/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/xarray-grass)](https://pypistats.org/packages/xarray-grass)
[![tests](https://github.com/lrntct/xarray-grass/actions/workflows/tests.yml/badge.svg)](https://github.com/lrntct/xarray-grass/actions/workflows/tests.yml)

A [GRASS](https://grass.osgeo.org/) backend for [Xarray](https://xarray.dev/).
Explore all your GRASS rasters with Xarray.

## Installation

Install the package using `uv` or `pip`:

`uv add xarray-grass`

You need to install GRASS independently.

## Loading GRASS data as an Xarray Dataset

```python
>>> import xarray as xr
>>> test_ds = xr.open_dataset("/home/lc/grassdata/nc_spm_08_grass7/PERMANENT/", raster=["boundary_county_500m", "elevation"])
>>> test_ds
<xarray.Dataset> Size: 253kB
Dimensions:               (y: 150, x: 140)
Coordinates:
  * y                     (y) float32 600B 2.2e+05 2.2e+05 ... 2.207e+05
  * x                     (x) float32 560B 6.383e+05 6.383e+05 ... 6.39e+05
Data variables:
    boundary_county_500m  (y, x) float64 168kB ...
    elevation             (y, x) float32 84kB ...
Attributes:
    crs_wkt:      PROJCRS["NAD83(HARN) / North Carolina",BASEGEOGCRS["NAD83(H...
    Conventions:  CF-1.13-draft
    history:      2025-10-31 18:22:16.644873+00:00: Created with xarray-grass...
    source:       GRASS database. project: <nc_spm_08_grass7>, mapset:<PERMAN...

```

You can choose which maps you want to load with the `raster`, `raster_3d`, `strds` and `str3ds` parameters to `open_dataset`.
Those accept either a single string or an iterable.
If none of those are specified, the whole mapset will be loaded, ignoring single maps that are already registered in either a `strds` or `str3ds`;
those maps will be loaded into the Xarray Dataset for being part of the GRASS Space Time Dataset.
Any time-stamp associated to a single map not registered in a stds is ignored.

The extent and resolution of the resulting `Dataset` is defined by the region setting of GRASS, set with the `g.region` GRASS tool.
Note that in GRASS the 3D resolution is independent from the 2D resolution.
Therefore, 2D and 3D maps loaded in Xarray will not share the same dimensions and coordinates.
The coordinates in the Xarray `Dataset` correspond to the center of the GRASS cell.

If run from outside a GRASS session, `xarray-grass` will automatically create a session in the requested project and mapset.
If run from within GRASS, only maps from accessible mapsets could be loaded.
You can list the accessible mapsets with `g.mapsets` from GRASS.

In GRASS, the time dimension of various STDSs is not homogeneous, as it is for the spatial coordinates.
To reflect this, xarray-grass will create one time dimension for each STDS loaded.

From within a grass session, it is possible to access various mapsets.

## CF conventions attributes mapping

### DataArray attributes

|CF name  |Origin in GRASS|
|---------|---------------|
|long_name|The "title" field from "r.info", "r3.info", or "t.info"|
|source   |Concatenation of "source1" and "source2" from "r.info" or "r3.info". In case of STDS, taken from the first map.|
|units    |The "unit" field from "r.info" or "r3.info". In case of STDS, taken from the first map.|
|comment  |The "comments" field from "r.info" or "r3.info". In case of STDS, taken from the first map.|

The attributes of the coordinates are in line with CF Conventions.

### Dataset attributes

The attributes set at the dataset level are:
  - `crs_wkt` from the g.proj command
  - `Conventions`, the CF Convention version
  - `history`, the time of creation and version of xarray-grass
  - `source`, the name of the current grass project and mapset

## Writing an Xarray Dataset or DataArray to GRASS


```python
import xarray as xr
from xarray_grass import to_grass

mapset = "/home/lc/grassdata/nc_spm_08_grass7/PERMANENT/"  # Or pathlib.Path

test_ds = xr.open_dataset(
    mapset,
    raster=["boundary_county_500m", "elevation"],
    strds="LST_Day_monthly@modis_lst",
)

print(test_ds)

# Let's write the modis time series back into the PERMANENT mapset

da_modis = test_ds["LST_Day_monthly"]
# xarray-grass needs the CRS information to write to GRASS
da_modis.attrs["crs_wkt"] = test_ds.attrs["crs_wkt"]

to_grass(
    dataset=da_modis,
    mapset=mapset,
    dims={
        "LST_Day_monthly": {"start_time": "start_time_LST_Day_monthly"},
    },
    overwrite=False
)
```

The above `print` statement should return this:

```
<xarray.Dataset> Size: 3MB
Dimensions:                     (y: 165, x: 179, start_time_LST_Day_monthly: 24)
Coordinates:
  * y                           (y) float32 660B 2.196e+05 ... 2.212e+05
  * x                           (x) float32 716B 6.377e+05 ... 6.395e+05
  * start_time_LST_Day_monthly  (start_time_LST_Day_monthly) datetime64[ns] 192B ...
    end_time_LST_Day_monthly    (start_time_LST_Day_monthly) datetime64[ns] 192B ...
Data variables:
    boundary_county_500m        (y, x) float64 236kB ...
    elevation                   (y, x) float32 118kB ...
    LST_Day_monthly             (start_time_LST_Day_monthly, y, x) int32 3MB ...
Attributes:
    crs_wkt:      PROJCRS["NAD83(HARN) / North Carolina",BASEGEOGCRS["NAD83(H...
    Conventions:  CF-1.13-draft
    history:      2025-11-01 02:10:24.652838+00:00: Created with xarray-grass...
    source:       GRASS database. project: <nc_spm_08_grass7_test>, mapset:<P...
```

## Roadmap

### Goals for version 1.0

- [x] Load a single raster map
- [x] Load a single Space-time Raster Dataset (strds)
- [x] Load a single raster_3d map
- [x] Load a single str3ds
- [x] Load a combination of all the above
- [x] Load a full mapset
- [x] Support for the `drop_variables` parameter
- [ ] Write from xarray to GRASS
  - [x] Write to a 2D raster
  - [x] Write to STRDS
  - [x] Write to 3D raster
  - [x] Write to STR3DS
  - [x] Transpose if dimensions are not in the expected order
  - [ ] Support time units for relative time
  - [ ] Support `end_time`
  - [ ] Accept writing into a specific mapset (GRASS 8.5)
  - [ ] Accept non homogeneous 3D resolution in NS and EW dimensions (GRASS 8.5)
- [ ] Lazy loading of all raster types
- [ ] Properly test with lat-lon location

### Stretch goals

- [ ] Load all mapsets from a GRASS project (ex location)
- [ ] Read CRS definitions from CF compatible fields
