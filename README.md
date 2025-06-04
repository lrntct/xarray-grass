# xarray-grass

![PyPI - Version](https://img.shields.io/pypi/v/xarray-grass?label=pypi%20package)
![PyPI - Downloads](https://img.shields.io/pypi/dm/xarray-grass)
[![tests](https://github.com/lrntct/xarray-grass/actions/workflows/tests.yml/badge.svg)](https://github.com/lrntct/xarray-grass/actions/workflows/tests.yml)

A [GRASS](https://grass.osgeo.org/) backend for [Xarray](https://xarray.dev/).
Explore all your GRASS rasters with Xarray.

## Installation and usage

Install the package using `uv` or `pip`:

`uv add xarray-grass`


```python
>>> import xarray as xr
>>> test_ds = xr.open_dataset("/home/lc/grassdata/nc_spm_08_grass7/PERMANENT/", raster=["boundary_county_500m", "elevation"])
>>> test_ds
<xarray.Dataset> Size: 244kB
Dimensions:               (y: 150, x: 135)
Coordinates:
  * y                     (y) float32 600B 2.2e+05 2.2e+05 ... 2.207e+05
  * x                     (x) float32 540B 6.383e+05 6.383e+05 ... 6.39e+05
Data variables:
    boundary_county_500m  (y, x) float64 162kB ...
    elevation             (y, x) float32 81kB ...
Attributes:
    crs_proj:  +proj=lcc +lat_0=33.75 +lon_0=-79 +lat_1=36.1666666666667 +lat...
    crs_wkt:   PROJCRS["NAD83(HARN) / North Carolina",BASEGEOGCRS["NAD83(HARN...
```

You can choose which map you want to load with the `raster`, `raster_3d`, `strds` and `str3ds` parameters to `open_dataset`.
Those accept either a single string or an iterable.

If run from outside a GRASS session, the tool will automatically create a session in the requested project and mapset.
If run from within GRASS, only maps from accessible mapsets could be loaded.
In GRASS, you can list the accessible mapsets with `g.mapsets`.


## Roadmap

### Version 1.0 goal

- [x] Load a single raster map
- [x] Load a single Space-time Raster Dataset (strds)
- [x] Load a single raster_3d map
- [x] Load a single str3ds
- [x] Load a combination of all the above
- [ ] Load a full mapset
- [ ] Support for the `drop_variables` parameter
- [ ] Lazy loading of all raster types
- [ ] Write from xarray to GRASS

### Stretch goals

- [ ] Load a full GRASS project (ex location)
