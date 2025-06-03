# xarray-grass

![PyPI - Version](https://img.shields.io/pypi/v/xarray-grass?label=pypi%20package)
![PyPI - Downloads](https://img.shields.io/pypi/dm/xarray-grass)
[![tests](https://github.com/lrntct/xarray-grass/actions/workflows/tests.yml/badge.svg)](https://github.com/lrntct/xarray-grass/actions/workflows/tests.yml)

An Xarray backend for GRASS raster data.

## Roadmap

### Formats support

- [x] Load a single raster map
- [x] Load a single Space-time Raster Dataset (strds)
- [x] Load a single raster_3d map
- [x] Load a single str3ds
- [x] Load list of all the above
- [ ] Load a full mapset

### Other functionalities

- [ ] Lazy loading of all raster types
- [ ] Write from xarray to GRASS
- [ ] Load dataset following the CF conventions as much as possible.

### Stretch goals

- [ ] Load a full project (ex location)
