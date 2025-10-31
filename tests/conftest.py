import hashlib
from pathlib import Path
import zipfile
from tempfile import TemporaryDirectory
from datetime import datetime, timedelta

import requests
import pytest
import xarray as xr
import numpy as np
import pandas as pd
import grass_session
import grass.script as gs
import grass.temporal as tgis


from xarray_grass import GrassConfig
from xarray_grass import GrassInterface

NC_BASIC_URL = (
    "https://grass.osgeo.org/sampledata/north_carolina/nc_basic_spm_grass7.zip"
)
MODIS_URL = "https://grass.osgeo.org/sampledata/north_carolina/nc_spm_mapset_modis2015_2016_lst_grass8.zip"
NC_BASIC_SHA256 = "f24e564de45c1b19cafca3e15d3f7fbdd844b6a11d733b36d3e36207e4c0b676"
MODIS_SHA256 = "a9d8fd8b30c511feeeb8ce7ee5d8da70eaa6e7b233161a9d1b1a7ce694bcccba"


def sha256(file_path: Path) -> str:
    """Calculate the SHA256 hash of a file."""
    hash_sha256 = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def download_file(url: str, local_dir: Path, sha256_str: str) -> None:
    """Download a file from a URL to a local path."""
    file_name = Path(url).name
    local_file_path = local_dir / file_name
    # Check if the file exists and has the right hash
    try:
        assert sha256(local_file_path) == sha256_str
    except (AssertionError, FileNotFoundError):
        # Download the file
        file_response = requests.get(url, stream=True, timeout=5)
        if file_response.status_code == 200:
            with local_file_path.open("wb") as data_file:
                for chunk in file_response.iter_content(chunk_size=8192):
                    data_file.write(chunk)
        else:
            print(f"Failed to download file: Status code {file_response.status_code}")
    return local_file_path


@pytest.fixture(scope="session")
def test_data_path() -> Path:
    """Path to the permanent test data directory.
    This directory is created in the current test directory."""
    test_dir = Path(__file__).parent / Path("test_data")
    test_dir.mkdir(exist_ok=True)
    return test_dir


@pytest.fixture(scope="session")
def temp_gisdb(test_data_path: Path) -> GrassConfig:
    """create a temporary GISDB with downloaded data."""
    tmp_dir = TemporaryDirectory(prefix="xarray_grass_test_")
    grassdata = Path(tmp_dir.name)
    # NC sample data
    nc_sample_data = download_file(NC_BASIC_URL, test_data_path, NC_BASIC_SHA256)
    with zipfile.ZipFile(nc_sample_data, "r") as zip_ref:
        zip_ref.extractall(grassdata)
    test_project = grassdata / nc_sample_data.stem
    # NC modis data
    nc_modis_data = download_file(MODIS_URL, test_data_path, MODIS_SHA256)
    with zipfile.ZipFile(nc_modis_data, "r") as zip_ref:
        zip_ref.extractall(test_project)
    yield GrassConfig(
        gisdb=str(grassdata),
        project=str(test_project.stem),
        mapset="PERMANENT",
        grassbin=None,
    )
    tmp_dir.cleanup()


@pytest.fixture(scope="session")
def grass_session_fixture(temp_gisdb: GrassConfig):
    """Initialize a GRASS session for tests."""
    with grass_session.Session(
        gisdb=temp_gisdb.gisdb,
        location=temp_gisdb.project,
        mapset=temp_gisdb.mapset,
    ) as session:
        # add the mapset to the session
        gs.run_command("g.mapsets", mapset="modis_lst")

        # Set smaller resolution for faster tests, and info for 3D rasters
        gs.run_command("g.region", flags="o", b=0, t=1500, res3=500, res=500)

        # Add relative and absolute str3ds
        gen_str3ds(temporal_type="relative")
        gen_str3ds(temporal_type="absolute")

        # Add custom str3ds with specific absolute times
        gen_str3ds_custom_absolute(
            start_times=[
                "2014-10-01 00:10:10",
                "2015-10-01 01:11:59",
                "2017-06-01 18:13:56",
            ],
            end_times=[
                "2015-09-01 05:11:23",
                "2017-06-01 18:13:56",
                "2018-02-01 13:10:45.67",
            ],
            str3ds_name="test_str3ds_custom_absolute",
        )

        # Add strds with relative times in seconds
        gen_strds_relative(
            start_times=[0, 1000, 1500],
            end_times=[1000, 1500, 3600],
            time_unit="seconds",
            strds_name="test_strds_relative_seconds",
            seed_add=200,
        )

        # Add strds with relative times in days
        gen_strds_relative(
            start_times=[30, 76, 78],
            end_times=[76, 78, 90],
            time_unit="days",
            strds_name="test_strds_relative_days",
            seed_add=300,
        )

        yield session
        session.close()


@pytest.fixture(scope="function")
def grass_test_region(grass_session_fixture):
    """A test region fixture to make sure the tests are run in a predictable region"""
    # Save current region
    gs.run_command("g.region", save="test_backup_region")
    # Set test region
    gs.run_command("g.region", flags="o", b=0, t=1500, res3=500, res=500)
    yield
    # Restore original region
    gs.run_command("g.region", region="test_backup_region")
    # Clean up the saved region
    gs.run_command("g.remove", type="region", name="test_backup_region", flags="f")


@pytest.fixture(scope="class")
def grass_i():
    return GrassInterface()


def gen_str3ds(
    temporal_type: str = "relative", str3ds_length: int = 3
) -> tgis.Raster3DDataset:
    """Generate an synthetic str3ds."""
    grass_i = GrassInterface()
    if temporal_type == "relative":
        time_unit = "months"
        str3ds_times = [i + 1 for i in range(str3ds_length)]
    elif temporal_type == "absolute":
        time_unit = ""
        base = datetime(year=2000, month=1, day=1)
        str3ds_times = [base - timedelta(days=x * 30) for x in range(str3ds_length)]
    else:
        raise ValueError(f"Unknown temporal type: {temporal_type}")
    # Create the str3ds
    stds_id = grass_i.get_id_from_name(f"test_str3ds_{temporal_type}")
    stds = tgis.open_new_stds(
        name=stds_id,
        type="str3ds",
        temporaltype=temporal_type,
        title="",
        descr="",
        semantic="mean",
    )
    # create MapDataset objects list
    map_dts_lst = []
    for i, map_time in enumerate(str3ds_times):
        # Generate random map. Given seed for reproducibility
        map_name = f"test3d_{map_time}"
        if temporal_type == "absolute":
            formatted_date = map_time.strftime("%Y%m%d")
            map_name = f"test3d_{formatted_date}"
        gs.raster3d.mapcalc3d(exp=f"{map_name}=rand(10,100)", seed=i)
        # create MapDataset
        map_id = grass_i.get_id_from_name(map_name)
        map_dts = tgis.Raster3DDataset(map_id)
        # load spatial data from map
        map_dts.load()
        # set time
        if temporal_type == "relative":
            map_dts.set_relative_time(
                start_time=map_time, end_time=None, unit=time_unit
            )
        elif temporal_type == "absolute":
            map_dts.set_absolute_time(start_time=map_time)
        else:
            assert False, "unknown temporal type!"
        # populate the list
        map_dts_lst.append(map_dts)
    # Finally register the maps
    tgis.register.register_map_object_list(
        type="raster3d",
        map_list=map_dts_lst,
        output_stds=stds,
        delete_empty=False,
        unit=time_unit,
    )
    return stds


def gen_str3ds_custom_absolute(
    start_times: list, end_times: list, str3ds_name: str = "test_str3ds_custom_absolute"
) -> tgis.Raster3DDataset:
    """Generate a synthetic str3ds with custom absolute start and end times."""
    grass_i = GrassInterface()

    # Parse datetime strings to datetime objects
    start_dts = [datetime.fromisoformat(t.replace(" ", "T")) for t in start_times]
    end_dts = [datetime.fromisoformat(t.replace(" ", "T")) for t in end_times]

    # Create the str3ds
    stds_id = grass_i.get_id_from_name(str3ds_name)
    stds = tgis.open_new_stds(
        name=stds_id,
        type="str3ds",
        temporaltype="absolute",
        title="",
        descr="",
        semantic="mean",
    )

    # Create MapDataset objects list
    map_dts_lst = []
    for i, (start_time, end_time) in enumerate(zip(start_dts, end_dts)):
        # Generate random map. Given seed for reproducibility
        formatted_date = start_time.strftime("%Y%m%d_%H%M%S")
        map_name = f"test3d_abs_{formatted_date}"
        gs.raster3d.mapcalc3d(exp=f"{map_name}=rand(10,100)", seed=i + 100)

        # Create MapDataset
        map_id = grass_i.get_id_from_name(map_name)
        map_dts = tgis.Raster3DDataset(map_id)
        # Load spatial data from map
        map_dts.load()
        # Set time
        map_dts.set_absolute_time(start_time=start_time, end_time=end_time)
        # Populate the list
        map_dts_lst.append(map_dts)

    # Finally register the maps
    tgis.register.register_map_object_list(
        type="raster3d",
        map_list=map_dts_lst,
        output_stds=stds,
        delete_empty=False,
        unit="",
    )
    return stds


def gen_strds_relative(
    start_times: list,
    end_times: list,
    time_unit: str,
    strds_name: str,
    seed_add: int = 200,
) -> tgis.RasterDataset:
    """Generate a synthetic strds with relative times in seconds."""
    grass_i = GrassInterface()

    # Create the strds
    stds_id = grass_i.get_id_from_name(strds_name)
    stds = tgis.open_new_stds(
        name=stds_id,
        type="strds",
        temporaltype="relative",
        title="",
        descr="",
        semantic="mean",
    )

    # Create MapDataset objects list
    map_dts_lst = []
    for i, (start_time, end_time) in enumerate(zip(start_times, end_times)):
        # Generate random map
        map_name = f"test_raster_sec_{start_time}"
        gs.mapcalc(exp=f"{map_name}=rand(0,100)", seed=i + seed_add)

        # Create MapDataset
        map_id = grass_i.get_id_from_name(map_name)
        map_dts = tgis.RasterDataset(map_id)
        # Load spatial data from map
        map_dts.load()
        # Set time
        map_dts.set_relative_time(
            start_time=start_time, end_time=end_time, unit=time_unit
        )
        # Populate the list
        map_dts_lst.append(map_dts)

    # Finally register the maps
    tgis.register.register_map_object_list(
        type="raster",
        map_list=map_dts_lst,
        output_stds=stds,
        delete_empty=False,
        unit=time_unit,
    )
    return stds


def create_sample_dataarray(
    dims_spec: dict,
    shape: tuple,
    crs_wkt: str,
    name: str = "test_data",
    time_dim_type: str = "absolute",  # "absolute", "relative", or "none"
    fill_value_generator=None,
) -> xr.DataArray:
    """
    Creates a sample xr.DataArray for testing.

    dims_spec: Dict mapping dimension names to their coordinate values.
               The keys should be the standard names ('time', 'z', 'y', 'x') and
               will be translated to 'y_3d', 'x_3d' based on context (2D vs 3D).
               The order of keys in dims_spec must match the desired final
               dimension order of the DataArray and the order in `shape`.
               Example: {'time': pd.date_range(...), 'y': np.arange(...), ...}
    shape: Tuple defining the shape of the data array, matching the order of dims_spec.
    crs_wkt: WKT string for the crs_wkt attribute.
    name: Name of the xr.DataArray.
    time_dim_type: If 'time' in dims_spec, specifies if time is 'absolute'
                   (datetime objects) or 'relative' (numeric).
    fill_value_generator: Function to generate data, e.g., lambda s: np.random.rand(*s).
                          If None, uses np.random.rand(*shape).
    """
    if fill_value_generator is None:
        data = np.random.rand(*shape)
    else:
        data = fill_value_generator(shape)

    coords = {}
    actual_dims_ordered = []  # This will store the final dimension names in the correct order

    # Define standard internal keys expected in dims_spec
    # These will be mapped to actual_dim_names based on context

    # Determine context for spatial dimension naming (2D or 3D)
    is_3d_spatial_context = "z" in dims_spec

    for dim_key in dims_spec.keys():  # Iterate in the order provided by dims_spec
        coord_values = dims_spec[dim_key]
        actual_dim_name = dim_key  # Default to key

        if dim_key == "time":
            if time_dim_type == "absolute":
                coords[actual_dim_name] = pd.to_datetime(coord_values)
            else:  # relative or none
                coords[actual_dim_name] = coord_values
        elif dim_key == "z":
            actual_dim_name = "z"  # Standard name
            coords[actual_dim_name] = coord_values
        elif dim_key == "y":
            if is_3d_spatial_context:
                actual_dim_name = "y_3d"
            else:  # 2D
                actual_dim_name = "y"
            coords[actual_dim_name] = coord_values
        elif dim_key == "x":
            if is_3d_spatial_context:
                actual_dim_name = "x_3d"
            else:  # 2D
                actual_dim_name = "x"
            coords[actual_dim_name] = coord_values
        else:  # Other dimensions (e.g., custom, non-spatial, non-temporal)
            coords[actual_dim_name] = coord_values

        actual_dims_ordered.append(actual_dim_name)

    if len(actual_dims_ordered) != len(shape):
        raise ValueError(
            f"Number of dimensions derived from dims_spec ({len(actual_dims_ordered)}) "
            f"does not match length of shape ({len(shape)}). "
            f"Ensure dims_spec keys are ordered correctly: {list(dims_spec.keys())} vs {actual_dims_ordered}"
        )

    da = xr.DataArray(
        data,
        coords=coords,
        dims=actual_dims_ordered,
        name=name,
    )
    da.attrs["crs_wkt"] = crs_wkt
    return da


def create_sample_dataset(
    data_vars_specs: dict,
    crs_wkt: str,
    global_time_dim_type: str = "absolute",
) -> xr.Dataset:
    """
    Creates a sample xr.Dataset for testing.

    data_vars_specs: Dict where keys are variable names and values are dicts
                     of parameters for create_sample_dataarray (dims_spec, shape, name,
                     optionally time_dim_type, fill_value_generator).
                     The 'dims_spec' within each variable's spec should follow the
                     ordering and naming conventions for create_sample_dataarray.
    crs_wkt: WKT string for the crs_wkt attribute of the dataset and its DataArrays.
    global_time_dim_type: Default for time_dim_type if not in var_spec.
    """
    data_vars = {}
    for var_name, spec in data_vars_specs.items():
        # Ensure required keys are present in spec
        if not all(k in spec for k in ["dims_spec", "shape"]):
            raise ValueError(
                f"Variable spec for '{var_name}' is missing 'dims_spec' or 'shape'."
            )

        dims_spec = spec["dims_spec"]
        shape = spec["shape"]
        da_name = spec.get("name", var_name)
        time_type = spec.get("time_dim_type", global_time_dim_type)
        fill_gen = spec.get("fill_value_generator", None)

        data_vars[var_name] = create_sample_dataarray(
            dims_spec=dims_spec,
            shape=shape,
            crs_wkt=crs_wkt,
            name=da_name,
            time_dim_type=time_type,
            fill_value_generator=fill_gen,
        )
    ds = xr.Dataset(data_vars)
    ds.attrs["crs_wkt"] = crs_wkt
    return ds
