import hashlib
from pathlib import Path
import zipfile
from tempfile import TemporaryDirectory
from datetime import datetime, timedelta
import os

import requests
import pytest
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
        gs.run_command("g.region", b=0, t=100, tbres=10, rows=150, cols=135)

        # Add relative and absolute str3ds
        gen_str3ds(temporal_type="relative")
        gen_str3ds(temporal_type="absolute")

        yield session
        session.close()


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
    str3ds_length = 10
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
