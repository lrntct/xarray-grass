import hashlib
from pathlib import Path
import zipfile

import requests
import pytest

from xarray_grass import GrassConfig

NC_BASIC_URL = (
    "https://grass.osgeo.org/sampledata/north_carolina/nc_basic_spm_grass7.zip"
)
MODIS_URL = "https://grass.osgeo.org/sampledata/north_carolina/nc_spm_mapset_modis2015_2016_lst.zip"
NC_BASIC_SHA256 = "f24e564de45c1b19cafca3e15d3f7fbdd844b6a11d733b36d3e36207e4c0b676"
MODIS_SHA256 = "06b78ea035464d155b221375a1ad2f81a76f6a35462cb4ebb87c9a34a19a58e0"


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
    """create a temporary GISDB"""
    # create grassdata directory
    grassdata = test_data_path / Path("grassdata")
    grassdata.mkdir(exist_ok=True)
    # NC sample data
    nc_sample_data = download_file(NC_BASIC_URL, test_data_path, NC_BASIC_SHA256)
    with zipfile.ZipFile(nc_sample_data, "r") as zip_ref:
        zip_ref.extractall(grassdata)
    test_project = grassdata / nc_sample_data.stem
    # NC modis data
    nc_modis_data = download_file(MODIS_URL, test_data_path, MODIS_SHA256)
    with zipfile.ZipFile(nc_modis_data, "r") as zip_ref:
        zip_ref.extractall(test_project)
    return GrassConfig(
        gisdb=str(grassdata),
        project=str(test_project.stem),
        mapset="PERMANENT",
        grassbin=None,
    )
