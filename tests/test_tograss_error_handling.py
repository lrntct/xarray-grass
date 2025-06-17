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
import tempfile

import xarray as xr
import numpy as np
import pytest
from pyproj import CRS
from xarray_grass import GrassInterface, to_grass
from .conftest import create_sample_dataarray


@pytest.mark.usefixtures("grass_session_fixture")
class TestToGrassErrorHandling:
    def test_missing_crs_wkt_attribute(self, temp_gisdb, grass_i: GrassInterface):
        """Test error handling when input xarray object is missing 'crs_wkt' attribute."""
        mapset_name = temp_gisdb.mapset
        mapset_path = Path(temp_gisdb.gisdb) / temp_gisdb.project / mapset_name
        # Create a DataArray without crs_wkt
        # The helper function always adds it, so we create one manually here.
        sample_da_no_crs = xr.DataArray(
            np.random.rand(2, 2),
            coords={"y": [0, 1], "x": [0, 1]},
            dims=("y", "x"),
            name="data_no_crs",
        )
        # Intentionally do not set sample_da_no_crs.attrs["crs_wkt"]

        with pytest.raises(
            (KeyError, AttributeError, ValueError),
            match=r"(crs_wkt|CRS mismatch|has no attribute 'attrs')",
        ):
            to_grass(dataset=sample_da_no_crs, mapset=str(mapset_path), create=False)

    def test_incompatible_crs_wkt(self, temp_gisdb, grass_i: GrassInterface):
        """Test error handling with an incompatible 'crs_wkt' attribute."""
        mapset_name = temp_gisdb.mapset
        mapset_path = Path(temp_gisdb.gisdb) / temp_gisdb.project / mapset_name
        session_crs_wkt = grass_i.get_crs_wkt_str()
        # Create an incompatible CRS WKT string
        incompatible_crs = CRS.from_epsg(4326)  # WGS 84
        if CRS.from_wkt(session_crs_wkt).equals(incompatible_crs):
            # If by chance the session CRS is compatible, pick another one
            incompatible_crs = CRS.from_epsg(23032)  # UTM zone 32N, Denmark
        incompatible_crs_wkt = incompatible_crs.to_wkt()

        sample_da = create_sample_dataarray(
            dims_spec={"y": np.arange(2.0), "x": np.arange(2.0)},
            shape=(2, 2),
            crs_wkt=incompatible_crs_wkt,  # Set the incompatible CRS
            name="data_incompatible_crs",
        )

        with pytest.raises(
            ValueError,
            match=r"CRS mismatch",
        ):
            to_grass(dataset=sample_da, mapset=str(mapset_path), create=False)

    def test_invalid_mapset_path_non_existent_parent(
        self, temp_gisdb, grass_i: GrassInterface
    ):
        """Test error with mapset path having a non-existent parent directory."""
        pytest.skip("Skipping mapset creation test due to GRASS <8.5 tgis.init() bug.")
        session_crs_wkt = grass_i.get_crs_wkt_str()
        # Path to a non-existent directory, then the mapset
        mapset_path = (
            Path(temp_gisdb.gisdb)
            / temp_gisdb.project
            / "non_existent_parent_dir"
            / "my_mapset"
        )

        sample_da = create_sample_dataarray(
            dims_spec={"y": np.arange(2.0), "x": np.arange(2.0)},
            shape=(2, 2),
            crs_wkt=session_crs_wkt,
            name="data_invalid_parent",
        )

        with pytest.raises(
            ValueError,
            match=r"Mapset.*not found and its parent directory.*is not a valid GRASS project",
        ):
            to_grass(dataset=sample_da, mapset=str(mapset_path), create=True)

    def test_invalid_mapset_path_is_file(self, temp_gisdb, grass_i: GrassInterface):
        """Test error with mapset path being an existing file."""
        pytest.skip("Skipping mapset creation test due to GRASS <8.5 tgis.init() bug.")
        session_crs_wkt = grass_i.get_crs_wkt_str()

        # Create an empty file where the mapset directory would be
        file_as_mapset_path = (
            Path(temp_gisdb.gisdb) / temp_gisdb.project / "file_instead_of_mapset"
        )
        with open(file_as_mapset_path, "w") as f:
            f.write("This is a file.")

        assert file_as_mapset_path.is_file()

        sample_da = create_sample_dataarray(
            dims_spec={"y": np.arange(2.0), "x": np.arange(2.0)},
            shape=(2, 2),
            crs_wkt=session_crs_wkt,
            name="data_mapset_is_file",
        )

        with pytest.raises(ValueError, match=r"not a directory"):
            to_grass(dataset=sample_da, mapset=str(file_as_mapset_path), create=True)

        file_as_mapset_path.unlink()  # Clean up the created file

    def test_parent_dir_not_grass_location(self, grass_i: GrassInterface):
        """Test error when parent of mapset is not a GRASS Location (create=True)."""
        pytest.skip("Skipping mapset creation test due to GRASS <8.5 tgis.init() bug.")
        session_crs_wkt = grass_i.get_crs_wkt_str()

        with tempfile.TemporaryDirectory(
            prefix="not_a_grass_loc_"
        ) as tmp_non_grass_dir:
            mapset_path_in_non_grass_loc = Path(tmp_non_grass_dir) / "my_mapset_here"

            sample_da = create_sample_dataarray(
                dims_spec={"y": np.arange(2.0), "x": np.arange(2.0)},
                shape=(2, 2),
                crs_wkt=session_crs_wkt,
                name="data_non_grass_parent",
            )

            # This relies on to_grass checking if the parent is a GRASS location.
            # The exact error message might vary based on implementation.
            with pytest.raises(
                ValueError,
                match=r"(not a valid GRASS project|Parent directory.*not a GRASS location|Invalid GIS database)",
            ):
                to_grass(
                    dataset=sample_da,
                    mapset=str(mapset_path_in_non_grass_loc),
                    create=True,
                )

    def test_create_false_mapset_not_exists(self, temp_gisdb, grass_i: GrassInterface):
        """Test error when create=False and mapset does not exist."""
        session_crs_wkt = grass_i.get_crs_wkt_str()
        non_existent_mapset_path = (
            Path(temp_gisdb.gisdb) / temp_gisdb.project / "mapset_does_not_exist_at_all"
        )

        assert not non_existent_mapset_path.exists()  # Ensure it really doesn't exist

        sample_da = create_sample_dataarray(
            dims_spec={"y": np.arange(2.0), "x": np.arange(2.0)},
            shape=(2, 2),
            crs_wkt=session_crs_wkt,
            name="data_create_false_no_mapset",
        )

        with pytest.raises(
            ValueError,
            match=r"is not a valid directory",
        ):
            to_grass(
                dataset=sample_da, mapset=str(non_existent_mapset_path), create=False
            )

    def test_mapset_not_accessible_simplified(self, grass_i: GrassInterface):
        """Test simplified 'mapset not accessible' by providing a syntactically valid but unrelated path."""
        pytest.skip("Skipping mapset creation test due to GRASS <8.5 tgis.init() bug.")
        session_crs_wkt = grass_i.get_crs_wkt_str()

        # A path that is unlikely to be a GRASS mapset accessible to the current session
        # This doesn't create a separate GRASS session, just uses a bogus path.
        # The function should ideally detect this isn't a valid mapset within the current GISDB.
        unrelated_path = "/tmp/some_completely_random_unrelated_path_for_mapset_test"
        # Ensure it doesn't exist, or the error might be different (e.g. "is a file")
        if Path(unrelated_path).exists():
            try:
                if Path(unrelated_path).is_dir():
                    import shutil

                    shutil.rmtree(unrelated_path)
                else:
                    Path(unrelated_path).unlink()
            except OSError:
                pytest.skip(f"Could not clean up unrelated_path: {unrelated_path}")

        sample_da = create_sample_dataarray(
            dims_spec={"y": np.arange(2.0), "x": np.arange(2.0)},
            shape=(2, 2),
            crs_wkt=session_crs_wkt,
            name="data_unrelated_mapset_path",
        )

        # The expected error could be about invalid path, not a GRASS mapset, or not in current GISDB.
        with pytest.raises(
            ValueError,
            match=r"not found and .* is not a valid GRASS project",
        ):
            to_grass(dataset=sample_da, mapset=unrelated_path, create=True)
            to_grass(
                dataset=sample_da, mapset=unrelated_path, create=False
            )  # Also test with create=False


@pytest.mark.usefixtures("grass_session_fixture")
class TestToGrassInputValidation:
    def test_invalid_dataset_type(self, temp_gisdb, grass_i: GrassInterface):
        """Test error handling for invalid 'dataset' parameter type.
        That a first try. Let's see how it goes considering that the tested code uses duck typing."""
        mapset_path = Path(temp_gisdb.gisdb) / temp_gisdb.project / temp_gisdb.mapset

        invalid_datasets = [123, "a string", [1, 2, 3], {"data": np.array([1])}, None]
        for invalid_ds in invalid_datasets:
            with pytest.raises(
                AttributeError,
                match=r"object has no attribute 'attrs'",
            ):
                to_grass(dataset=invalid_ds, mapset=str(mapset_path), create=False)

    def test_invalid_dims_parameter_type(self, temp_gisdb, grass_i: GrassInterface):
        """Test error handling for invalid 'dims' parameter type or content."""
        session_crs_wkt = grass_i.get_crs_wkt_str()
        # Use PERMANENT mapset to avoid issues with tgis.init() for newly created mapsets
        mapset_path = Path(temp_gisdb.gisdb) / temp_gisdb.project / temp_gisdb.mapset
        # No need to create PERMANENT mapset as it's guaranteed by temp_gisdb fixture

        sample_da = create_sample_dataarray(
            dims_spec={"y": np.arange(2.0), "x": np.arange(2.0)},
            shape=(2, 2),
            crs_wkt=session_crs_wkt,
            name="data_for_dims_validation",
        )

        invalid_dims_params = [
            "not_a_dict",
            123,
            ["y", "x"],
        ]
        for invalid_dims in invalid_dims_params:
            with pytest.raises(
                TypeError,
                match=r"dims parameter must be a mapping",  # More specific regex
            ):
                to_grass(
                    dataset=sample_da,
                    mapset=str(mapset_path),
                    dims=invalid_dims,
                    create=False,
                )

    def test_invalid_mapset_parameter_type(self, temp_gisdb, grass_i: GrassInterface):
        """Test error handling for invalid 'mapset' parameter type."""
        session_crs_wkt = grass_i.get_crs_wkt_str()
        sample_da = create_sample_dataarray(
            dims_spec={"y": np.arange(2.0), "x": np.arange(2.0)},
            shape=(2, 2),
            crs_wkt=session_crs_wkt,
            name="data_for_mapset_type_validation",
        )

        invalid_mapset_params = [
            123,
            None,
            ["path", "to", "mapset"],
            {"path": "mapset_dir"},
        ]
        for invalid_mapset in invalid_mapset_params:
            with pytest.raises(
                TypeError,
                match=r"(mapset parameter must be a string or a Path|Invalid mapset type|argument should be a str or an os.PathLike object)",
            ):
                to_grass(dataset=sample_da, mapset=invalid_mapset, create=True)
