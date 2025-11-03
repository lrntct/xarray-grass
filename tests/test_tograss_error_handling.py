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
            to_grass(dataset=sample_da_no_crs)

    def test_incompatible_crs_wkt(self, temp_gisdb, grass_i: GrassInterface):
        """Test error handling with an incompatible 'crs_wkt' attribute."""
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
            to_grass(dataset=sample_da)

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
            to_grass(dataset=sample_da)
            to_grass(dataset=sample_da)


@pytest.mark.usefixtures("grass_session_fixture")
class TestToGrassInputValidation:
    def test_invalid_dataset_type(self, temp_gisdb, grass_i: GrassInterface):
        """Test error handling for invalid 'dataset' parameter type.
        That a first try. Let's see how it goes considering that the tested code uses duck typing."""
        invalid_datasets = [123, "a string", [1, 2, 3], {"data": np.array([1])}, None]
        for invalid_ds in invalid_datasets:
            with pytest.raises(
                TypeError,
                match=r"'dataset must be either an Xarray DataArray or Dataset",
            ):
                to_grass(dataset=invalid_ds)

    def test_invalid_dims_parameter_type(self, temp_gisdb, grass_i: GrassInterface):
        """Test error handling for invalid 'dims' parameter type or content."""
        session_crs_wkt = grass_i.get_crs_wkt_str()

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
                match=r"'dims' parameter must be of type",
            ):
                to_grass(
                    dataset=sample_da,
                    dims=invalid_dims,
                )

    def test_invalid_dims_invalid_var(self, temp_gisdb, grass_i: GrassInterface):
        session_crs_wkt = grass_i.get_crs_wkt_str()
        sample_da = create_sample_dataarray(
            dims_spec={"y": np.arange(2.0), "x": np.arange(2.0)},
            shape=(2, 2),
            crs_wkt=session_crs_wkt,
            name="data_for_dims_validation",
        )
        with pytest.raises(
            ValueError,
            match=r"not found in the input dataset. Variables found",
        ):
            to_grass(
                dataset=sample_da,
                dims={"invalid_var_name": {"x": "my_x"}},
            )
