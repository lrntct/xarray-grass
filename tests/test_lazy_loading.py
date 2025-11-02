from pathlib import Path

import pytest
import xarray as xr


ACTUAL_STRDS = "LST_Day_monthly@modis_lst"


@pytest.mark.usefixtures("grass_session_fixture", "grass_test_region")
class TestXarrayGrass:
    def test_strds_lazy_loading_detailed(self, grass_i, temp_gisdb) -> None:
        """Detailed diagnostic to identify exactly where lazy loading breaks.

        This test instruments the code to find the exact point where
        LazilyIndexedArray gets converted to numpy array.
        """
        from unittest.mock import patch, wraps
        from xarray_grass.grass_interface import GrassInterface

        mapset_path = (
            Path(temp_gisdb.gisdb) / Path(temp_gisdb.project) / Path(temp_gisdb.mapset)
        )

        # Track read calls
        read_calls = []
        original_read = GrassInterface.read_raster_map

        def tracked_read(map_id):
            import traceback

            read_calls.append(
                {
                    "map_id": map_id,
                    "stack": "".join(
                        traceback.format_stack()[-8:-1]
                    ),  # More stack frames
                }
            )
            return original_read(map_id)

        # Track concat calls more carefully
        concat_details = []
        original_concat = xr.concat

        @wraps(original_concat)
        def tracked_concat(objs, dim, **kwargs):
            if objs:
                first_var = objs[0]._variable if hasattr(objs[0], "_variable") else None
                concat_details.append(
                    {
                        "num_objs": len(objs),
                        "dim": dim,
                        "first_obj_data_type": type(objs[0].data).__name__,
                        "first_obj_is_lazy": isinstance(
                            objs[0].data, xr.core.indexing.LazilyIndexedArray
                        ),
                        "first_var_data_type": type(first_var._data).__name__
                        if first_var is not None
                        else None,
                        "first_var_is_lazy": isinstance(
                            first_var._data, xr.core.indexing.LazilyIndexedArray
                        )
                        if first_var is not None
                        else None,
                    }
                )
            result = original_concat(objs, dim, **kwargs)
            if concat_details:
                concat_details[-1]["result_data_type"] = type(result.data).__name__
                concat_details[-1]["result_is_lazy"] = isinstance(
                    result.data, xr.core.indexing.LazilyIndexedArray
                )
            return result

        # Patch at module level to capture all GrassInterface instances
        with (
            patch("xarray_grass.xarray_grass.GrassInterface") as MockGI,
            patch("xarray.concat", side_effect=tracked_concat),
        ):
            # Make mock return real instances with tracked read method
            def create_tracked_instance(*args, **kwargs):
                real_instance = GrassInterface(*args, **kwargs)
                # Replace the static method with our tracking wrapper
                real_instance.read_raster_map = staticmethod(tracked_read)
                return real_instance

            MockGI.side_effect = create_tracked_instance

            test_dataset = xr.open_dataset(mapset_path, strds=ACTUAL_STRDS)

            print(f"\n=== Total read_raster_map calls: {len(read_calls)}")
            print(f"=== Total concat calls: {len(concat_details)}")

            if concat_details:
                print("\n=== Concat details:")
                for i, call in enumerate(concat_details):
                    print(
                        f"  Call {i + 1}: Concatenating {call['num_objs']} objects on dim '{call['dim']}'"
                    )
                    print(
                        f"    First obj data: {call['first_obj_data_type']} (lazy={call['first_obj_is_lazy']})"
                    )
                    if call["first_var_data_type"]:
                        print(
                            f"    First obj Variable._data: {call['first_var_data_type']} (lazy={call['first_var_is_lazy']})"
                        )
                    print(
                        f"    Result data: {call['result_data_type']} (lazy={call['result_is_lazy']})"
                    )

            strds_name = grass_i.get_name_from_id(ACTUAL_STRDS)
            da = test_dataset[strds_name]

            print("\n=== Final DataArray:")
            print(f"    data type: {type(da.data).__name__}")
            print(f"    Variable._data type: {type(da._variable._data).__name__}")
            print(
                f"    Is LazilyIndexedArray: {isinstance(da.data, xr.core.indexing.LazilyIndexedArray)}"
            )

    def test_strds_lazy_loading(self, grass_i, temp_gisdb) -> None:
        """Test that STRDS data is actually loaded lazily and not eagerly.

        This test verifies that:
        1. Opening a STRDS doesn't immediately load all data into memory
        2. Data is only loaded when actually accessed via .values
        3. Individual time slices are loaded on demand, not all at once
        """
        from unittest.mock import patch
        from xarray_grass.grass_interface import GrassInterface

        mapset_path = (
            Path(temp_gisdb.gisdb) / Path(temp_gisdb.project) / Path(temp_gisdb.mapset)
        )

        # Track read_raster_map calls across ALL GrassInterface instances
        read_calls = []
        original_read = GrassInterface.read_raster_map

        def tracked_read_raster_map(map_id):
            """Wrapper that tracks calls while preserving functionality"""
            read_calls.append(map_id)
            return original_read(map_id)

        # Patch at module level where GrassInterface is used to create instances
        # This ensures ALL instances created during xr.open_dataset() are tracked
        with patch("xarray_grass.xarray_grass.GrassInterface") as MockGI:
            # Make the mock return real GrassInterface instances with tracked read method
            def create_tracked_instance(*args, **kwargs):
                real_instance = GrassInterface(*args, **kwargs)
                # Replace the static method with our tracking wrapper
                real_instance.read_raster_map = staticmethod(tracked_read_raster_map)
                return real_instance

            MockGI.side_effect = create_tracked_instance

            # Open the dataset - this should NOT load any raster data
            test_dataset = xr.open_dataset(mapset_path, strds=ACTUAL_STRDS)
            strds_name = grass_i.get_name_from_id(ACTUAL_STRDS)

            print(f"\n=== After opening: {len(read_calls)} maps read")

            # The key assertion: opening should NOT trigger data loading
            assert len(read_calls) == 0, (
                f"Expected 0 raster reads during dataset opening, "
                f"but {len(read_calls)} maps were read. "
                f"Lazy loading is NOT working!"
            )

            # Get the DataArray - still shouldn't load data
            da = test_dataset[strds_name]
            assert len(read_calls) == 0, (
                f"Getting DataArray triggered {len(read_calls)} reads"
            )

            # Check it's a lazy array (MemoryCachedArray wrapping LazilyIndexedArray)
            print(
                f"=== DataArray._variable._data type: {type(da._variable._data).__name__}"
            )

            # Now access a single time slice via .values - this SHOULD trigger loading
            read_calls.clear()
            _ = da.isel({f"start_time_{strds_name}": 0}).values

            print(f"=== After accessing first slice: {len(read_calls)} maps read")

            # Should have loaded exactly 1 map (the first time slice)
            assert len(read_calls) == 1, (
                f"Expected 1 raster read when accessing first time slice, "
                f"got {len(read_calls)}"
            )

            # Access another slice - should load one more
            read_calls.clear()
            _ = da.isel({f"start_time_{strds_name}": 1}).values

            print(f"=== After accessing second slice: {len(read_calls)} maps read")

            assert len(read_calls) == 1, (
                f"Expected 1 raster read when accessing second time slice, "
                f"got {len(read_calls)}"
            )

            print("\n✅ Lazy loading is working correctly!")
            print("   - Opening STRDS: 0 maps loaded")
            print("   - Accessing slice 0: 1 map loaded")
            print("   - Accessing slice 1: 1 map loaded")
