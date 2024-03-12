import xarray as xr
import pandas as pd
import dask.array.core as da

'''
Utility functions for Preprocessing-Step
'''

def crop_spatial_dimension(data, crop_region=None, divisible_factor=32):
    """
    Crops the given xarray dataset based on a specified region and ensures spatial dimensions are divisible by a specified factor.

    In many convolutional neural network architectures, especially those involving multiple downsampling and upsampling operations,
    it is common to design the network such that the spatial dimensions are divisible by certain factors (e.g., 32). This ensures that
    the dimensions can be downsampled and upsampled without resulting in fractional spatial dimensions.

    Parameters:
    -----------
    data : xr.Dataset
        Xarray dataset to be cropped and adjusted.
    crop_region : tuple or None, optional
        A tuple (min_lon, min_lat, max_lon, max_lat) specifying the region to crop the dataset to. If set to None, no initial cropping is performed.
        Default is None.
    divisible_factor : int, optional
        The desired factor to ensure divisibility for both longitude and latitude. Default is 32.

    Returns:
    ----------
    xr.Dataset: Cropped and adjusted xarray dataset with spatial dimensions divisible by the specified factor.
    
    """
    # Crop based on specified region if provided
    if crop_region is not None:
        cropped_data = data.sel(
            longitude=slice(crop_region[0], crop_region[2]),
            latitude=slice(crop_region[1], crop_region[3])
        )
    else:
        cropped_data = data.copy()  # If no crop region is specified, use the entire dataset

    # Check if spatial dimensions are already divisible by the factor
    if cropped_data.longitude.size % divisible_factor == 0 and cropped_data.latitude.size % divisible_factor == 0:
        return cropped_data  # No further cropping needed, return the cropped dataset

    # Calculate the new size that is divisible by the factor for both longitude and latitude
    new_longitude_size = (cropped_data.longitude.size // divisible_factor) * divisible_factor
    new_latitude_size = (cropped_data.latitude.size // divisible_factor) * divisible_factor

    # Calculate the starting & ending index to achieve symmetric cropping
    lon_start = (cropped_data.longitude.size - new_longitude_size) // 2
    lat_start = (cropped_data.latitude.size - new_latitude_size) // 2

    lon_end = lon_start + new_longitude_size
    lat_end = lat_start + new_latitude_size

    # Crop the hr_data dataset for spatial dimensions divisibility
    cropped_data = cropped_data.sel(
        longitude=slice(cropped_data.longitude.values[lon_start], cropped_data.longitude.values[lon_end - 1]),
        latitude=slice(cropped_data.latitude.values[lat_start], cropped_data.latitude.values[lat_end - 1])
    )

    return cropped_data


def pad_lr_to_match_hr(hr_data, lr_data, method="linear"):
    """
    Pad the low-resolution data to match the dimensions of high-resolution data using interpolation.

    Parameters:
    -----------
    hr_data : xr.Dataset
        High-resolution dataset.
    lr_data : xr.Dataset
        Low-resolution dataset to be padded.
    method : str, optional
        Interpolation method. Default is "linear".

    Returns:
    --------
    xr.Dataset
        Padded low-resolution dataset with dimensions matching those of the high-resolution dataset.
    """
    # Reindex the low-resolution data to match the dimensions of high-resolution data
    lr_data_reindexed = lr_data.reindex(latitude=hr_data['latitude'], longitude=hr_data['longitude'])

    # epxloration needed as some latitude dim constist only of nan
    values_interp_long = lr_data_reindexed.interpolate_na(dim='longitude', method = method, fill_value="extrapolate")
    values_interp_lat = values_interp_long.interpolate_na(dim='latitude', method = method, fill_value="extrapolate") 
    
    return values_interp_lat


def crop_era5_to_cerra(lr_data, lr_lsm_z, hr_data):
    """
    Crop the low resolution datasets to match the geographical area covered by the high resolution datasets.

    Parameters:
    - lr_data (xarray.Dataset): Low resolution temperature data (t2m) with dimensions (time, latitude, longitude).
    - lr_lsm_z (xarray.Dataset): Low resolution land surface mask (lsm) and orography (orog) with dimensions (time, latitude, longitude).
    - hr_data (xarray.Dataset): High resolution temperature data (t2m) with dimensions (time, latitude, longitude).

    Returns:
    - lr_t2m_cropped (xarray.Dataset): Cropped low resolution temperature data (t2m) matching CERRA dimensions.
    - lr_lsm_z_cropped (xarray.Dataset): Cropped low resolution land surface mask (lsm) and orography (orog) matching CERRA dimensions.
    """
    # Crop ERA5 to match CERRA area without additional 5% coverage
    lr_t2m_cropped = lr_data.sel(
        longitude=slice(hr_data.longitude[0], hr_data.longitude[-1]),
        latitude=slice(hr_data.latitude[0], hr_data.latitude[-1])
    )

    lr_lsm_z_cropped = lr_lsm_z.sel(
        longitude=slice(hr_data.longitude[0], hr_data.longitude[-1]),
        latitude=slice(hr_data.latitude[0], hr_data.latitude[-1])
    )

    return lr_t2m_cropped, lr_lsm_z_cropped


def sort_ds(data):
    """
    Sorts a dataset based on latitude and then on longitude.

    Parameters:
    - data (xarray.Dataset): The input dataset to be sorted. It should contain variables for latitude and longitude.

    Returns:
    xarray.Dataset: The sorted dataset.
    """
    data = data.sortby('latitude')
    data = data.sortby('longitude')

    return data

def combine_data(data, additional_features, var_names):
    """
    Combine additional data variables with the original dataset.

    Parameters:
    - data (xarray.Dataset): Original dataset containing the 'time', 'latitude', and 'longitude' dimensions.
    - additional_features (xarray.Dataset): Additional dataset containing further feature variables specified in var_names
      with the 'time', 'latitude', and 'longitude' dimensions.
    - var_names (list): List of variable names to be combined from additional_data.

    Returns:
    xarray.Dataset: Combined dataset with variables from additional_data added.
    """
    # Drop time dimension (not needed since only one time period)
    additional_features = additional_features.isel(time=0)

    # Extract and expand additional variables from additional_data
    additional_vars = {}
    for var_name in var_names:
        var_data = additional_features[var_name].drop_vars('time')
        var_expanded = var_data.expand_dims(time=data['time'])
        additional_vars[var_name] = (["time", "latitude", "longitude"], var_expanded.data)

    # Combine data variables with the original data
    combined_data = data.assign(**additional_vars)

    return combined_data


def extract_t2m_at_specific_times(data, specific_times = ['12:00'], chunk_size=10):
    """
    Extracts the 't2m' data from the ERA5 dataset at specific times, per default (12:00).

    Parameters:
    - data (xarray.Dataset): Dataset containing 't2m' variable.
    - specific_times (list, optional): List of specific times to extract data. Default is ['12:00'].
    - chunk_size (int, optional): Size of chunks for processing large datasets stored in chunks. Default is 10.

    Returns:
    - t2m_at_specific_times (xarray.Dataset): Extracted 't2m' data at specific times.
    """
    # Extract the time values
    times = data['time'].values

    # Extract indices of the specific times
    time_indices = [i for i, t in enumerate(times) if pd.to_datetime(t).strftime('%H:%M') in specific_times]

    # Extract t2m data at specific times
    t2m = data['t2m']

    if isinstance(t2m.data, da.Array):
        # Iterate over chunks of time indices
        chunks = [time_indices[i:i + chunk_size] for i in range(0, len(time_indices), chunk_size)]
        t2m_at_specific_times_list = []

        for chunk in chunks:
            t2m_at_specific_times_chunk = t2m.isel(time=chunk).compute()
            t2m_at_specific_times_list.append(t2m_at_specific_times_chunk)

        t = 1
        # Concatenate chunks back into a single DataArray
        t2m_at_specific_times = xr.concat(t2m_at_specific_times_list, dim='time')
    else:
        # DataArray case
        t2m_at_specific_times = t2m.isel(time=time_indices)

    # Create a new dataset with the same structure as the original dataset
    extracted_data = xr.Dataset({
        't2m': (['time', 'latitude', 'longitude'], t2m_at_specific_times.values)
    },
        coords={
            'longitude': data['longitude'],
            'latitude': data['latitude'],
            'time': data['time'].isel(time=time_indices)
        },
        attrs={'units': 'K', 'long_name': '2 metre temperature'}
    )

    return extracted_data


