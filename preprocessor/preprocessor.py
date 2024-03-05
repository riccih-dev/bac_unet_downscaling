import xarray as xr
import pandas as pd
import dask.array.core as da
import numpy as np
from scipy.stats import boxcox, yeojohnson
from scipy.stats.mstats import winsorize

'''
Utility functions for Preprocessing-Step
'''

def crop_spatial_dimension(data, crop_region=None, divisible_factor=2):
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
        hr_data = data.sel(
            longitude=slice(crop_region[0], crop_region[2]),
            latitude=slice(crop_region[1], crop_region[3])
        )
    else:
        hr_data = data.copy()  # If no crop region is specified, use the entire dataset

    # Check if spatial dimensions are already divisible by the factor
    if hr_data.longitude.size % divisible_factor == 0 and hr_data.latitude.size % divisible_factor == 0:
        return hr_data  # No further cropping needed, return the cropped dataset

    # Calculate the new size that is divisible by the factor for both longitude and latitude
    new_longitude_size = (hr_data.longitude.size // divisible_factor) * divisible_factor
    new_latitude_size = (hr_data.latitude.size // divisible_factor) * divisible_factor

    # Calculate the starting & ending index to achieve symmetric cropping
    lon_start = (hr_data.longitude.size - new_longitude_size) // 2
    lat_start = (hr_data.latitude.size - new_latitude_size) // 2

    lon_end = lon_start + new_longitude_size
    lat_end = lat_start + new_latitude_size

    # Crop the hr_data dataset for spatial dimensions divisibility
    cropped_data = hr_data.sel(
        longitude=slice(hr_data.longitude.values[lon_start], hr_data.longitude.values[lon_end - 1]),
        latitude=slice(hr_data.latitude.values[lat_start], hr_data.latitude.values[lat_end - 1])
    )

    return cropped_data

def pad_lr_to_match_hr(hr_data, lr_data, var_names=['t2m'], method="linear"):
    """
    Pad the low-resolution data to match the dimensions of high-resolution data using interpolation.

    Returns:
    --------
    xr.Dataset
        Padded low-resolution dataset.
    """
    # Reindex the low-resolution data to match the dimensions of high-resolution data
    #lr_data_reindexed = lr_data.reindex(latitude=hr_data['latitude'], longitude=hr_data['longitude'])

    #values_interp_long = lr_data_reindexed.interpolate_na(dim='longitude', method = 'linear', fill_value="extrapolate")
    #values_interp_lat = values_interp_long.interpolate_na(dim='latitude', method = 'linear', fill_value="extrapolate") # epxloration needed as some latitude dim constist only of nan
    
    #return values_interp_lat

    # Reindex the low-resolution data to match the dimensions of high-resolution data
    lr_data_reindexed = lr_data.reindex(latitude=hr_data['latitude'], longitude=hr_data['longitude'])

    if method == "linear":
        values_interp_long = lr_data_reindexed.interpolate_na(dim='longitude', method='linear', fill_value="extrapolate")
        values_interp_lat = values_interp_long.interpolate_na(dim='latitude', method='linear', fill_value="extrapolate")
    elif method == "nearest":
        values_interp_long = lr_data_reindexed.interpolate_na(dim='longitude', method='nearest')
        values_interp_lat = values_interp_long.interpolate_na(dim='latitude', method='nearest')
    elif method == "knn":
        # not implemented 
        raise ValueError("Unsupported padding method: {}".format(method))
    elif method == "mirroring":
        # Use mirroring padding
        values_interp_long = lr_data_reindexed.pad(
            longitude=(
                ((lr_data_reindexed.dims['longitude'] - lr_data.dims['longitude']) // 2) % lr_data_reindexed.dims['longitude'],
                ((lr_data_reindexed.dims['longitude'] - lr_data.dims['longitude']) // 2) % lr_data_reindexed.dims['longitude']
            ),
            mode="wrap"
        )
        values_interp_lat = values_interp_long.pad(
            latitude=(
                ((lr_data_reindexed.dims['latitude'] - lr_data.dims['latitude']) // 2) % lr_data_reindexed.dims['latitude'],
                ((lr_data_reindexed.dims['latitude'] - lr_data.dims['latitude']) // 2) % lr_data_reindexed.dims['latitude']
            ),
            mode="wrap"
        )
    elif method == "reflection":
        # Use reflection padding
        values_interp_long = lr_data_reindexed.pad(longitude=((lr_data_reindexed.dims['longitude'] - lr_data.dims['longitude']) // 2 + 1, (lr_data_reindexed.dims['longitude'] - lr_data.dims['longitude']) // 2), mode="reflect")
        values_interp_lat = values_interp_long.pad(latitude=((lr_data_reindexed.dims['latitude'] - lr_data.dims['latitude']) // 2 + 1, (lr_data_reindexed.dims['latitude'] - lr_data.dims['latitude']) // 2), mode="reflect")
    else:
        raise ValueError("Unsupported padding method: {}".format(method))

    return values_interp_lat




def transform(data, var_names=['t2m'], transform_type='sqrt'):
    transformed_data = data.copy()

    for var_name in var_names:
        var_values = transformed_data[var_name]

        if transform_type == "log":
            data_adjusted = var_values + 1e-8
            transformed_values = np.log(data_adjusted)
        elif transform_type == "sqrt":
            transformed_values = np.sqrt(var_values)
        elif transform_type == "boxcox":
            transformed_values = boxcox(var_values.values.flatten())
        elif transform_type == "yeojohnson":
            transformed_values, _ = yeojohnson(var_values)
        else:
            transformed_values = var_values

        transformed_data[var_name] = transformed_values

    return transformed_data


def reverse_transform(transformed_data, var_names=['t2m'], transform_type='log'):
    original_data = transformed_data.copy()

    for var_name in var_names:
        var_values = original_data[var_name]

        if transform_type == "log":
            original_values = np.exp(var_values)
        elif transform_type == "sqrt":
            original_values = np.square(var_values)
        elif transform_type == "boxcox":
            original_values = boxcox(var_values, inverse=True)
        elif transform_type == "yeojohnson":
            original_values = yeojohnson(var_values, inverse=True)
        else:
            original_values = var_values

        original_data[var_name] = original_values

    return original_data


def remove_outliers_iqr(data, var_names=['t2m']):
    """
    Remove outliers from data using Interquartile Range (IQR).

    Args:
        data (np.ndarray): Data array.

    Returns:
        np.ndarray: Data array with outliers removed.
    """
    for var_name in var_names:
        var_values = data[var_name]
        q1 = np.percentile(var_values, 25)
        q3 = np.percentile(var_values, 75)
        iqr_val = q3 - q1
        lower_bound = q1 - 1.5 * iqr_val
        upper_bound = q3 + 1.5 * iqr_val
        data[var_name] = np.clip(var_values, lower_bound, upper_bound)

    return data

def winsorize_outliers(data, var_names=['t2m'], limits=[0.05, 0.05]):
    """
    Winsorize outliers from data.

    Args:
        data (np.ndarray): Data array.
        var_names (list): List of variable names to process.
        limits (list): Winsorizing limits for each variable.

    Returns:
        np.ndarray: Data array with outliers winsorized.
    """
    winsorized_data = data.copy()

    for var_name in var_names:
        var_values = winsorized_data[var_name]
        winsorized_data[var_name] = winsorize(var_values, limits=limits)

    return winsorized_data

def crop_era5_to_cerra(lr_data, lr_lsm_z, hr_data):
    """
    Crop the low resolution datasets to match the geographical area covered by the high resolution datasets.

    Parameters:
    - lr_data (xr.Dataset): Low resolution temperature data (t2m) with dimensions (time, latitude, longitude).
    - lr_lsm_z (xr.Dataset): Low resolution land surface mask (lsm) and orography (orog) with dimensions (time, latitude, longitude).
    - hr_data (xr.Dataset): High resolution temperature data (t2m) with dimensions (time, latitude, longitude).

    Returns:
    - lr_t2m_cropped (xr.Dataset): Cropped low resolution temperature data (t2m) matching CERRA dimensions.
    - lr_lsm_z_cropped (xr.Dataset): Cropped low resolution land surface mask (lsm) and orography (orog) matching CERRA dimensions.
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
    Extracts the 't2m' data from the ERA5 dataset at specific times, per default (00:00, 06:00, 12:00, 18:00).

    Parameters:
    - data (xarray.Dataset): dataset containing 't2m' variable.

    Returns:
    - t2m_at_specific_times (xarray.DataArray): Extracted 't2m' data at specific times.
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

    #print("time extracting completed")

    return extracted_data


