'''
Utility functions for Preprocessing-Step
'''

def crop_spatial_dimension(data, divisible_factor=2):
    """
    Crops the given xarray dataset to ensure spatial dimensions (longitude and latitude) are divisible by a specified factor.

    In many convolutional neural network architectures, especially those involving multiple downsampling and upsampling operations,
    it is common to design the network such that the spatial dimensions are divisible by certain factors (e.g., 32). This ensures that
    the dimensions can be downsampled and upsampled without resulting in fractional spatial dimensions.

    Parameters:
    -----------
    data : xr.Dataset
        Xarray dataset to be cropped.
    divisible_factor : int, optional
        The desired factor to ensure divisibility for both longitude and latitude. Default is 32.

    Returns:
    ----------
    xr.Dataset: Cropped xarray dataset with spatial dimensions divisible by the specified factor.
    """
     # Check if spatial dimensions are already divisible by the factor
    if data.longitude.size % divisible_factor == 0 and data.latitude.size % divisible_factor == 0:
        return data  # No cropping needed, return the original dataset


    # Calculate the new size that is divisible by the factor for both longitude and latitude
    new_longitude_size = (data.longitude.size // divisible_factor) * divisible_factor
    new_latitude_size = (data.latitude.size // divisible_factor) * divisible_factor

    # Calculate the starting & ending index to achieve symmetric cropping
    lon_start = (data.longitude.size - new_longitude_size) // 2
    lat_start = (data.latitude.size - new_latitude_size) // 2

    lon_end = lon_start + new_longitude_size
    lat_end = lat_start + new_latitude_size

    # Crop the ERA5 dataset
    cropped_data = data.sel(
        longitude=slice(data.longitude.values[lon_start], data.longitude.values[lon_end - 1]),
        latitude=slice(data.latitude.values[lat_start], data.latitude.values[lat_end - 1])
    )

    return cropped_data

def pad_lr_to_match_hr(hr_data, lr_data):
    """
    Pad the low-resolution data to match the dimensions of high-resolution data using interpolation.

    Returns:
    --------
    xr.Dataset
        Padded low-resolution dataset.
    """
    # Reindex the low-resolution data to match the dimensions of high-resolution data
    lr_data_reindexed = lr_data.reindex(latitude=hr_data['latitude'], longitude=hr_data['longitude'])

    values_interp_long = lr_data_reindexed.interpolate_na(dim='longitude', method = 'linear', fill_value="extrapolate")
    values_interp_lat = values_interp_long.interpolate_na(dim='latitude', method = 'linear', fill_value="extrapolate") # epxloration needed as some latitude dim constist only of nan
    
    return values_interp_lat

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


