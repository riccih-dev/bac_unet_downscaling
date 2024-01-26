
import xarray as xr
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import numpy as np

# TODO
# - create a better way to crop data (form each border equally)
# - adjust splitting so it workds with correct years + test it 

def split_data(data, test_size=0.11):
    """
    Splits a given xarray dataset into training, validation, and test sets based on time.

    Parameters:
    -----------
    data : xr.Dataset
        Xarray dataset to be split.
    test_size : float, optional
        The proportion of data to include in the test split. Default is 0.2.

    Returns:
    ----------
    xr.Dataset: Xarray dataset for training.
    xr.Dataset: Xarray dataset for validation.
    xr.Dataset: Xarray dataset for testing.
    """

    common_time = data['time'].values

    train_time, test_time = train_test_split(common_time, test_size=test_size, shuffle=False)
    train_time, val_time = train_test_split(train_time, test_size=test_size, shuffle=False)

    # Use joblib to parallelize the following operations
    train_data, val_data, test_data = Parallel(n_jobs=-1)(delayed(__split)((t, data)) for t in [train_time, val_time, test_time]) # type: ignore
    return train_data, val_data, test_data


def __split(args):
    """
    Helper function to extract data for a specific time slice.

    Parameters:
    -----------
    args : tuple
        A tuple containing time slice and the original data.

    Returns:
    ----------
    xr.Dataset: Xarray dataset for the specified time slice.
    """

    t, data = args
    return data.sel(time=t)


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



def store_to_disk(file_name, data, file_path="./data/"):
    """
    Store data to disk in a specified folder.

    Parameters:
    - file_name (str): The name of the file to be saved.
    - data: The data to be stored.
    - folder_path (str, optional): The path to the folder where the file will be saved.
      Default is "data".
    """
    file = f"{file_path}{file_name}.nc"

    # first option
    write_job = data.to_netcdf(file, compute=False) #try zarr

    print(f"Writing to {file}")
    write_job.compute()

    # second option 
    path = "./data/"
    write_job = data.to_zarr(path, mode='w', compute=False)

    print(f"Writing to {file}")
    write_job.compute(progress_bar=True)


def load_from_disk(file_name, file_path="./data/"):
    """
    Load climate data from disk.

    Parameters:
    - file_path (str): Path to the directory containing the file. Default is "data".
    - file_name (str): Name of the file.

    Returns:
    - xr.Dataset: Loaded climate data.
    """
    file_path = file_path + file_name + ".nc"
    return xr.open_dataset(file_path)


def find_input_shape(data):
    """
    Determines the input shape for a given xarray dataset to be used as input for a model. The input shape is determined by the number of time steps, 
    latitude points, longitude points, and data variables in the dataset.

    Parameters:
    -----------
    data : xr.Dataset
        Xarray dataset to determine the input shape.

    Returns:
    ----------
    tuple: Input shape tuple (latitude_points, longitude_points, num_variables).
    """

    # Extracting dimensions from the dataset
    time_steps = len(data['time'])
    latitude_points = len(data['latitude'])
    longitude_points = len(data['longitude'])
    num_variables = len(data.data_vars)  # Assuming all data variables are used as input

    # Reshaping into the input shape
    #input_shape = (time_steps, latitude_points, longitude_points, num_variables)
    return (latitude_points, longitude_points, num_variables) 

def pad_lr_data_to_match_hr(hr_data, lr_data):
    """
    Pad the low-resolution data to match the dimensions of high-resolution data.

    Returns:
    --------
    xr.Dataset
        Padded low-resolution dataset.
    """
    # Reindex the LR data to match the dimensions of HR data
    padded_lr_data = lr_data.reindex(latitude=hr_data['latitude'], longitude=hr_data['longitude'])#, fill_value=-9999) #TODO: changed!!!!

    # Fill NaN values in the padded low-resolution data with the corresponding values from the original low-resolution data
    for var_name in lr_data.data_vars:
        padded_lr_data[var_name] = padded_lr_data[var_name].combine_first(lr_data[var_name])

    return padded_lr_data


def crop_era5_to_cerra(lr_data, lr_lsm_z, hr_data):
    """
    Crop the low resolution datasets to match the geographical area covered by the high resolution datasets 
    with an additional 5% coverage on each side.

    Parameters:
    - lr_data (xr.Dataset): Low resolution temperature data (t2m) with dimensions (time, latitude, longitude).
    - lr_lsm_z (xr.Dataset): Low resultion land surface mask (lsm) and orography (orog) with dimensions (time, latitude, longitude).
    - hr_data (xr.Dataset): High resolution temperature data (t2m) with dimensions (time, latitude, longitude).


    Returns:
    - lr_t2m_cropped (xr.Dataset): Cropped lr temperature data (t2m) matching CERRA dimensions.
    - lr_lsm_z_cropped (xr.Dataset): Cropped lr land surface mask (lsm) and orography (orog) matching CERRA dimensions.
    """
    # Calculate the additional 10% coverage based on hr area
    lon_increase = 0.05 * (hr_data.longitude[-1] - hr_data.longitude[0])
    lat_increase = 0.05 * (hr_data.latitude[-1] - hr_data.latitude[0])

    # Crop lr t2m data
    lr_t2m_cropped = lr_data.sel(
        longitude=slice(hr_data.longitude[0] - lon_increase, hr_data.longitude[-1] + lon_increase),
        latitude=slice(hr_data.latitude[0] - lat_increase, hr_data.latitude[-1] + lat_increase)
    )

    # Crop lr lsm and orog data
    lr_lsm_z_cropped = lr_lsm_z.sel(
        longitude=slice(hr_data.longitude[0] - lon_increase, hr_data.longitude[-1] + lon_increase),
        latitude=slice(hr_data.latitude[0] - lat_increase, hr_data.latitude[-1] + lat_increase)
    )

    return lr_t2m_cropped, lr_lsm_z_cropped


def pad_hr_to_match_lr_border(lr_data, hr_data, hr_lsm_orog):
    """
    Pad high-resolution data to match the geographical coverage of low-resolution data.
    To address the 5% difference in geographical coverage between lr and hr, the hr data needs to be padded.

    Parameters:
    - lr_data (xarray.Dataset): low-resolution dataset containing 't2m' variable, like era5
    - hr_data (xarray.Dataset): high-resolution dataset containing 't2m' variable, like cerra.
    - hr_lsm_orog (xarray.Dataset): high-resolution dataset for land-sea mask ('lsm') and
      orography ('orog') variables.

    Returns:
    - padded_hr (xarray.Dataset): Padded hr dataset with the same geographical coverage as lr for 't2m'.
    - padded_hr_lsm_orog (xarray.Dataset): Padded hr dataset with the same geographical coverage as lr
      for 'lsm' and 'orog'.
    """

    # Extract longitude and latitude values
    lr_longitude, lr_latitude = lr_data.longitude, lr_data.latitude
    hr_longitude, hr_latitude = hr_data.longitude, hr_data.latitude

    # Find longitude values in lr data but not in hr data
    missing_longitude_values = lr_longitude[~np.isin(lr_longitude, hr_longitude)]
    missing_latitude_values = lr_latitude[~np.isin(lr_latitude, hr_latitude)]

    # Create padded datasets
    padded_hr = hr_data.reindex(longitude=np.concatenate([hr_longitude, missing_longitude_values]),
                                 latitude=np.concatenate([hr_latitude, missing_latitude_values])) #, fill_value=-9999) #TODO CHANGED!!!

    padded_hr_lsm_orog = hr_lsm_orog.reindex(longitude=np.concatenate([hr_longitude, missing_longitude_values]),
                                 latitude=np.concatenate([hr_latitude, missing_latitude_values]))#, fill_value=-9999) #TODO CHANGED!!
    
    padded_hr = padded_hr.sortby(['longitude', 'latitude'])
    padded_hr_lsm_orog = padded_hr_lsm_orog.sortby(['longitude', 'latitude'])

    # Fill NaN values in the padded data with the corresponding values from the original ds
    padded_hr['t2m'] = padded_hr['t2m'].combine_first(hr_data['t2m'])
    padded_hr_lsm_orog['lsm'] = padded_hr_lsm_orog['lsm'].combine_first(hr_lsm_orog['lsm'])
    padded_hr_lsm_orog['orog'] = padded_hr_lsm_orog['orog'].combine_first(hr_lsm_orog['orog'])

    return padded_hr, padded_hr_lsm_orog
