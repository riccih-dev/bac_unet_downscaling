
import xarray as xr
from sklearn.model_selection import train_test_split

# TODO
# - create a better way to crop data (form each border equally)
# - adjust splitting so it workds with correct years + test it 

def split_data(data, test_size=0.2):
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


def crop_spatial_dimension(data, divisible_factor=32):
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

