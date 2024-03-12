from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed

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

    write_job = data.to_netcdf(file, compute=False) #try zarr

    print(f"Writing to {file}")
    write_job.compute()


def split_dataset(lr_data, hr_data, test_size=0.11):
    """
    Splits a given pair of xarray datasets into training, validation, and test sets.

    Parameters:
    -----------
    lr_data : xr.Dataset
        Xarray dataset containing low-resolution data.
    hr_data : xr.Dataset
        Xarray dataset containing high-resolution data.
    test_size : float, optional
        The proportion of data to include in the test split. Default is 0.2.

    Returns:
    ----------
    xr.Dataset: Xarray dataset for training (low-resolution).
    xr.Dataset: Xarray dataset for validation (low-resolution).
    xr.Dataset: Xarray dataset for testing (low-resolution).
    xr.Dataset: Xarray dataset for training (high-resolution).
    xr.Dataset: Xarray dataset for validation (high-resolution).
    xr.Dataset: Xarray dataset for testing (high-resolution).
    """
    lr_train_data, lr_val_data, lr_test_data = __split_dataset_by_time(lr_data, test_size)
    hr_train_data, hr_val_data, hr_test_data = __split_dataset_by_time(hr_data, test_size)

    return lr_train_data, lr_val_data, lr_test_data, hr_train_data, hr_val_data, hr_test_data


def __split_dataset_by_time(data, test_size):
    """
    Splits a given xarray dataset into training, validation, and test sets based on time.

    Parameters:
    -----------
    data : xr.Dataset
        Xarray dataset to be split.
    test_size : float, optional
        The proportion of data to include in the test split.

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
    train_data, val_data, test_data = Parallel(n_jobs=-1)(delayed(__extract_time_slice)((t, data)) for t in [train_time, val_time, test_time]) # type: ignore
    return train_data, val_data, test_data

def __extract_time_slice(args):
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