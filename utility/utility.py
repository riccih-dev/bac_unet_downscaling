
import xarray as xr
import numpy as np
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import json
import os




def predictions_to_xarray_additional_features(input_data, predicted_data, var_names):
    """
    Convert predicted data with additional features to xarray.Dataset.

    Parameters:
    -----------
    input_data : xarray.Dataset
        Input data containing coordinates ('longitude', 'latitude', 'time').
    predicted_data : numpy.ndarray
        Predicted data with additional features.
    var_names : list of str
        Variable names corresponding to the additional features.

    Returns:
    --------
    xarray.Dataset
        Dataset containing the predicted data with additional features.
    """
    coords = {
        'longitude': input_data['longitude'],
        'latitude': input_data['latitude'],
        'time': input_data['time']
    }
    
    # Create a new xarray.Dataset with the predicted data
    predicted_dataset = xr.Dataset()

    for i, variable_name in enumerate(var_names): 
        # Access the i-th element in the predicted_variable list
        predicted_data_i = predicted_data[i]

        # Extract the predicted values
        predicted_values_i = np.squeeze(predicted_data_i, axis=-1)

        # Add a new variable to the predicted_dataset
        predicted_dataset[variable_name] = (['time', 'latitude', 'longitude'], predicted_values_i)

    # Assign coordinates
    predicted_dataset = predicted_dataset.assign_coords(coords)

    return predicted_dataset


def predictions_to_xarray_t2m(input_data, predicted_data):
    """
    Convert predicted temperature data to xarray.Dataset.

    Parameters:
    -----------
    input_data : xarray.Dataset
        Input data containing coordinates ('longitude', 'latitude', 'time').
    predicted_data : numpy.ndarray
        Predicted temperature data.

    Returns:
    --------
    xarray.Dataset
        Dataset containing the predicted temperature data.
    """
    predicted_data = np.squeeze(predicted_data)

    coords = {
        'longitude': input_data['longitude'],
        'latitude': input_data['latitude'],
        'time': input_data['time']
    }

    predicted_dataset = xr.Dataset(
        {'t2m': (['time', 'latitude', 'longitude'], predicted_data)},
        coords=coords
    )

    return predicted_dataset


def prepare_data_for_model_fit(dataset):
    """
    Prepare input data for model fitting.

    Parameters:
    -----------
    dataset : xarray.Dataset
        Input data containing multiple variables.

    Returns:
    --------
    numpy.ndarray
        Numpy array containing the prepared input data for model fitting.
    """
    return np.stack([dataset[var].values for var in dataset.data_vars], axis=-1)




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
    Determines the input shape for a given xarray dataset to be used as input for a model. The input shape is determined by the
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
    latitude_points = len(data['latitude'])
    longitude_points = len(data['longitude'])
    num_variables = len(data.data_vars)

    # Reshaping into the input shape
    return (latitude_points, longitude_points, num_variables) 


def save_to_json(filename_suffix, model_setup, training_loss, validation_loss, metric_results):
    filename = os.path.join('results', f'model_and_results_{filename_suffix}.json')

    # Convert float32 values to float64 for serialization
    training_loss = [float(item) for item in training_loss]
    validation_loss = [float(item) for item in validation_loss]

    [float(item) for item in training_loss]

    for key, value in metric_results.items():
        metric_results[key] = [float(item) for item in value]

    output_data = {
        'model_setup': model_setup,
        'training_history': {
            'training_loss': training_loss,
            'validation_loss': validation_loss
        },
        'evaluation_metrics': metric_results
    }

    with open(filename, 'w') as json_file:
        json.dump(output_data, json_file, indent=4)




