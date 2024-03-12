import xarray as xr
import numpy as np


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
