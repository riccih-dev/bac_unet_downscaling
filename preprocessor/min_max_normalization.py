from sklearn.preprocessing import MinMaxScaler
import numpy as np 
import xarray as xr
import pickle


class MinMaxNormalizatiton:    
    """Calculate standardized anomalies for input data."""
    def __init__(self):
        self.__variable_stats = {}

    def store_stats_to_disk(self, filename):
        """
        Store variable statistics to disk using pickle.
        """
        with open(filename, 'wb') as file:
            pickle.dump(self.__variable_stats, file)


    def load_stats_from_disk(self, filename):
        """
        Load variable statistics from disk using pickle.
        """
        try:
            with open(filename, 'rb') as file:
                self.__variable_stats = pickle.load(file)
        except FileNotFoundError:
            print(f"File '{filename}' not found. No stats loaded.")


    def normalize_t2m(self, lr_data, hr_data):
        """
        Normalize the input low-resolution and high-resolution datasets using Min-Max normalization.

        Parameters:
        -----------
        lr_data : xr.Dataset
            Low-resolution observed data.
        hr_data : xr.Dataset
            High-resolution observed data.
        var_name : str
            Name of the variable to be normalized.

        Returns:
        --------
        tuple of xr.Dataset
            Tuple containing the normalized low-resolution and high-resolution datasets using Min-Max normalization.
        """
        normalized_lr = self.__normalize(lr_data, 't2m', 'lr_t2m')
        normalized_hr = self.__normalize(hr_data, 't2m', 'hr_t2m')
        return normalized_lr, normalized_hr


    def normalize_additional_features(self, lr_data, hr_data, var_names):    
        """
        Normalize the input low-resolution (lr_data) and high-resolution (hr_data) datasets for additional features using Min-Max normalization.

        Parameters:
        -----------
        lr_data : xr.Dataset
            Low-resolution observed data.
        hr_data : xr.Dataset
            High-resolution observed data.
        var_names : list of str
            Names of the variables to be normalized.

        Returns:
        --------
        tuple of xr.Dataset
            Tuple containing the normalized low-resolution and high-resolution datasets for additional features using Min-Max normalization.
        """
        for var_name in var_names:
            lr_data = self.__normalize(lr_data, var_name[0], 'lr_'+var_name[0])
            hr_data = self.__normalize(hr_data, var_name[1], 'hr_'+var_name[1])

        return lr_data, hr_data
    

    def reset_variable_stats(self):
        """
        This method clears the stored statistics used in the normalization process.
        """
        self.__variable_stats = {}


   
    def __normalize(self, data, var_name, data_name):
        """
        Normalize the input low-resolution and high-resolution datasets for a specific variable using Min-Max normalization.

        Parameters:
        -----------
        lr_data : xr.Dataset
            Low-resolution observed data.
        hr_data : xr.Dataset
            High-resolution observed data.
        var_name : list of str
            Names of the variables to be normalized. The first element is the variable name for lr_data, and the second element is for hr_data.


        Returns:
        --------
        tuple of xr.Dataset
            Tuple containing the normalized low-resolution and high-resolution datasets for a specific variable using Min-Max normalization.
        """
        normalized_data = data.copy()

        # Extract the specified variable from the datasets
        data_variable = data[var_name]

        # Calculate min and max over all data
        xmin = float(np.min(data_variable).values)
        xmax = float(np.max(data_variable).values)

        self.__variable_stats[data_name] = {'xmin': xmin, 'xmax': xmax}

        # Custom Min-Max normalization
        normalized_variable = (data_variable - xmin) / (xmax - xmin)

        # Update the original datasets with the normalized values
        normalized_data[var_name] = normalized_variable

        return normalized_data
    
    def denormalize(self, data, var_name):
        """
        Denormalize the input data for a specific variable using Min-Max denormalization.

        Parameters:
        -----------
        data : xr.Dataset
            Normalized observed data.
        var_name : str
            Name of the variable to be denormalized.

        Returns:
        --------
        xr.Dataset
            Denormalized dataset for the specified variable.
        """
        if var_name=='orog':
            var_name ='z'

        data_name = 'lr_'+var_name 
        if data_name not in self.__variable_stats:
            raise ValueError(f"Variable {data_name} not found in variable stats. Make sure to normalize the data first.")
    
        # Retrieve the stored min and max values
        xmin = np.array(self.__variable_stats[data_name]['xmin'])
        xmax = np.array(self.__variable_stats[data_name]['xmax'])

        return data * (xmax - xmin) + xmin



    
