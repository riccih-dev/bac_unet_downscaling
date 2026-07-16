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


    def normalize_t2m(self, lr_data, hr_data, fit=True):
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
        fit : bool, optional
            If True (default), compute min/max from this data and store it before
            normalizing. If False, reuse previously fitted min/max instead.

            IMPORTANT: to avoid data leakage, only ever call this with fit=True on the
            training split. Call it with fit=False on validation/test splits so their
            values do not influence the min/max used to normalize the training data.

        Returns:
        --------
        tuple of xr.Dataset
            Tuple containing the normalized low-resolution and high-resolution datasets using Min-Max normalization.
        """
        normalized_lr = self.__normalize(lr_data, 't2m', 'lr_t2m', fit=fit)
        normalized_hr = self.__normalize(hr_data, 't2m', 'hr_t2m', fit=fit)
        return normalized_lr, normalized_hr



    def normalize_additional_features(self, data, var_names, data_source='lr_', fit=True):
        """
        Normalize the input data for additional features using Min-Max normalization.

        Parameters:
        -----------
        data : xr.Dataset
            Input dataset containing the variables to be normalized.
        var_names : list of str
            Names of the variables to be normalized.
        data_source : str, optional
            Prefix to be added to the normalized variable names. Default is 'lr_'.
        fit : bool, optional
            If True (default), compute min/max from this data before normalizing.
            If False, reuse previously fitted min/max. See normalize_t2m for why this
            matters for avoiding data leakage.

        Returns:
        --------
        xr.Dataset
            Normalized dataset for additional features using Min-Max normalization.
        """
        for var_name in var_names:
            data = self.__normalize(data, var_name, data_source+var_name, fit=fit)

        return data
    

    def reset_variable_stats(self):
        """
        This method clears the stored statistics used in the normalization process.
        """
        self.__variable_stats = {}


   
    def __normalize(self, data, var_name, data_name, fit=True):
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
        fit : bool, optional
            If True, (re)compute min/max from this data. If False, reuse previously
            stored min/max for this data_name.

        Returns:
        --------
        tuple of xr.Dataset
            Tuple containing the normalized low-resolution and high-resolution datasets for a specific variable using Min-Max normalization.
        """
        normalized_data = data.copy()

        # Extract the specified variable from the datasets
        data_variable = data[var_name]

        if fit:
            # Calculate min and max over all data
            xmin = float(np.min(data_variable).values)
            xmax = float(np.max(data_variable).values)
            self.__variable_stats[data_name] = {'xmin': xmin, 'xmax': xmax}
        else:
            if data_name not in self.__variable_stats:
                raise ValueError(
                    f"No fitted min/max for '{data_name}'. Call normalize with fit=True "
                    "on the training split before calling it with fit=False on other splits."
                )
            xmin = self.__variable_stats[data_name]['xmin']
            xmax = self.__variable_stats[data_name]['xmax']

        # Custom Min-Max normalization
        normalized_variable = (data_variable - xmin) / (xmax - xmin)

        # Update the original datasets with the normalized values
        normalized_data[var_name] = normalized_variable

        return normalized_data
    
    def denormalize(self, data, var_name, resolution):
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

        data_name = resolution+'_'+var_name 
        if data_name not in self.__variable_stats:
            raise ValueError(f"Variable {data_name} not found in variable stats. Make sure to normalize the data first.")
    
        # Retrieve the stored min and max values
        xmin = np.array(self.__variable_stats[data_name]['xmin'])
        xmax = np.array(self.__variable_stats[data_name]['xmax'])

        return data * (xmax - xmin) + xmin



    
