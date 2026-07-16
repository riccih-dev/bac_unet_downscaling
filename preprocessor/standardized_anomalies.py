import numpy as np
import xarray as xr
import json
import pickle

class StandardizedAnomalies:
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
        Normalize the input low-resolution and high-resolution datasets for temperature using standardized anomalies.

        Parameters:
        -----------
        lr_data : xr.Dataset
            Low-resolution observed data.
        hr_data : xr.Dataset
            High-resolution observed data.
        fit : bool, optional
            If True (default), compute the climatology (mean/std) from this data and
            store it before normalizing. If False, reuse previously fitted statistics
            instead of recomputing them.

            IMPORTANT: to avoid data leakage, only ever call this with fit=True on the
            training split. Call it with fit=False on validation/test splits so their
            values do not influence the statistics used to normalize the training data.

        Returns:
        --------
        tuple of xr.Dataset
            Tuple containing the normalized low-resolution and high-resolution datasets for temperature using standardized anomalies.
        """
        normalized_lr = self.__normalize(data=lr_data, var_name='t2m', data_name='lr_t2m', dim=['time'], fit=fit)
        normalized_hr = self.__normalize(data=hr_data, var_name='t2m', data_name='hr_t2m', dim=['time'], fit=fit)
        return normalized_lr, normalized_hr


    def normalize_additional_features(self, data, var_names, data_source='lr_', fit=True):
        """
        Normalize the input low-resolution and high-resolution datasets for additional features using standardized anomalies.

        Parameters:
        -----------
        data : xr.Dataset
            Input dataset containing the variables to be normalized.
        var_names : list of str
            Names of the variables to be normalized.
        data_source : str, optional
            Prefix to be added to the normalized variable names. Default is 'lr_'.
        fit : bool, optional
            If True (default), compute statistics from this data before normalizing.
            If False, reuse previously fitted statistics. See normalize_t2m for why
            this matters for avoiding data leakage.

        Returns:
        --------
        xr.Dataset
            Normalized dataset for additional features using Standardized Anomalies.
        """
        for var_name in var_names:
            data = self.__normalize(data=data, var_name=var_name, data_name=data_source+var_name, dim=['time', 'longitude', 'latitude'], fit=fit)

        return data
    

    def reset_variable_stats(self):
        """
        This method clears the stored statistics, including mean and standard deviation,
        for variables used in the normalization process.
        """
        self.__variable_stats = {}


    def __normalize(self, data, var_name, data_name, dim=['time'], fit=True):
        """
        Normalize the input low-resolution and high-resolution datasets for a specific variable using standardized anomalies.

        Parameters:
        -----------
        lr_data : xr.Dataset
            Low-resolution observed data.
        hr_data : xr.Dataset
            High-resolution observed data.
        var_name : list of str
            Names of the variables to be normalized. The first element is the variable name for lr_data, and the second element is for hr_data.
        fit : bool, optional
            If True, (re)compute climatology from this data. If False, reuse
            previously stored climatology for this data_name.

        Returns:
        --------
        tuple of xr.Dataset
            Tuple containing the normalized low-resolution and high-resolution datasets for a specific variable using standardized anomalies.
        """
        normalized_data = data.copy()
        data_variable = data[var_name]

        if fit:
            self.__calculate_climatology(data_variable, data_name, dim)
        elif data_name not in self.__variable_stats:
            raise ValueError(
                f"No fitted climatology for '{data_name}'. Call normalize with fit=True "
                "on the training split before calling it with fit=False on other splits."
            )

        # Standardize high- data and low-resolution data
        normalized_variable = self.__calculate_standardized_anomalies(data_variable, data_name)

        # Update the original datasets with the normalized values
        normalized_data[var_name] = normalized_variable

        return normalized_data


    def __calculate_climatology(self, data_variable, data_name, dim):
        '''
        calculates climatology (long-term average) of high-resolution data

        Parameters:
        ----------
        - hr_data: array high resolution data
        '''
        mu = data_variable.mean(dim=dim)
        sigma = data_variable.std(dim=dim)

        self.__variable_stats[data_name] = {'mu': mu, 'sigma': sigma}


    def __calculate_standardized_anomalies(self, data, data_name):
        """
        Calculate anomalies and standardize them by using climatology.

        Parameters:
        ----------
        data : xr.Dataset
            Observed data.

        Returns:
        --------
        xr.Dataset
            Standardized anomalies.
        """    
        mu = self.__variable_stats[data_name]['mu']
        sigma = self.__variable_stats[data_name]['sigma']

        # anomalies: data - climatology (mean) 
        # standardized anomalies: anomalies / deviation
        return (data - mu)/sigma
        

    def denormalize(self, anomalies, var_name, resolution):
        """
        Denormalize anomalies to obtain predicted data using climatology statistics.

        Parameters:
        -----------
        anomalies : xarray.Dataset
            Anomalies data containing variables, dimensions, and coordinates.
        var_name : str
            Name of the variable for which denormalization is performed.

        Returns:
        --------
        xarray.Dataset
            Predicted data obtained by denormalizing anomalies.
        """
        data_name = resolution+'_'+var_name 


        if data_name not in self.__variable_stats:
            raise ValueError(f"Variable {data_name} not found in variable stats. Make sure to normalize the data first.")
 
        mu = np.array(self.__variable_stats[data_name]['mu'])
        sigma = np.array(self.__variable_stats[data_name]['sigma'])

        return anomalies * sigma + mu