import numpy as np

class StandardizedAnomalies:
    """Calculate standardized anomalies for input data."""
    def __init__(self):
        self.__variable_stats = {}


    def normalize_t2m(self, lr_data, hr_data):
        """
        Normalize the input low-resolution and high-resolution datasets for temperature using standardized anomalies.

        Parameters:
        -----------
        lr_data : xr.Dataset
            Low-resolution observed data.
        hr_data : xr.Dataset
            High-resolution observed data.

        Returns:
        --------
        tuple of xr.Dataset
            Tuple containing the normalized low-resolution and high-resolution datasets for temperature using standardized anomalies.
        """
        return self.__normalize(lr_data, hr_data, ['t2m', 't2m'])
    

    def normalize_additional_features(self, lr_data, hr_data, var_names):
        """
        Normalize the input low-resolution and high-resolution datasets for additional features using standardized anomalies.

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
            Tuple containing the normalized low-resolution and high-resolution datasets for additional features using standardized anomalies.
        """
        for var_name in var_names:
            lr_data, hr_data = self.__normalize(lr_data, hr_data, var_name)

        return lr_data, hr_data
    
    def reset_variable_stats(self):
        """
        This method clears the stored statistics, including mean and standard deviation,
        for variables used in the normalization process.
        """
        self.__variable_stats = {}

    def __normalize(self, lr_data, hr_data, var_name):
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

        Returns:
        --------
        tuple of xr.Dataset
            Tuple containing the normalized low-resolution and high-resolution datasets for a specific variable using standardized anomalies.
        """
        lr_normalized, hr_normalized = lr_data.copy(), hr_data.copy()

        # Extract the specified variable from the datasets
        lr_variable = lr_data[var_name[0]]
        hr_variable = hr_data[var_name[1]]

        # Calculate climatology for high resolution data
        self.__calculate_climatology(hr_variable, var_name[1])

        # Standardize high- data and low-resolution data
        anomalies_lr_data = self.__calculate_standardized_anomalies(lr_variable, var_name[1]) #use var_name of hr, as this was used for storing climatology variables
        anomalies_hr_data = self.__calculate_standardized_anomalies(hr_variable, var_name[1])

        # Update the original datasets with the normalized values
        lr_normalized[var_name[0]] = anomalies_lr_data
        hr_normalized[var_name[1]] = anomalies_hr_data

        return lr_normalized, hr_normalized


    def __calculate_climatology(self, hr_data, var_name):
        '''
        calculates climatology (long-term average) of high-resolution data

        Parameters:
        ----------
        - hr_data: array high resolution data
        '''
        mu = np.mean(hr_data)
        sigma = np.std(hr_data)

        self.__variable_stats[var_name] = {'mu': mu, 'sigma': sigma}


    def __calculate_standardized_anomalies(self, data, var_name):
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
        mu = self.__variable_stats[var_name]['mu']
        sigma = self.__variable_stats[var_name]['sigma']

        # anomalies: data - climatology (mean) 
        # standardized anomalies: anomalies / deviation
        return (data - mu)/sigma
    
    def normalize_t2m_for_prediciton(self,lr_data, var_name='t2m'):
        lr_normalized = lr_data.copy()

        # Extract the specified variable from the datasets
        lr_variable = lr_data[var_name]
    
        anomalies_lr_data = self.__calculate_standardized_anomalies(lr_variable, var_name)

        lr_normalized[var_name] = anomalies_lr_data

        return lr_normalized
    

    def denormalize(self, anomalies, var_name='t2m'):
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
        if var_name not in self.__variable_stats:
            raise Exception(f"Climatology not calculated for variable: {var_name}")
        
        #predicted_data = anomalies.copy()
        #anomalies_variable = anomalies[var_name]

        mu = np.array(self.__variable_stats[var_name]['mu'])
        sigma = np.array(self.__variable_stats[var_name]['sigma'])

        
        #predicted_data[var_name] = data * sigma + mu

        return anomalies * sigma + mu