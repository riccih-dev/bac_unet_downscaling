from sklearn.preprocessing import MinMaxScaler


class MinMaxNormalizatiton:    
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
        return self.__normalize(lr_data, hr_data, ['t2m', 't2m'])


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
            lr_data, hr_data = self.__normalize(lr_data, hr_data, var_name)

        return lr_data, hr_data
    

    def __normalize(self, lr_data, hr_data, var_name):
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
        # Extract the specified variable from the datasets
        lr_variable = lr_data[var_name[0]]
        hr_variable = hr_data[var_name[1]]

        # Normalize the variable without storing the scaler
        min_max_scaler = MinMaxScaler()

        anomalies_lr_data = min_max_scaler.fit_transform(lr_variable)
        anomalies_hr_data = min_max_scaler.transform(hr_variable)

        # Update the original datasets with the normalized values
        lr_data[var_name[0]] = anomalies_lr_data
        hr_data[var_name[1]] = anomalies_hr_data

        return lr_data, hr_data