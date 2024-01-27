
from model.unet import UNetModel
from data_loader.data_service import split_data, find_input_shape
from data_loader.data_loader import DataLoader
from preprocessor.standardized_anomalies import StandardizedAnomalies
from preprocessor.min_max_normalization import MinMaxNormalizatiton
from preprocessor.preprocessor import crop_spatial_dimension, crop_era5_to_cerra, pad_lr_to_match_hr, sort_ds, combine_data
from evaluation.metrics import DownscalingMetrics

import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import numpy as np 


class DownscalingPipeline:
    def __init__(self, normalization_type='standardized_anomalies'):
        self.__normalization_type = normalization_type
        """
        Constructor for DownscalingPipeline class.

        Parameters:
        -----------
        normalization_type : str, optional
            The type of normalization to be applied. Default is "standardized_anomalies".
            Valid values are "standardized_anomalies" or "min_max".

        Raises:
        -------
        ValueError
            If normalization_type is not one of the valid values.
        """
        valid_normalization_types = ["standardized_anomalies", "min_max"]

        if normalization_type not in valid_normalization_types:
            raise ValueError(f"Invalid normalization_type. Supported types are {valid_normalization_types}.")


        if self.__normalization_type == "standardized_anomalies":
            self.__normalizer = StandardizedAnomalies()
        elif self.__normalization_type == "min_max":
            self.__normalizer = MinMaxNormalizatiton()
    
        self.__normalization_type = normalization_type


    def load_cerra(self):
        """
        Loads CERRA data for the time period specified in the config file.
        
        Returns:
        ---------- 
        xr.Dataset: Xarray dataset containing temperature (t2m) data from CERRA.
        xr.Dataset: Xarray dataset containing land surface model (LSM) data from CERRA.
        xr.Dataset: Xarray dataset containing orographic data from CERRA.
        """
        data_loader = DataLoader()
        cerra, cerra_lsm_orog = data_loader.load_cerra_data()

        return cerra, cerra_lsm_orog
    

    def load_era5(self):
        """
        Loads ERA5 data for the time period specified in the config file.
        
        Returns:
        ---------- 
        xr.Dataset: Xarray dataset containing temperature (t2m) data from ERA5.
        xr.Dataset: Xarray dataset containing land surface model (LSM) data from ERA5.
        xr.Dataset: Xarray dataset containing geopotential height (z) data from ERA5.
        """
        data_loader = DataLoader()
        era5, era5_lsm_orog = data_loader.load_era5_data()

        return era5, era5_lsm_orog
    
    def preprocess_data(self, lr_data, hr_data, lr_lsm_z, hr_lsm_orog, reset_climatology_stats = True):
        '''
        Performs pre-processing step by cropping spatial dimension, 
        and normalizes additional variables using standardized anomalies or min-max normalization.

        Parameters:
        -----------
        lr_data : xr.Dataset
            Low-resolution input dataset.
        hr_data : xr.Dataset
            High-resolution input dataset.
        lr_lsm_z : xr.Dataset
            Low-resolution dataset containing additional variables (e.g., lsm and z).
        hr_lsm_orog : xr.Dataset
            High-resolution dataset containing additional variables (e.g., lsm and orog).

        Returns:
        --------
        tuple of xr.Dataset
            Tuple containing normalized low-resolution temperature data, normalized high-resolution temperature data,
            normalized low-resolution additional variables, and normalized high-resolution additional variables.
        '''
        if (reset_climatology_stats):
            self.__normalizer.reset_variable_stats()

        lr_data, lr_lsm_z = sort_ds(lr_data), sort_ds(lr_lsm_z)
        hr_data, hr_lsm_orog = sort_ds(hr_data), sort_ds(hr_lsm_orog)
    
        # CROP ERA5 to match spatial area of CERRA DS
        lr_data, lr_lsm_z = crop_era5_to_cerra(lr_data, lr_lsm_z, hr_data)
 
        # Crop spatial dimensions to ensure divisibility for neural network operations
        hr_data = crop_spatial_dimension(hr_data)
        hr_lsm_orog = crop_spatial_dimension(hr_lsm_orog)

        # Pad era5 data to match the dimensions of cerra using interpolation
        lr_data = pad_lr_to_match_hr(hr_data, lr_data)
        lr_lsm_z = pad_lr_to_match_hr(hr_lsm_orog, lr_lsm_z)

        # Normalize based on normalizer_type defined in constructor
        anomalies_lr_data, anomalies_hr_data = self.__normalizer.normalize_t2m(lr_data, hr_data)
        anomalies_lr_lsm_z, anomalies_hr_lsm_orog = self.__normalizer.normalize_additional_features(lr_lsm_z, hr_lsm_orog, [['lsm','lsm'], ['z', 'orog']])

        combined_anomalies_lr_data = combine_data(anomalies_lr_data, anomalies_lr_lsm_z, ['lsm', 'z'])
        combined_anomalies_hr_data = combine_data(anomalies_hr_data, anomalies_hr_lsm_orog, ['lsm', 'orog'])

        return combined_anomalies_lr_data, combined_anomalies_hr_data


    

    def split_data(self, lr_data, hr_data):
        """
        Splits a given xarray dataset into training, validation, and test sets.

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
        lr_train_data, lr_val_data, lr_test_data = split_data(lr_data)
        hr_train_data, hr_val_data, hr_test_data = split_data(hr_data)

        return lr_train_data, lr_val_data, lr_test_data, hr_train_data, hr_val_data, hr_test_data
    

    def fit_model(self, X_train, y_train, X_val, y_val, additional_features = True, loss_type='mae', num_epochs=50, batch_size=32, show_summary=False):
        """
        Fit the U-Net model with training data and validate using validation data.

        Parameters:
        -----------
        X_train : xarray.Dataset
            Training input data with variables, dimensions, and coordinates.
        y_train : xarray.Dataset
            Training target data with variables, dimensions, and coordinates.
        X_val : xarray.Dataset
            Validation input data with variables, dimensions, and coordinates.
        y_val : xarray.Dataset
            Validation target data with variables, dimensions, and coordinates.
        loss_type : str, optional
            Type of loss function to be used during model compilation (default is 'mae').
        num_epochs : int, optional
            Number of training epochs (default is 50).
        batch_size : int, optional
            Batch size for training (default is 32).
        show_summary : bool, optional
            Whether to print the model summary (default is False).

        Returns:
        --------
        keras.models.Model
            Compiled and trained U-Net model.
        """
        model_service = UNetModel()
        input_shape = find_input_shape(X_train)
        self.model = model_service.create_model(input_shape)
        
        if(show_summary):
            self.model.summary();
        
        # FIXME: ADD Learning_Rate_Schedular as callback (Irenes Model -> Notion)
        # FIXME: ADD Optimizer as callback (Irenes Model -> Notion)
        # FIXME: ADD Early Stopping as callback (Irenes Model -> Notion)


        X_train_np = np.stack([X_train[var].values for var in X_train.data_vars], axis=-1)
        y_train_np = np.stack([y_train[var].values for var in y_train.data_vars], axis=-1)

        X_val_np = np.stack([X_val[var].values for var in X_val.data_vars], axis=-1)
        y_val_np = np.stack([y_val[var].values for var in y_val.data_vars], axis=-1)

        if additional_features: 
            output_vars = ["output_temp", "output_lsm", "output_orog"]
            y_train_dict  = {output_vars[i]: y_train_np[:, :, :, i] for i in range(len(output_vars))}
            y_val_dict = {output_vars[i]: y_val_np[:, :, :, i] for i in range(len(output_vars))}

            loss_type_dict = {output_vars[i]: loss_type for i in range(len(output_vars))}
            loss_weight_dict = {output_vars[i]: 1.0 for i in range(len(output_vars))}

            self.model.compile(loss=loss_type_dict, loss_weights=loss_weight_dict)
                               

            self.history = self.model.fit(
                x=X_train_np,
                y=y_train_dict,
                epochs=num_epochs, batch_size=batch_size,
                validation_data=(X_val_np, y_val_dict))

        else:
            self.model.compile(loss=loss_type)

            self.history = self.model.fit(
                x=X_train_np, y=y_train_np,
                epochs=num_epochs, batch_size=batch_size,
                validation_data=(X_val_np, y_val_np))

        return self.model 
        
    def predict(self, lr_data, additional_features=True):
        '''
        downscales low-resolution temperature data using trained UNet model

        Parameters:
        ----------
        - lr_data: low-resolution data

        Returns:
        ----------
        prediced downscaled temperature

        '''        
        # Standardize new low-resolution data # NOO NEED, already preprocessed 
        #data_standardized = self.__normalizer.normalize_t2m_for_prediciton(lr_data)
        #data_standardized = data_standardized['t2m']
        
        #t2m_data = lr_data['t2m']

        # peforms prediction using trained U-Net model
        #predicted_anomalies = self.model.predict(t2m_data)

        data = np.stack([lr_data[var].values for var in lr_data.data_vars], axis=-1)

        predictions = self.model.predict(data)
        predicted_data = self.predictions_to_xarray(lr_data, predictions)
        
        return self.denormalize(predicted_data)
    

    def predictions_to_xarray(self,input_data, predicted_data):
        coords = {
            'longitude': input_data['longitude'],
            'latitude': input_data['latitude'],
            'time': input_data['time']
        }

        # Create a new xarray.Dataset with the predicted data
        predicted_dataset = xr.Dataset()

        for i, variable_name in enumerate(['t2m', 'lsm', 'orog']):  # Assuming these are your variable names
            # Access the i-th element in the predicted_variable list
            predicted_data_i = predicted_data[i]

            # Extract the predicted values (assuming they are in the last dimension)
            predicted_values_i = np.squeeze(predicted_data_i, axis=-1)

            # Add a new variable to the predicted_dataset
            predicted_dataset[variable_name] = (['time', 'latitude', 'longitude'], predicted_values_i)

        # Assign coordinates
        predicted_dataset = predicted_dataset.assign_coords(coords)

        return predicted_dataset
    

    def denormalize(self, data, additional_features=True):
        denormalized_data = data.copy()
        if additional_features: 
            denormalized_data['lsm']= self.__normalizer.denormalize(data['lsm'], 'lsm')
            denormalized_data['orog']= self.__normalizer.denormalize(data['orog'], 'orog') 

        denormalized_data['t2m'] = self.__normalizer.denormalize(data['t2m'], 't2m')
        
        return denormalized_data

    # FIXME: put into visualization
    def show_training_history(self):
        """
        Display plots of training and validation loss over epochs.
        """
        # Obtain information from the history object
        training_loss = self.history.history['loss']
        validation_loss = self.history.history['val_loss']

        # Plot the training history with two plots side by side
        plt.figure(figsize=(12, 5))

        # Plot training loss
        plt.subplot(1, 2, 1)
        plt.plot(training_loss, label='Training Loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Plot validation loss
        plt.subplot(1, 2, 2)
        plt.plot(validation_loss, label='Validation Loss')
        plt.title('Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Show the plots
        plt.tight_layout()
        plt.show()
        
    def evaluate_prediction(self, y_true, y_pred):
        """Calculate and visualize all metrics"""
        metric = DownscalingMetrics(y_true['t2m'].values, y_pred)
        
        rmse = metric.calculate_rmse()
        mae = metric.calculate_mae()
        max_error = metric.calculate_max_error()
        bias = metric.calculate_bias()#
        
        metrics_dict = {
            'RMSE': [rmse],
            'MAE': [mae],
            'Max Error': [max_error],
            'Bias': [bias]
        }

        metrics_df = pd.DataFrame(metrics_dict)
        print(metrics_df)
