
from model.unet import UNetModel
from model.modelconfig import UNetModelConfiguration
from data_loader.data_loader import DataLoader
from preprocessor.standardized_anomalies import StandardizedAnomalies
from preprocessor.min_max_normalization import MinMaxNormalizatiton
from preprocessor.preprocessor import crop_spatial_dimension, crop_era5_to_cerra, pad_lr_to_match_hr, sort_ds, combine_data, extract_t2m_at_specific_times
from evaluation.metrics import DownscalingMetrics
from visualization.evaluation_visualizer import EvaluationVisualization
from utility.utility import predictions_to_xarray_additional_features, predictions_to_xarray_t2m, prepare_data_for_model_fit, split_data, find_input_shape

import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import numpy as np 
import tensorflow as tf
import tabulate


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


    def preprocess_data(self, lr_data, hr_data, lr_lsm_z, hr_lsm_orog, stats_filename='', crop_region = [],reset_climatology_stats = True):
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

        lr_data, hr_data = extract_t2m_at_specific_times(lr_data), extract_t2m_at_specific_times(hr_data)
    
        # CROP ERA5 to match spatial area of CERRA DS
        # TODO: this step is needed ?
        lr_data, lr_lsm_z = crop_era5_to_cerra(lr_data, lr_lsm_z, hr_data)
 
        # move to preprocessor
        if crop_region is not None:
            min_lon, min_lat, max_lon, max_lat = crop_region

            hr_data = hr_data.isel(
                longitude=slice(min_lon, max_lon),
                latitude=slice(min_lat, max_lat)
            )

            hr_lsm_orog = hr_lsm_orog.isel(
                longitude=slice(min_lon, max_lon),
                latitude=slice(min_lat, max_lat)
            )

        # Crop spatial dimensions to ensure divisibility for neural network operations
        hr_data = crop_spatial_dimension(hr_data)
        hr_lsm_orog = crop_spatial_dimension(hr_lsm_orog)

        # Pad era5 data to match the dimensions of cerra using interpolation
        lr_data = pad_lr_to_match_hr(hr_data, lr_data)
        lr_lsm_z = pad_lr_to_match_hr(hr_lsm_orog, lr_lsm_z)

        # Normalize based on normalizer_type defined in constructor
        anomalies_lr_data, anomalies_hr_data = self.__normalizer.normalize_t2m(lr_data, hr_data)
        anomalies_lr_lsm_z, anomalies_hr_lsm_orog = self.__normalizer.normalize_additional_features(lr_lsm_z, hr_lsm_orog, [['lsm','lsm'], ['z', 'orog']])

        self.__normalizer.store_stats_to_disk(stats_filename)

        combined_anomalies_lr_data = combine_data(anomalies_lr_data, anomalies_lr_lsm_z, ['lsm', 'z'])
        #combined_anomalies_hr_data = combine_data(anomalies_hr_data, anomalies_hr_lsm_orog, ['lsm', 'orog'])

        return combined_anomalies_lr_data, anomalies_hr_data


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
    

    def fit_model(self, train_generator, val_generator, scheduler_type, learning_rate_value, filters, loss_type='mae', num_epochs=50, batch_size=32, show_summary=False):
        """
        Fit the U-Net model with training data generator and validate using validation data generator.

        Parameters:
        -----------
        train_generator : DataGenerator
            Training data generator.
        val_generator : DataGenerator
            Validation data generator.
        loss_type : str, optional
            Type of loss function to be used during model compilation (default is 'mae').
        num_epochs : int, optional
            Number of training epochs (default is 50).
        show_summary : bool, optional
            Whether to print the model summary (default is False).

        Returns:
        --------
        keras.models.Model
            Compiled and trained U-Net model.
        """
        # Create U-Net model
        model_service = UNetModel()
        input_shape = find_input_shape(train_generator.data)
        self.model = model_service.create_model(input_shape, filters)
        
        if(show_summary):
            self.model.summary();
        
        # Configure model parameters
        model_config = UNetModelConfiguration()
        callback = model_config.configure_callbacks(scheduler_type)
        optimizer = model_config.configure_optimizer(learning_rate_value)

        self.model.compile(optimizer=optimizer, loss=loss_type)

        # Fit the model using the generators
        self.history = self.model.fit(
            x=train_generator.generate_batches(),
            epochs=num_epochs,
            steps_per_epoch=len(train_generator),
            callbacks=[callback],
            validation_data=val_generator.generate_batches(),
            validation_steps=len(val_generator)
        )
 
        return self.model 
     
        
    def predict(self, lr_data, additional_features=False):
        '''
        downscales low-resolution temperature data using trained UNet model

        Parameters:
        ----------
        - lr_data: low-resolution data

        Returns:
        ----------
        prediced downscaled temperature

        '''        
        data = np.stack([lr_data[var].values for var in lr_data.data_vars], axis=-1)

        predictions_normalized = self.model.predict(data)

        if additional_features:
            prediction_norm_array= predictions_to_xarray_additional_features(lr_data, predictions_normalized, ['t2m', 'lsm', 'orog'])
            return  self.denormalize(prediction_norm_array, True)
        else:
            prediction_norm_array = predictions_to_xarray_t2m(lr_data, predictions_normalized)
            return self.denormalize(prediction_norm_array)
    

    def denormalize(self, data, stats_filename='', additional_features=False):
        """
        Denormalize the normalized input data.

        Parameters:
        -----------
        data : xarray.Dataset
            Normalized input data containing variables to be denormalized.
        additional_features : bool, optional
            Whether additional features like 'lsm' and 'orog' are present in the data (default is False).

        Returns:
        --------
        xarray.Dataset
            Denormalized dataset containing the original data values.

        Note:
        -----
        If `additional_features` is True, 'lsm' and 'orog' variables will be denormalized in addition to 't2m'.
        """
        denormalized_data = data.copy()
        self.__normalizer.load_stats_from_disk(stats_filename)

        if additional_features: 
            denormalized_data['lsm']= self.__normalizer.denormalize(data['lsm'], 'lsm')
            denormalized_data['orog']= self.__normalizer.denormalize(data['orog'], 'orog') 

        denormalized_data['t2m'] = self.__normalizer.denormalize(data['t2m'], 't2m')
        
        return denormalized_data


    def show_training_history(self, filename_suffix, show_graph=True):
        """
        Display a plot of training and validation loss over epochs.
        """
        visualizer = EvaluationVisualization()
        visualizer.show_training_history(self.history.history, filename_suffix, show_graph)
        
        
    def evaluate_prediction(self, y_true, y_pred):
        """Calculate and visualize all metrics"""
        metric = DownscalingMetrics(y_true['t2m'].values, y_pred['t2m'].values)

        metric_results = metric.calculate_metrics()

        print("\nMetrics:")
        print(tabulate.tabulate(metric_results, headers='keys', tablefmt="fancy_grid"))

        return metric_results

    def get_history(self):
        return self.history.history