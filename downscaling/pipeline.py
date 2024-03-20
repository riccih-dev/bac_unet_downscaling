
from downscaling.unet import UNetModel
from downscaling.modelconfig import UNetModelConfiguration
from data_operations.data_generator import DataGenerator
from data_operations.data_loader import DataLoader
from preprocessor.standardized_anomalies import StandardizedAnomalies
from preprocessor.min_max_normalization import MinMaxNormalizatiton
from preprocessor.utility import crop_spatial_dimension, pad_lr_to_match_hr, sort_ds, combine_data, extract_t2m_at_specific_times
from evaluation.metrics import DownscalingMetrics
from evaluation.visualizer import EvaluationVisualization
from downscaling.utility import  predictions_to_xarray_t2m, find_input_shape

import numpy as np 
import tabulate
import numpy as np


class DownscalingPipeline: 
    def __init__(self, normalization_type='standardized_anomalies'):
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

        if normalization_type == "standardized_anomalies":
            self.__normalizer = StandardizedAnomalies()
        elif normalization_type == "min_max":
            self.__normalizer = MinMaxNormalizatiton()
    
        self.__normalization_type = normalization_type

    def load_climate_data(self):
        """
        Loads climate data for both CERRA and ERA5 for the time period specified in the config file.
        
        Returns:
        ---------- 
        tuple: Tuple containing Xarray datasets for CERRA and ERA5 in the following order:
            (cerra_t2m, cerra_lsm_orog, era5_t2m, era5_lsm_z).
        """
        data_loader = DataLoader()
        cerra_t2m, cerra_lsm_orog = data_loader.load_cerra_data()
        era5_t2m, era5_lsm_z = data_loader.load_era5_data()

        return cerra_t2m, cerra_lsm_orog, era5_t2m, era5_lsm_z

    def run_downscaling_pipeline(self, normalization_type, train_data, val_data, lr_test_data, hr_test_data, model_setup, filename_suffix, stats_file, result_path='./results/'):
        """
        Run the entire downscaling pipeline, including model training, prediction, and evaluation.

        Parameters:
        -----------
        normalization_type : str
            Type of normalization to be used in the downscaling pipeline.
        train_data : tuple
            Tuple containing training data and corresponding labels.
        val_data : tuple
            Tuple containing validation data and corresponding labels.
        lr_test_data : xr.Dataset
            Low-resolution test data.
        hr_test_data : xr.Dataset
            High-resolution test data.
        model_setup : dict
            Dictionary containing model setup parameters, including:
            - 'scheduler_type': Type of learning rate scheduler.
            - 'learning_rate_value': Initial learning rate value.
            - 'num_epochs': Number of training epochs.
            - 'loss_type': Type of loss function.
            - 'filters': Number of filters in the U-Net model.
            - 'batch_size': Batch size for training.
        filename_suffix : str
            Suffix to be added to the filename for saving plots.
        stats_file : str
            File path to the statistics file used for denormalization.
        result_path : str, optional
            Directory path to save the results (default is './results/').
        """ 
        train_data_generator = DataGenerator(train_data[0], train_data[1], model_setup['batch_size'])
        val_data_generator = DataGenerator(val_data[0], val_data[1], model_setup['batch_size'])

        # Training
        model = self.fit_model(
            train_generator = train_data_generator,
            val_generator = val_data_generator,
            scheduler_type = model_setup['scheduler_type'],
            learning_rate_value = model_setup['learning_rate_value'],
            num_epochs = model_setup['num_epochs'],
            loss_type = model_setup['loss_type'],
            filters = model_setup['filters']
            #, show_summary = True
        )
        self.show_training_history(filename_suffix)

        # Prediction 
        result = self.predict(lr_test_data, stats_file)
        hr_test_denormalized = self.denormalize(data=hr_test_data, stats_filename=stats_file, resolution='hr')
        self.evaluate_prediction(hr_test_denormalized, result, model_setup=model_setup, filename_suffix=filename_suffix, result_path=result_path)

        # Evaluation
        visualizer = EvaluationVisualization()
        visualizer.spatial_plots(hr_test_denormalized, result, filename_suffix)
        visualizer.difference_maps(hr_test_denormalized, result, filename_suffix)
        visualizer.comparison_histograms(hr_test_denormalized, result, filename_suffix)

    def preprocess_data(self, lr_data, hr_data, lr_lsm_z, hr_lsm_orog, stats_filename='./data/preprocessed_data', crop_region = [6.5, 42.5, 16.5, 54.0], reset_climatology_stats = True):
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
        stats_filename : str, optional
            File path to store statistics. Default is './config/stats'.
        crop_region : list, optional
            Spatial region to crop. Default is [6.5, 42.5, 16.5, 54.0].
        reset_climatology_stats : bool, optional
            Whether to reset climatology statistics. Default is True.

        Returns:
        --------
        tuple of xr.Dataset
            Tuple containing normalized low-resolution temperature data and normalized high-resolution temperature data.
        '''
        if (reset_climatology_stats):
            self.__normalizer.reset_variable_stats()

        lr_data, lr_lsm_z = sort_ds(lr_data), sort_ds(lr_lsm_z)
        hr_data, hr_lsm_orog = sort_ds(hr_data), sort_ds(hr_lsm_orog)

        # Extract temperature data at specific times
        lr_data, hr_data = extract_t2m_at_specific_times(lr_data), extract_t2m_at_specific_times(hr_data)
    
        # Crop spatial dimensions
        hr_data = crop_spatial_dimension(hr_data, crop_region)
        hr_lsm_orog = crop_spatial_dimension(hr_lsm_orog, crop_region)

        # Pad low-resolution data to match high-resolution data dimensions
        lr_data = pad_lr_to_match_hr(hr_data, lr_data)
        lr_lsm_z = pad_lr_to_match_hr(hr_lsm_orog, lr_lsm_z)

        lr_data = lr_data.sortby('latitude', ascending = False)
        hr_data = hr_data.sortby('latitude', ascending = False)
        lr_lsm_z = lr_lsm_z.sortby('latitude', ascending = False)
        hr_lsm_orog = hr_lsm_orog.sortby('latitude', ascending = False)

        # Normalize based on normalizer_type defined in constructor
        anomalies_lr_data, anomalies_hr_data = self.__normalizer.normalize_t2m(lr_data, hr_data)
        anomalies_lr_lsm_z = self.__normalizer.normalize_additional_features(lr_lsm_z, ['lsm','z'])
        combined_anomalies_lr_data = combine_data(anomalies_lr_data, anomalies_lr_lsm_z, ['lsm', 'z'])

        self.__normalizer.store_stats_to_disk(stats_filename)

        return combined_anomalies_lr_data, anomalies_hr_data

    def fit_model(self, train_generator, val_generator, scheduler_type, learning_rate_value, filters, loss_type='mae', num_epochs=50, batch_size=16, show_summary=False):
        """
        Fit the U-Net model with training data generator and validate using validation data generator.

        Parameters:
        -----------
        train_generator : DataGenerator
            Training data generator.
        val_generator : DataGenerator
            Validation data generator.
        scheduler_type : str
            Type of learning rate scheduler to be used ('constant', 'step', or 'exponential').
        learning_rate_value : float
            Initial learning rate value.
        filters : int
            Number of filters to be used in the U-Net model.
        loss_type : str, optional
            Type of loss function to be used during model compilation (default is 'mae').
        num_epochs : int, optional
            Number of training epochs (default is 50).
        batch_size : int, optional
            Batch size for training (default is 16).
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

        self.model.compile(optimizer=optimizer, loss=loss_type, run_eagerly=True)

        # Fit the model using the generators
        self.history = self.model.fit(
            x=train_generator.generate_batches(),
            steps_per_epoch = len(train_generator),
            batch_size=batch_size,
            epochs=num_epochs,
            callbacks=[callback],
            validation_data=val_generator.generate_batches(),
            validation_steps=len(val_generator)
        )

        return self.model 
     
        
    def predict(self, lr_data, stats_file):
        '''
        Downscale low-resolution temperature data using trained UNet model.

        Parameters:
        ----------
        lr_data : xr.Dataset
            Low-resolution temperature data.

        stats_file : str
            File path to the statistics file used for denormalization.

        Returns:
        ----------
        xr.Dataset
            Predicted downscaled temperature.
        '''         
        data = np.stack([lr_data[var].values for var in lr_data.data_vars], axis=-1)

        predictions_normalized = self.model.predict(data)

        prediction_norm_array = predictions_to_xarray_t2m(lr_data, predictions_normalized)
        return self.denormalize(data = prediction_norm_array, stats_filename=stats_file, resolution='lr')
    

    def denormalize(self, data, resolution, stats_filename):
        """
        Denormalize the normalized input data.

        Parameters:
        -----------
        data : xarray.Dataset
            Normalized input data containing variables to be denormalized.
        resolution : str
            Resolution of the data ('lr' for low-resolution, 'hr' for high-resolution).
        stats_filename : str
            File path to the statistics file used for denormalization.

        Returns:
        --------
        xarray.Dataset
            Denormalized dataset containing the original data values.
        """
        denormalized_data = data.copy()
        self.__normalizer.load_stats_from_disk(stats_filename)
        denormalized_data['t2m'] = self.__normalizer.denormalize(data['t2m'], 't2m', resolution)
        
        return denormalized_data


    def show_training_history(self, filename_suffix, show_graph=True):
        """
        Display a plot of training and validation loss over epochs.

        Parameters:
            filename_suffix (str): Suffix to append to the filename for saving the plot.
            show_graph (bool, optional): Whether to display the graph. Defaults to True.
        """
        visualizer = EvaluationVisualization()
        visualizer.show_training_history(self.history.history, filename_suffix, show_graph)
        
    
    def evaluate_prediction(self, y_true, y_pred,  model_setup, filename_suffix, result_path='./results/'):
        """
        Evaluate the prediction results using provided ground truth and predictions, and visualize all metrics.

        Parameters:
            y_true (xarray.DataArray): Array containing the ground truth data.
            y_pred (xarray.DataArray): Array containing the predicted data.
            model_setup (dict): Dictionary containing information about the model setup and configuration.
            filename_suffix (str): Suffix to append to the filename for saving evaluation results.
            result_path (str, optional): Path to save the evaluation results. Defaults to './results/'.
        """
        metric = DownscalingMetrics(y_true['t2m'].values, y_pred['t2m'].values)
        metric_results = metric.calculate_metrics()

        print("\nMetrics:")
        print(tabulate.tabulate(metric_results, headers='keys', tablefmt="fancy_grid"))

        if filename_suffix:
            history = self.history.history
            metric.save_evaluation_summary(filename_suffix, model_setup, history['loss'], history['val_loss'], metric_results, result_path)