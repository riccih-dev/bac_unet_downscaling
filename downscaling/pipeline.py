
from model.unet import UNetModel
from model.modelconfig import UNetModelConfiguration
from data_loader.data_loader import DataLoader
from preprocessor.standardized_anomalies import StandardizedAnomalies
from preprocessor.min_max_normalization import MinMaxNormalizatiton
from preprocessor.preprocessor import crop_spatial_dimension, pad_lr_to_match_hr, sort_ds, combine_data, extract_t2m_at_specific_times, transform, reverse_transform, winsorize_outliers
from evaluation.metrics import DownscalingMetrics
from visualization.evaluation_visualizer import EvaluationVisualization
from utility.utility import predictions_to_xarray_additional_features, predictions_to_xarray_t2m, prepare_data_for_model_fit, split_data, find_input_shape

import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import numpy as np 
import tensorflow as tf
import tabulate

# importing necessary libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.preprocessing import PowerTransformer
from scipy.special import inv_boxcox

from scipy.stats.mstats import winsorize

sns.set_theme()
sns.set_palette(palette = "rainbow")



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
    
    def preprocess_data(self, lr_data, hr_data, lr_lsm_z, hr_lsm_orog, transform = False, stats_filename='./config/stats', crop_region = [6.5, 42.5, 16.5, 54.0], reset_climatology_stats = True):
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
    
        # Crop spatial dimensions to ensure divisibility for neural network operations
        hr_data = crop_spatial_dimension(hr_data, crop_region)
        hr_lsm_orog = crop_spatial_dimension(hr_lsm_orog, crop_region)

        if transform:
            lr_data = self.transform(lr_data)
            hr_data = self.transform(hr_data)

        # Pad era5 data to match the dimensions of cerra using interpolation
        lr_data = pad_lr_to_match_hr(hr_data, lr_data)
        lr_lsm_z = pad_lr_to_match_hr(hr_lsm_orog, lr_lsm_z)

        # Normalize based on normalizer_type defined in constructor
        anomalies_lr_data, anomalies_hr_data = self.__normalizer.normalize_t2m(lr_data, hr_data)
        # TODO: can be adjusted, only calc for lr add normalization
        anomalies_lr_lsm_z, anomalies_hr_lsm_orog = self.__normalizer.normalize_additional_features(lr_lsm_z, hr_lsm_orog, [['lsm','lsm'], ['z', 'orog']])

        self.__normalizer.store_stats_to_disk(stats_filename)

        combined_anomalies_lr_data = combine_data(anomalies_lr_data, anomalies_lr_lsm_z, ['lsm', 'z'])
        
        return combined_anomalies_lr_data, anomalies_hr_data
    

    def transform(self, data, feature='t2m', handle_outlier = False, print_info = False):  
            data_transformed = data.copy()
        
            # -- Transform the data --- 
            self.transformer = PowerTransformer(standardize=False, method='yeo-johnson') 
            yeojohn_df = pd.DataFrame(self.transformer.fit_transform(data[feature].values.reshape(-1,1)))
            
            data_transformed['t2m'] = xr.DataArray(yeojohn_df[0].values.reshape(data_transformed['t2m'].shape),
                                                    coords=data_transformed['t2m'].coords,
                                                    dims=data_transformed['t2m'].dims)
            
            #Lets get the Lambdas that were found
            print (self.transformer.lambdas_)

            if print_info:
                print(f"Skewness was {round(data.to_dataframe().skew()[feature],2)} before & is {round(yeojohn_df.skew()[0],2)} after Yeo-johnson transformation.")

                ev = EvaluationVisualization()
                print('Before transformation: ')
                ev.histograms_single_ds(data)
                ev.qq_plot(data)
                self.show_outliers(data)

                print('After transformation: ')
                ev.histograms_single_ds(data_transformed)
                ev.qq_plot(data_transformed)

            
             # --- Handle Outlier:  Winsorizing ---
            if handle_outlier:
                data_winsorized = winsorize(data_transformed[feature].values, limits=(0.005, 0.005))
                data_transformed['t2m']= xr.DataArray(data_winsorized,  coords=data[feature].coords,  dims=data[feature].dims,  name=feature)

                if print_info:
                    print('Before Outlier Handling: ')
                    self.show_outliers(data)

                    print('After Outlier Handling:')
                    self.show_outliers(data_transformed)
            

            return data_transformed

    def show_outliers(self, data):
        t2m_values = data['t2m'].values.flatten()

        # Identify outliers using IQR method
        q1 = np.percentile(t2m_values, 25)
        q3 = np.percentile(t2m_values, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers_indices = np.where((t2m_values < lower_bound) | (t2m_values > upper_bound))

        # Calculate the ratio of outliers
        ratio_of_outliers = len(outliers_indices[0]) / len(t2m_values) * 100
        print(f"{len(outliers_indices[0])} Outliers detected of {len(t2m_values)} - {ratio_of_outliers:.2f}%")
        if len(outliers_indices[0]) > 0:
            print(f"Outliers detected at indices: {outliers_indices[0]}")
            # Show some of the outliers (adjust the number as needed)
            print(f"Example of some outliers: {t2m_values[outliers_indices[0][:5]]}")

        # Boxplot
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=t2m_values)
        plt.title('Boxplot of the original data')
        plt.show()


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
     
        
    def predict(self, lr_data, stats_file, additional_features=False):
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
            return  self.denormalize(data = prediction_norm_array, stats_filename=stats_file, additional_features=True, resolution='lr')
        else:
            prediction_norm_array = predictions_to_xarray_t2m(lr_data, predictions_normalized)
            return self.denormalize(data = prediction_norm_array, stats_filename=stats_file, resolution='lr')
    

    def denormalize(self, data, resolution, stats_filename='', is_transformed=False, additional_features=False):
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
            denormalized_data['lsm']= self.__normalizer.denormalize(data['lsm'], 'lsm', resolution)
            denormalized_data['orog']= self.__normalizer.denormalize(data['orog'], 'orog', resolution) 

        denormalized_data['t2m'] = self.__normalizer.denormalize(data['t2m'], 't2m', resolution)

        if is_transformed:
            detransformed_t2m = pd.DataFrame(self.transformer.inverse_transform(denormalized_data['t2m'].values.reshape(-1,1)))
            denormalized_data['t2m']= xr.DataArray(detransformed_t2m[0].values.reshape(data['t2m'].shape),
                                                        coords=data['t2m'].coords,
                                                        dims=data['t2m'].dims)
        
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
        """returns the models history"""
        return self.history.history