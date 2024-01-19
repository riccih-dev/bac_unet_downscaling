
from model.unet import UNetModel
from data_loader.data_service import split_data, crop_spatial_dimension, find_input_shape
from data_loader.data_loader import DataLoader
from preprocessor.standardized_anomalies import StandardizedAnomalies
from preprocessor.min_max_normalization import MinMaxNormalizatiton

# TODO 
# - make sure every method has a doc 
# - adjust predict => works correctly with pre/post processing

class DownscalingPipeline:
    def __init__(self, normalization_type='standardized_anomaly'):
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
        data_loader = DataLoader()
        cerra, cerra_lsm_orog = data_loader.load_cerra_data()

        return cerra, cerra_lsm_orog
    
    def load_era5(self):
        data_loader = DataLoader()
        era5, era5_lsm_orog = data_loader.load_era5_data()

        return era5, era5_lsm_orog
    
    def preprocess_data(self, lr_data, hr_data):
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
        # Crop spatial dimensions to ensure divisibility for neural network operations
        # In many CNN architectures, it's common to design the network with spatial dimensions divisible by certain factors (e.g., 32).
        # This ensures compatibility with downsampling and upsampling operations.
        lr_data = crop_spatial_dimension(lr_data)
        hr_data = crop_spatial_dimension(hr_data)

        # Normalize temperature data based on normalizer_type defined in constructor
        anomalies_lr_data, anomalies_hr_data = self.__normalizer.normalize_t2m(lr_data, hr_data)

        # Normalize additional variables (lsm and z for lr, lsm and orog for hr)
        anomalies_lr_lsm_z, anomalies_hr_lsm_orog = self.__normalizer.normalize_additional_features(lr_data, hr_data, [['lsm','lsm'], ['z', 'orog']])

        return anomalies_lr_data, anomalies_hr_data, anomalies_lr_lsm_z, anomalies_hr_lsm_orog


    

    def split_data(self, lr_data, hr_data):
        lr_train_data, lr_val_data, lr_test_data = split_data(lr_data)
        hr_train_data, hr_val_data, hr_test_data = split_data(hr_data)

        return lr_train_data, lr_val_data, lr_test_data, hr_train_data, hr_val_data, hr_test_data
    

    def fit_model(self, X_train, y_train, X_val, y_val, loss_type, num_epochs, batch_size):
        # todo default values for hyperparamters
        model_service = UNetModel()
        input_shape = find_input_shape(X_train)
        self.model = model_service.create_model(input_shape)

        self.model.compile(loss_type=loss_type)
        self.model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_val, y_val))

        return self.model 
        
    def predict(self, lr_data):
        '''
        downscales low-resolution temperature data using trained UNet model

        Parameters:
        ----------
        - lr_data: low-resolution data

        Returns:
        ----------
        prediced downscaled temperature

        '''
        # Standardize new low-resolution data
        data_standardized = self.__standardizer.calculate_standardized_anomalies(lr_data)

        # peforms prediction using trained U-Net model
        predicted_anomalies = self.model.predict(data_standardized)

        # post-processing by inversing the standardization
        downscaled_temperature = self.__standardizer.inverse_standardization(predicted_anomalies)

        return downscaled_temperature
    

    
        
