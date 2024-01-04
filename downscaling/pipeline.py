
from model.unet import UNetModel
from data_loader.data_service import load_data, split_data, crop_data, split_data
from preprocessor.standardized_anomalies import StandardizedAnomalies
import xarray as xr

# TODO 
# -  adjust load_data 
#   - remove concat from load_data as soon as tested if it works in data_loader correctly
#   - use paramater to adjust if stored to disk or even better use methods from data_service directly in experiment to decide whether store/load it
# - preprocess_data
#   - is it necessary to align spatial grid and match time / coordinates (i guess not - but verify to make sure)
#   - add info why cropping is necessary 
#   - make sure standardization with anomalies works correctly 
#   - create possibility to decide with preprocessing should be used (SA, or normal)
# - make sure every method has a doc 

class DownscalingPipeline:
    def __init__ (self, input_shape):
        self.__input_shape = input_shape #TODO: find suitable shape (64,64,1)

    def load_data(self):
        era5_list, cerra_list = load_data()

        cerra_ds = xr.concat(cerra_list, dim='time')
        era5_ds = xr.concat(era5_list, dim='time')

        return cerra_ds, era5_ds
    
    def preprocess_data(self, lr_data, hr_data):
        '''
        pefroms pre-processing step by aligning low- & high-resolution data, 
        and standardizes data via standardized anomalies
        '''
        lr_data = crop_data(lr_data)
        hr_data = crop_data(hr_data)

        # Calculate climatology for observed data
        self.__standardizer = StandardizedAnomalies()
        self.__standardizer.calculate_climatology(hr_data)

        # Standardize high- data and low-resolution data
        anomalies_lr_data = self.__standardizer.calculate_standardized_anomalies(hr_data)
        anomalies_hr_data = self.__standardizer.calculate_standardized_anomalies(lr_data)

        return anomalies_lr_data, anomalies_hr_data
    

    def split_data(self, lr_data, hr_data):
        lr_train_data, lr_val_data, lr_test_data = split_data(lr_data)
        hr_train_data, hr_val_data, hr_test_data = split_data(hr_data)

        return lr_train_data, lr_val_data, lr_test_data, hr_train_data, hr_val_data, hr_test_data
    

    def fit_model(self, X_train, y_train, X_val, y_val, loss_type, num_epochs, batch_size):
        # todo default values for hyperparamters
        model_service = UNetModel()
        self.model = model_service.create_model(self.__input_shape)

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
    

    
        
