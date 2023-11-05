
from model.UNet import UNetModel
from sklearn.model_selection import train_test_split
from utility.Loader.DataLoaderService import load_data
from utility.Preprocessor.StandardizedAnomalies import StandardizedAnomalies
from utility.Evaluation.Evaluation import Evaluation
from utility.Visualization.Visualizer import Visualizer

class DownscalingPipeline:
    def __init__ (self, input_shape):
        self.__anomalies_calculator = StandardizedAnomalies()
        self.__evaluator = Evaluation()
        self.__visualizer = Visualizer()

        self.input_shape = input_shape #TODO: find suitable shape (64,64,1)

        self.__lr_data = []
        self.__hr_data = []


    def run(self, num_epochs, batch_size, loss_type):
        '''
        creates and trains downscaling UNet model 
        '''
        #TODO: anpassen, an korrektes Daten laden
        self.__load_data()

        self.__preprocess_standardized()
       
        self.__fit_unet_model(loss_type, num_epochs, batch_size)
        
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
        data_standardized = self.__anomalies_calculator.calculate_standardized_anomalies(lr_data)

        # peforms prediction using trained U-Net model
        predicted_anomalies = self.model.predict(data_standardized)

        # post-processing by inversing the standardization
        downscaled_temperature = self.__anomalies_calculator.inverse_standardization(predicted_anomalies)

        return downscaled_temperature
    
    
    def evaluate_result(self, predicted_data):
        ''' 
        evaluates the predicted data using RMSE and BIAS, and visualizes the results in a plot
        '''
        rmse = self.__evaluator.calculate_rmse(predicted_data, self.hr_data)
        bias = self.__evaluator.calculate_bias(predicted_data, self.hr_data)
        self.__visualizer.plot_evaluation(rmse, 'RMSE')
        self.__visualizer.plot_evaluation(bias, 'Bias')

    def get_model_summary(self):
        self.model.summary()

    def __load_data(self):
        self.lr_data, self.hr_data = load_data()

    def __preprocess_standardized(self):
        '''
        pefroms pre-processing step by aligning low- & high-resolution data, 
        and standardizes data via standardized anomalies
        '''
        # TODO: aligning spatial grid and match time / coordinates ? 

        # Calculate climatology for observed data
        self.__anomalies_calculator.calculate_climatology(self.__hr_data)

        # Standardize high- data and low-resolution data
        self.__lr_standardized_anomalies = self.__anomalies_calculator.calculate_standardized_anomalies(self.__hr_data)
        self.__hr_standardized_anomalies = self.__anomalies_calculator.calculate_standardized_anomalies(self.__lr_data)

        # split data into train and validation sets 
        self.__X_train, self.__X_val, self.__y_train, self.__y_val = train_test_split(self.__lr_standardized_anomalies, self.__hr_standardized_anomalies, test_size=0.2)


    def __fit_unet_model(self, loss_type, num_epochs, batch_size):
        self.__modelService = UNetModel(self.input_shape)
        self.model = self.__modelService.create_model()

        self.model.compile(loss_type=loss_type)
        self.model.fit(self.__X_train, self.__y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(self.__X_val, self.__y_val))

        
