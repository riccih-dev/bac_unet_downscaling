import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error
import json
import os

class DownscalingMetrics:
    """Calculate evaluation metrics for model performance."""

    def __init__(self, y_true, y_pred):
        """
        Initialize the DownscalingMetrics class.

        Parameters:
            y_true (numpy.ndarray): Array containing the true values.
            y_pred (numpy.ndarray): Array containing the predicted values.
        """
        # flattening is performed to make it easier to calculate metrics for multidimensional array
        self.y_true = y_true.reshape(-1)
        self.y_pred = y_pred.reshape(-1)

    def calculate_metrics(self): 
        """
        Calculate evaluation metrics including RMSE, MAE, and their percentages in relation to the true value intervals.

        Returns:
            dict: Dictionary containing evaluation metrics including RMSE, MAE, and their percentages in relation to true values.
        """           
        rmse = self.__calculate_rmse()
        percentage_rmse = (rmse / np.ptp(self.y_true)) * 100

        mae = self.__calculate_mae()
        percentage_mae = (mae / np.ptp(self.y_true)) * 100

        interval_true = self.__calculate_interval(self.y_true)
        interval_pred = self.__calculate_interval(self.y_pred)

        metrics_dict = {
            'true_values_interval': [str(interval_true[0]),str(interval_true[1])],
            'predictions_interval': [str(interval_pred[0]),str(interval_pred[1])],
            'rmse': [rmse],
            'rmse_percentage_true': [percentage_rmse],
            'mae': [mae],
            'mae_percentage_true': [percentage_mae],
        }


        return metrics_dict
    

    def __calculate_interval(self, data):
        """
        Calculate the interval of the given data.

        Parameters:
            data (numpy.ndarray): Array containing the data.

        Returns:
            tuple: A tuple containing the minimum and maximum values of the data.
        """
        return np.min(data), np.max(data)
    

    def __calculate_rmse(self):
        """
        Calculate Root Mean Squared Error (RMSE).
        Measures the average magnitude of the errors between predicted and actual values.
        """
        return np.sqrt(mean_squared_error(self.y_true, self.y_pred))


    def __calculate_mae(self):
        """
        Calculate Mean Absolute Error (MAE).
        Similar to RMSE but takes the absolute value of errors, providing an average absolute error.
        """
        return mean_absolute_error(self.y_true, self.y_pred)

    def save_evaluation_summary(self, filename_suffix, model_setup, training_loss, validation_loss, metric_results, result_path):
        filename = os.path.join(result_path, f'model_and_results_{filename_suffix}.json')

        # Convert float32 values to float64 for serialization
        training_loss = [float(item) for item in training_loss]
        validation_loss = [float(item) for item in validation_loss]

        [float(item) for item in training_loss]

        for key, value in metric_results.items():
            metric_results[key] = [float(item) for item in value]

        output_data = {
            'model_setup': model_setup,
            'training_history': {
                'training_loss': training_loss,
                'validation_loss': validation_loss
            },
            'evaluation_metrics': metric_results
        }

        with open(filename, 'w') as json_file:
            json.dump(output_data, json_file, indent=4)





