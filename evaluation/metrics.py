import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error,mean_absolute_error,max_error

class DownscalingMetrics:
    """Calculate evaluation metrics for model performance."""

    def __init__(self, y_true, y_pred):
        # flattening is performed to make it easier to calculate metrics for multidimensional array
        self.y_true = y_true.reshape(-1)
        self.y_pred = y_pred.reshape(-1)

    def calculate_metrics(self): 
        """Calculate all metrics"""        
        rmse = self.__calculate_rmse()
        mae = self.__calculate_mae()
        max_error = self.__calculate_max_error()
        bias = self.__calculate_bias()
        
        metrics_dict = {
            'RMSE': [rmse],
            'MAE': [mae],
            'Max Error': [max_error],
            'Bias': [bias]
        }

        return metrics_dict


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


    def __calculate_max_error(self):
        """
        Calculate the largest absolute difference between any predicted and actual values. 
        Indicates the maximum error made by the model.
        """
        return max_error(self.y_true, self.y_pred)

    
    def __calculate_bias(self):
        """Calculate Bias."""
        return np.mean(self.y_true - self.y_pred)




