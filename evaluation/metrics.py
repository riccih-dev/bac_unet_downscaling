from audioop import rms
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
        max_err = self.__calculate_max_error()
        bias = self.__calculate_bias()
        
        self.__rmse_percentage_range(rmse)
        self.__mae_percentage_range(mae)

        metrics_dict = {
            'RMSE': [rmse],
            'MAE': [mae],
            'Max Error': [max_err],
            'Bias': [bias]
        }

        return metrics_dict

    def __rmse_percentage_range(self,rmse):
        true_range = np.ptp(self.y_true )
        percentage_true_rmse = (rmse / true_range) * 100

        pred_range = np.ptp(self.y_pred)
        percentage_rmse_pred = (rmse / pred_range) * 100

        print('--- true: ---')
        print('min', np.min(self.y_true))
        print('max', np.max(self.y_true))
        print('range true:', true_range)
        print("rmse % true: ",percentage_true_rmse)


        print('--- pred: ---')
        print('min', np.min(self.y_pred))
        print('max', np.max(self.y_pred))
        print('range pred', pred_range) 
        print("rmse % pred: ",percentage_rmse_pred)


    def __mae_percentage_range(self,mae):
        true_range = np.ptp(self.y_true )
        percentage_true_mae = (mae / true_range) * 100

        pred_range = np.ptp(self.y_pred)
        percentage_mae_pred = (mae / pred_range) * 100
        
        print('---------')
        print("mae % true: ",percentage_true_mae)
        print("mae % pred: ",percentage_mae_pred)



    def __calculate_rmse(self):
        """
        Calculate Root Mean Squared Error (RMSE).
        Measures the average magnitude of the errors between predicted and actual values.
        """
        o = np.sqrt(np.mean((self.y_true - self.y_pred)) ** 2)
        f = np.sqrt(mean_squared_error(self.y_true, self.y_pred))

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




