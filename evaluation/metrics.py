import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity

class DownscalingMetrics:
    """Calculate evaluation metrics for model performance."""

    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def calculate_rmse(self):
        """
        Calculate Root Mean Squared Error (RMSE).
        Measures the average magnitude of the errors between predicted and actual values.
        """
        # y_true is assumed to be a multi-dimensional array, flattening is performed to make it easier to calculate metrics that require one-dimensional inputs
        # TODO: check if this is needed
        return np.sqrt(mean_squared_error(self.y_true.flatten(), self.y_pred.flatten()))

    def calculate_mae(self):
        """
        Calculate Mean Absolute Error (MAE).
        Similar to RMSE but takes the absolute value of errors, providing an average absolute error.
        """
        return mean_absolute_error(self.y_true.flatten(), self.y_pred.flatten())

    # TODO: needed?
    def calculate_pearson_correlation(self):
        """
        Calculate Pearson Correlation Coefficient.
        Measures the linear correlation between predicted and actual values.
        """
        correlation_coefficient, _ = pearsonr(self.y_true.flatten(), self.y_pred.flatten())
        return correlation_coefficient

    # TODO: needed?
    def calculate_ssi(self):
        """
        Calculate Structural Similarity Index (SSI).
        Evaluates the structural similarity between the predicted and actual images.
        """
        ssi_index, _ = structural_similarity(self.y_true, self.y_pred, full=True)
        return ssi_index
    
    def calculate_bias(self, predicted_data, target_data):
        """Calculate Bias."""
        bias = np.mean(self.y_true - self.y_pred)
        return bias
    

# How to use the Code 
'''
# Create an instance of the DownscalingMetrics class
metrics_calculator = DownscalingMetrics(y_true, y_pred)

# Calculate and print metrics
rmse_result = metrics_calculator.calculate_rmse()
mae_result = metrics_calculator.calculate_mae()
pearson_result = metrics_calculator.calculate_pearson_correlation()
ssi_result = metrics_calculator.calculate_ssi()

print(f"Root Mean Squared Error (RMSE): {rmse_result}")
print(f"Mean Absolute Error (MAE): {mae_result}")
print(f"Pearson Correlation Coefficient: {pearson_result}")
print(f"Structural Similarity Index (SSI): {ssi_result}")
'''





