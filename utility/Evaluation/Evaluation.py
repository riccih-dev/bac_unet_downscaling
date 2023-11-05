import numpy as np

class Evaluation:
    """Calculate evaluation metrics for model performance."""
    def calculate_rmse(self, predicted_data, target_data):
        """Calculate Root Mean Squared Error (RMSE)."""
        rmse = np.sqrt(np.mean(np.square(predicted_data - target_data)))
        return rmse
    
    def calculate_bias(self, predicted_data, target_data):
        """Calculate Bias."""
        bias = np.mean(predicted_data - target_data)
        return bias