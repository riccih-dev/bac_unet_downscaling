import matplotlib.pyplot as plt

class EvaluationVisualization:
    def __init__(self, actual_data, predicted_data):
        """
        Initialize the DownscalingVisualization class.

        Parameters:
        - actual_data (numpy.ndarray): Actual high-resolution temperature data.
        - predicted_data (numpy.ndarray): Predicted high-resolution temperature data.
        """
        self.actual_data = actual_data
        self.predicted_data = predicted_data

    def spatial_plots(self):
        """
        Create spatial plots for both predicted and actual high-resolution temperature data.
        This can help identify spatial patterns and differences.
        """
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(self.actual_data[0, :, :], cmap='coolwarm')
        plt.title("Actual Data")

        plt.subplot(1, 2, 2)
        plt.imshow(self.predicted_data[0, :, :], cmap='coolwarm')
        plt.title("Predicted Data")

        plt.suptitle("Spatial Plots")
        plt.show()

    def time_series_plots(self, location):
        """
        Generate time series plots for specific locations or regions
        to observe how well your downscaling model captures temporal variations.

        Parameters:
        - location (tuple): Coordinates (latitude, longitude) of the location to plot.
        """
        plt.figure(figsize=(12, 6))

        plt.plot(self.actual_data[:, location[0], location[1]], label='Actual Data')
        plt.plot(self.predicted_data[:, location[0], location[1]], label='Predicted Data')

        plt.title("Time Series Plots")
        plt.xlabel("Time")
        plt.ylabel("Temperature")
        plt.legend()
        plt.show()

    def difference_maps(self):
        """
        Create maps showing the differences between predicted and actual values.
        This can highlight areas where the downscaling model performs well or needs improvement.
        """
        difference_map = self.actual_data[0, :, :] - self.predicted_data[0, :, :]

        plt.figure(figsize=(8, 6))
        plt.imshow(difference_map, cmap='seismic', vmin=-5, vmax=5)
        plt.colorbar()
        plt.title("Difference Maps")
        plt.show()

    def histograms(self):
        """
        Plot histograms of predicted and actual temperature values
        to understand the distribution and identify any biases.
        """
        plt.figure(figsize=(12, 6))

        plt.hist(self.actual_data.flatten(), bins=50, alpha=0.5, label='Actual Data')
        plt.hist(self.predicted_data.flatten(), bins=50, alpha=0.5, label='Predicted Data')

        plt.title("Histograms")
        plt.xlabel("Temperature")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()



# How to use the code
'''
# actual_data_t2m and predicted_data_t2m are assumed to be provided
visualization = DownscalingVisualization(actual_data_t2m, predicted_data_t2m)
visualization.spatial_plots()
visualization.time_series_plots((latitude, longitude))
visualization.difference_maps()
visualization.histograms()
visualization.qq_plots()
'''
        
