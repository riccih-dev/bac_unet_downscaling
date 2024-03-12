import matplotlib.pyplot as plt
from cartopy import crs as ccrs, feature as cfeature
import os

class EvaluationVisualization:
    def show_training_history(self, history, filename_suffix, show_graph=True):
        """
        Display a plot of training and validation loss over epochs.

        Parameters:
        -----------
        history : dict
            Dictionary containing the training and validation loss over epochs.
        filename_suffix : str
            Suffix to be added to the filename for saving the plot.
        show_graph : bool, optional
            Whether to display the graph. Default is True.
        """
        # Obtain information from the history object
        training_loss = history['loss']
        validation_loss = history['val_loss']

        # Plot training and validation loss on the same plot
        plt.figure(figsize=(8, 5))

        # Plot both training and validation loss
        plt.plot(training_loss, label='Training Loss')
        plt.plot(validation_loss, label='Validation Loss')

        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        filename = os.path.join('results', f'training_history_plot_{filename_suffix}.png')
        plt.savefig(filename)

        if show_graph:
            plt.show()
        plt.close()

    def spatial_plots(self, y, prediction,filename_suffix, show_graph=True):
        """
        Create spatial plots for both predicted and actual high-resolution temperature data.
        This can help identify spatial patterns and differences.

        Parameters:
        -----------
        y : xarray.Dataset
            Actual high-resolution temperature data.
        prediction : xarray.Dataset
            Predicted high-resolution temperature data.
        filename_suffix : str
            Suffix to be added to the filename for saving the plot.
        show_graph : bool, optional
            Whether to display the graph. Default is True.
        """
        # Calculate the mean along the time dimension
        y_mean = y.mean(dim='time')
        prediction_mean = prediction.mean(dim='time')

        # Create side-by-side subplots
        fig, axs = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': ccrs.PlateCarree()})

        # Plot actual data
        im1 = axs[0].imshow(y_mean['t2m'], cmap='coolwarm', extent=[y['longitude'].min(), y['longitude'].max(), y['latitude'].min(), y['latitude'].max()])
        axs[0].set_title("Actual Data")
        axs[0].coastlines(resolution='50m', color='black', linewidth=1.0)
        axs[0].add_feature(cfeature.BORDERS, linewidth=0.8, color='black')
        axs[0].gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

        # Plot predicted data
        im2 = axs[1].imshow(prediction_mean['t2m'], cmap='coolwarm', extent=[prediction['longitude'].min(), prediction['longitude'].max(), prediction['latitude'].min(), prediction['latitude'].max()])
        axs[1].set_title("Predicted Data")
        axs[1].coastlines(resolution='50m', color='black', linewidth=1.0)
        axs[1].add_feature(cfeature.BORDERS, linewidth=0.8, color='black')
        axs[1].gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

        # Set a common colorbar for both subplots
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im2, cax=cbar_ax, orientation='vertical')

        # Add a title for the entire figure
        plt.suptitle("Spatial Plots")

        filename = os.path.join('results', f'spatial_plot_{filename_suffix}.png')
        plt.savefig(filename)
        if show_graph:
            plt.show()

    def difference_maps(self, y, prediction,filename_suffix, show_graph=True):
        """
        Create maps showing the differences between predicted and actual values.
        This can highlight areas where the downscaling model performs well or needs improvement.

        Parameters:
        -----------
        y : xarray.Dataset
            Actual high-resolution temperature data.
        prediction : xarray.Dataset
            Predicted high-resolution temperature data.
        filename_suffix : str
            Suffix to be added to the filename for saving the plot.
        show_graph : bool, optional
            Whether to display the graph. Default is True.
        """
        # Calculate the mean along the time dimension
        y_mean = y.mean(dim='time')
        prediction_mean = prediction.mean(dim='time')

        # Calculate the difference map
        difference_map = y_mean['t2m'] - prediction_mean['t2m']

        # Create the difference map plot
        plt.figure(figsize=(8, 6))
        plt.imshow(difference_map, cmap='seismic', vmin=-5, vmax=5)
        plt.colorbar(label='Temperature Difference (°C)')
        plt.title("Difference Maps")

        filename = os.path.join('results', f'difference_map_plot_{filename_suffix}.png')
        plt.savefig(filename)

        if show_graph:
            plt.show()

        plt.close()

    def comparison_histograms(self, true_data, predicted_data, filename_suffix='', var='t2m', show_graph=True):
        """
        Compare histograms of predicted and actual temperature values to understand the distribution and identify any biases.

        Parameters:
        -----------
        true_data : xarray.Dataset
            Actual high-resolution temperature data.
        predicted_data : xarray.Dataset
            Predicted high-resolution temperature data.
        filename_suffix : str, optional
            Suffix to be added to the filename for saving the plot. Default is ''.
        var : str, optional
            Variable name for which histograms are plotted. Default is 't2m'.
        show_graph : bool, optional
            Whether to display the graph. Default is True.
        """
        # Extracting the temperature values for histograms
        true_values = true_data[var].values.flatten()
        predicted_values = predicted_data[var].values.flatten()

        # Create histograms
        plt.figure(figsize=(12, 6))
        plt.hist(true_values, bins=50, alpha=0.5, label=f'Actual {var} Data')
        plt.hist(predicted_values, bins=50, alpha=0.5, label=f'Predicted {var} Data')

        # Add labels and legend
        plt.title("Histograms")
        plt.xlabel("Temperature (°C)")
        plt.ylabel("Frequency")
        plt.legend()

        if filename_suffix != '':
            filename = os.path.join('results', f'histogram_plot_{filename_suffix}.png')
            plt.savefig(filename)
            
        if show_graph:
            plt.show()
            
        plt.close()

    def plot_histogram(self, data, var_name = 't2m'):
        """
        Plot histograms of a single dataset's temperature values to understand the distribution and identify any biases.

        Parameters:
        -----------
        data : xarray.Dataset
            High-resolution temperature data.
        var_name : str, optional
            Variable name for which histograms are plotted. Default is 't2m'.
        filename_suffix : str, optional
            Suffix to be added to the filename for saving the plot. Default is ''.
        show_graph : bool, optional
            Whether to display the graph. Default is True.
        """
        data = data[var_name].values.flatten()

        # Create histograms
        plt.figure(figsize=(12, 6))
        plt.hist(data, bins=50, alpha=0.5, label=f'{var_name} Data')

        # Add labels and legend
        plt.title("Histograms")
        plt.xlabel(f'{var_name}')
        plt.ylabel("Frequency")
        plt.legend()
            
        plt.show()
        plt.close()







