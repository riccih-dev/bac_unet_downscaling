import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from cartopy import crs as ccrs, feature as cfeature

class ClimateDataVisualizer:
    '''
    Visualizes Climate Data.

    Example Usage:
    >>> visualizer = ClimateDataVisualizer()
    >>> visualizer.plot(ds)
    '''

    ERA5_VARIABLES = ['t2m', 'z', 'lsm']
    CERRA_VARIABLES = ['t2m', 'orog', 'lsm']
    COLORMAPS = ['coolwarm', 'viridis','Blues']

    def __plot(self, ds, variables):
        # Create subplots
        fig, ax = plt.subplots(
            nrows=2, ncols=2, figsize=(40, 40),
            subplot_kw={"projection": ccrs.PlateCarree()},
            gridspec_kw={'width_ratios': [1, 1]}
        )

        ax1 = ax[0, 0]
        ax2 = ax[0, 1]
        ax3 = ax[1, 0]

        # Plot temperature, geopotential, and land-sea mask
        map1 = ax1.contourf(ds['longitude'], ds['latitude'], ds[variables[1]].isel(time=0), cmap='coolwarm')
        map2 = ax2.pcolormesh(ds['longitude'], ds['latitude'], ds[variables[1]].isel(time=0), cmap='viridis')
        map3 = ax3.pcolormesh(ds['longitude'], ds['latitude'], ds[variables[1]].isel(time=0), cmap='Blues', shading='auto')

        self.__set_plot(ax1, map1, fig, '2m Temperature', ds)
        self.__set_plot(ax2, map2, fig, 'Geopotential', ds, True)
        self.__set_plot(ax3, map3, fig, 'Land-Sea Mask', ds, True)

        # Hide the last subplot in the second row
        ax[1, 1].axis('off')

        # Show the plot
        plt.show()



    def plot_era5_data(self, ds):
        '''
        Plot temperature, geopotential, and land-sea mask from era5 climate data.

        Parameters:
        - ds (xarray.Dataset): Climate data containing 't2m', 'z', and 'lsm' variables.

        Example:
        >>> visualizer.plot(my_climate_data)
        '''
        self.__plot(ds, self.ERA5_VARIABLES)



    def plot_cerra_data(self, ds):
        '''
        Plot temperature, geopotential, and land-sea mask from cerra climate data.

        Parameters:
        - ds (xarray.Dataset): Climate data containing 't2m', 'z', and 'lsm' variables.

        Example:
        >>> visualizer.plot(my_climate_data)
        '''
        self.__plot(ds, self.CERRA_VARIABLES)

    

    def __set_plot(self, ax, map_, fig, title, ds, invert_color=False):
        '''
        Set properties for a subplot.

        Parameters:
        - ax (matplotlib.axes._subplots.AxesSubplot): Subplot to be configured.
        - map_ (matplotlib.collections.QuadMesh): Plotting object.
        - fig (matplotlib.figure.Figure): Main figure.
        - title (str): Subplot title.
        - ds (xarray.Dataset): Climate data.
        - invert_color (bool, optional): Whether to invert plot colors. Default is False.
        '''
        ax.set_title(title, size=30)
        ax.set_xlabel("Longitude [°E]", fontsize=20)
        ax.set_ylabel("Latitude[°N]", fontsize=20)

        ax.set_yticks(np.arange(ds['latitude'].min(), ds['latitude'].max(), 5.))
        ax.set_xticks(np.arange(ds['longitude'].min(), ds['longitude'].max(), 5.))
        ax.tick_params(axis="both", which="both", direction="out", labelsize=15)

        color = 'dodgerblue' if invert_color else 'black'

        ax.coastlines(resolution='50m', color=color, linewidth=1.0)
        ax.add_feature(cfeature.BORDERS, linewidth=0.8, color=color)

        cbar = fig.colorbar(mappable=map_, ax=ax, orientation="horizontal", shrink=0.8)
        cbar.ax.tick_params(labelsize=15)


