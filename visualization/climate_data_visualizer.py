import numpy as np
import matplotlib.pyplot as plt
from cartopy import crs as ccrs, feature as cfeature
from preprocessor.preprocessor import sort_ds
import warnings

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
    COLORMAPS = ['coolwarm', 'viridis', 'Blues']

    def __init__(self):
        # Suppress Cartopy warning about facecolor
        warnings.filterwarnings("ignore", category=UserWarning, module="cartopy.mpl.style")

    def __plot_t2m(self, ax, lon, lat, data, title):
        map_ = ax.contourf(lon, lat, data.isel(time=0), cmap='coolwarm')
        self.__set_plot(ax, map_, title, data)


    def __plot_lsm(self, ax, lon, lat, data, title, var_name='lsm'):
        map_ = ax.pcolormesh(lon, lat, data[var_name].isel(time=0), cmap='Blues', shading='auto')
        self.__set_plot(ax, map_, title, data)


    def __plot_orog(self, ax, lon, lat, data, title, var_name='orog'):
        map_ = ax.pcolormesh(lon, lat, data[var_name].isel(time=0), cmap='viridis', shading='auto')
        self.__set_plot(ax, map_, title, data)


    def plot_climate_data(self, era5, era5_lsm_z, cerra, cerra_lsm_orog):
        '''
        Plot temperature, geopotential, and land-sea mask from era5 and cerra climate data.

        Parameters:
        - era5 (xarray.Dataset): ERA5 climate data containing 't2m' variable.
        - era5_lsm_z (xarray.Dataset): ERA5 climate data containing 'lsm' and 'z' variables.
        - cerra (xarray.Dataset): CERRA climate data containing 't2m' variable.
        - cerra_lsm_orog (xarray.Dataset): CERRA climate data containing 'lsm' and 'orog' variables.

        Example:
        >>> visualizer.plot_climate_data(era5_t2m, era5_lsm_z, cerra_t2m, cerra_lsm_orog)
        '''
        era5, era5_lsm_z= sort_ds(era5), sort_ds(era5_lsm_z)
        cerra, cerra_lsm_orog = sort_ds(cerra), sort_ds(cerra_lsm_orog)

        '''
        # Crop ERA5 datasets according to CERRA longitude and latitude ranges
        era5_t2m_cropped = era5.sel(
            longitude=slice(cerra['longitude'].min(), cerra['longitude'].max()),
            latitude=slice(cerra['latitude'].min(), cerra['latitude'].max())
        )
        era5_lsm_z_cropped = era5_lsm_z.sel(
            longitude=slice(cerra_lsm_orog['longitude'].min(), cerra_lsm_orog['longitude'].max()),
            latitude=slice(cerra_lsm_orog['latitude'].min(), cerra_lsm_orog['latitude'].max())
        )
        '''

        # Create subplots for ERA5 data
        fig_era5, ax_era5 = plt.subplots(
            nrows=1, ncols=3, figsize=(18,14),
            subplot_kw={"projection": ccrs.PlateCarree()},
            gridspec_kw={'width_ratios': [1, 1, 1]}
        )

        self.__plot_t2m(ax_era5[0], era5['longitude'], era5['latitude'],
                    era5['t2m'], 'ERA5 2m Temperature')

        self.__plot_lsm(ax_era5[1], era5_lsm_z['longitude'], era5_lsm_z['latitude'],
                    era5_lsm_z, 'ERA5 Land-Sea Mask')

        self.__plot_orog(ax_era5[2], era5_lsm_z['longitude'], era5_lsm_z['latitude'],
                    era5_lsm_z, 'ERA5 Orography', 'z')

        # Create subplots for CERRA data
        fig_cerra, ax_cerra = plt.subplots(
            nrows=1, ncols=3, figsize=(18, 14),
            subplot_kw={"projection": ccrs.PlateCarree()},
            gridspec_kw={'width_ratios': [1, 1, 1]}
        )

        self.__plot_t2m(ax_cerra[0], cerra['longitude'], cerra['latitude'],
                    cerra['t2m'], 'CERRA 2m Temperature')

        self.__plot_lsm(ax_cerra[1], cerra_lsm_orog['longitude'], cerra_lsm_orog['latitude'],
                    cerra_lsm_orog, 'CERRA Land-Sea Mask')

        self.__plot_orog(ax_cerra[2], cerra_lsm_orog['longitude'], cerra_lsm_orog['latitude'],
                    cerra_lsm_orog, 'CERRA Orography')

        plt.show()




    def __set_plot(self, ax, map_, title, ds, invert_color=False):
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
        ax.set_title(title, size=15)
        ax.set_xlabel("Longitude [°E]", fontsize=10)
        ax.set_ylabel("Latitude[°N]", fontsize=10)

        lat_ticks = np.arange(ds['latitude'].min(), ds['latitude'].max(), 5)
        lon_ticks = np.arange(ds['longitude'].min(), ds['longitude'].max(), 5)
        lat_labels = [str(int(lat)) + "°N" for lat in lat_ticks]
        lon_labels = [str(int(lon)) + "°E" for lon in lon_ticks]

        ax.set_yticks(lat_ticks)
        ax.set_xticks(lon_ticks)
        ax.set_yticklabels(lat_labels, fontsize=12)
        ax.set_xticklabels(lon_labels, fontsize=12)
        ax.tick_params(axis="both", which="both", direction="out", labelsize=8)

        color = 'dodgerblue' if invert_color else 'black'

        ax.coastlines(resolution='50m', color=color, linewidth=1.0)
        ax.add_feature(cfeature.BORDERS, linewidth=0.8, color=color)

        cbar = plt.colorbar(mappable=map_, ax=ax, orientation="horizontal", pad=0.05, shrink=1)
        cbar.ax.tick_params(labelsize=8)


