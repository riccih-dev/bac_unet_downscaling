import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from itertools import chain

class SpatialVisualizer:
    def __init__(self, dataset):
        self.dataset = dataset

    def visualize_spatial_area(self):
        """
        Visualize the spatial area defined by the latitude and longitude values in the dataset.
        """
        lon = self.dataset.longitude.values
        lat = self.dataset.latitude.values
        
        fig = plt.figure(figsize=(8, 6), edgecolor='w')
        m = Basemap(projection='cyl',
                    llcrnrlat=lat.min(), urcrnrlat=lat.max(),
                    llcrnrlon=lon.min(), urcrnrlon=lon.max())
        
        self.__draw_map(m)
        plt.show()

    def __draw_map(self, m, scale=1):
        """
        Draw a map with shaded relief, country borders, and grid lines.

        Parameters:
        ----------
        - m: mpl_toolkits.basemap.Basemap
            The Basemap instance for drawing the map.
        - scale: float, optional
            Scale factor for the shaded-relief image. Default is 1.
        """
        m.shadedrelief(scale=scale)
        m.drawcountries(linewidth=0.7, linestyle='solid', color='brown')
        
        # Draw latitude and longitude lines on the map
        num_lines = 100 
        lats = m.drawparallels(np.linspace(self.dataset.latitude.min(), self.dataset.latitude.max(), num_lines))
        lons = m.drawmeridians(np.linspace(self.dataset.longitude.min(), self.dataset.longitude.max(), num_lines))

        lat_lines = chain(*(tup[1][0] for tup in lats.items()))
        lon_lines = chain(*(tup[1][0] for tup in lons.items()))
        all_lines = chain(lat_lines, lon_lines)
        
        for line in all_lines:
            line.set(linestyle='-', alpha=0.05, color='w')



