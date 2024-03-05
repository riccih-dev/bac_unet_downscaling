import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from itertools import chain

class SpatialVisualizer:
    def __init__(self, dataset):
        self.dataset = dataset

    def visualize_spatial_area(self):
        lon = self.dataset.longitude.values
        lat = self.dataset.latitude.values
        
        fig = plt.figure(figsize=(8, 6), edgecolor='w')
        m = Basemap(projection='cyl',
                    llcrnrlat=lat.min(), urcrnrlat=lat.max(),
                    llcrnrlon=lon.min(), urcrnrlon=lon.max())
        
        self.draw_map(m)
        
        # Show the plot
        #plt.title('Spatial Area Visualization')
        plt.show()

    def draw_map(self, m, scale=1):
        # draw a shaded-relief image
        m.shadedrelief(scale=scale)

        # Draw country borders
        m.drawcountries(linewidth=0.7, linestyle='solid', color='brown')
        
        # Specify the number of lines you want
        num_lines = 100  # Adjust this based on your preference

        # lats and longs are returned as a dictionary
        lats = m.drawparallels(np.linspace(self.dataset.latitude.min(), self.dataset.latitude.max(), num_lines))
        lons = m.drawmeridians(np.linspace(self.dataset.longitude.min(), self.dataset.longitude.max(), num_lines))


        # keys contain the plt.Line2D instances
        lat_lines = chain(*(tup[1][0] for tup in lats.items()))
        lon_lines = chain(*(tup[1][0] for tup in lons.items()))
        all_lines = chain(lat_lines, lon_lines)
        
        # cycle through these lines and set the desired style
        for line in all_lines:
            line.set(linestyle='-', alpha=0.05, color='w')



