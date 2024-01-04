from data_loader.data_loader import DataLoader
from config import era5_url as era5

class Era5DataLoader(DataLoader):
    """
    Subclass of DataLoader for loading temperature and geopotential data from ERA5 dataset.
    """

    def __init__(self):
        DataLoader.__init__(self)
        self.__t2m_data_paths = era5.t2m_urls
        self.__lsm_data_path = era5.lsm_url
        self.__z_data_path = era5.z_url


    def load_t2m_data(self):
        """
        Loads temperature (t2m) data from ERA5 dataset.

        Returns:
            xr.Dataset: Xarray dataset containing temperature data.
        """
        return DataLoader._load_via_url(self, self.__t2m_data_paths)

    def load_lsm_data(self):
        """
        Loads land surface model (LSM) data from ERA5 dataset.

        Returns:
            xr.Dataset: Xarray dataset containing land surface model data.
        """
        return DataLoader._load_via_url(self, self.__lsm_data_path)
    
    
    def load_z_data(self):
        """
        Loads geopotential height (z) data from ERA5 dataset.

        Returns:
            xr.Dataset: Xarray dataset containing geopotential height data.
        """
        return DataLoader._load_via_url(self, self.__z_data_path)
    