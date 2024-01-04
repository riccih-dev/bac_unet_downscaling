
from data_loader.data_loader import DataLoader
from config import cerra_url as cerra

class CerraDataLoader(DataLoader):
    """
    Subclass of DataLoader for loading temperature and geopotential data from CERRA dataset.
    """

    def __init__(self):
        DataLoader.__init__(self)
        self.__t2m_data_paths = cerra.t2m_urls
        self.__lsm_data_path = cerra.lsm_url
        self.__orog_data_path = cerra.orog_url

    def load_t2m_data(self):
        """
        Loads temperature (t2m) data from CERRA dataset.

        Returns:
            xr.Dataset: Xarray dataset containing temperature data.
        """
        return DataLoader._load_via_url(self, self.__t2m_data_paths)

    def load_lsm_data(self):
        """
        Loads land surface model (LSM) data from CERRA dataset.

        Returns:
            xr.Dataset: Xarray dataset containing land surface model data.
        """
        return DataLoader._load_via_url(self, self.__lsm_data_path)
    
    def load_orog_data(self):
        """
        Loads orographic data from CERRA dataset.

        Returns:
            xr.Dataset: Xarray dataset containing orographic data.
        """
        return DataLoader._load_via_url(self, self.__orog_data_path)