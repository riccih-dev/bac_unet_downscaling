import requests
import xarray as xr
import io
from data.urls import era5_url as era5
from data.urls import cerra_url as cerra

class DataLoader:  
    """
    DataLoader for loading temperature and addtional feature data from different datasets (ERA5, CERRA).
    """

    def __init__(self):
        self.__era5_t2m_paths = era5.t2m_url_small_set
        self.__era5_lsm_geop_path = era5.lsm_geop_url

        self.__cerra_t2m_paths = cerra.t2m_urls_small_set
        self.__cerra_lsm_orog_path = cerra.lsm_orog_url

    def load_era5_data(self):
        """
        Loads ERA5 data for the time period specified in the config file.
        
        Returns:
        ---------- 
        xr.Dataset: Xarray dataset containing temperature (t2m) data from ERA5.
        xr.Dataset: Xarray dataset containing land surface model (LSM) data from ERA5.
        xr.Dataset: Xarray dataset containing geopotential height (z) data from ERA5.
        """
        era5 = self.load_via_url(self.__era5_t2m_paths)
        era5_lsm_orog = self.load_via_url(self.__era5_lsm_geop_path)

        return era5, era5_lsm_orog
    
    def load_cerra_data(self):
        """
        Loads CERRA data for the time period specified in the config file.
        
        Returns:
        ---------- 
        xr.Dataset: Xarray dataset containing temperature (t2m) data from CERRA.
        xr.Dataset: Xarray dataset containing land surface model (LSM) data from CERRA.
        xr.Dataset: Xarray dataset containing orographic data from CERRA.
        """
        cerra = self.load_via_url(self.__cerra_t2m_paths)
        cerra_lsm_orog = self.load_via_url(self.__cerra_lsm_orog_path)

        return cerra, cerra_lsm_orog
    

    @staticmethod
    def load_via_url(urls):
        """
        Loads datasets from a list of URLs and concatenates them along the 'time' dimension.

        Args:
            urls (list): List of URLs pointing to dataset files.

        Returns:
            xr.Dataset: Xarray dataset containing concatenated data.
        """
        
        datasets = []
        for url in urls:
            print(f"Loading dataset from URL: {url}")
            dataset = xr.open_dataset(io.BytesIO(requests.get(url, allow_redirects=True).content), engine='h5netcdf')
            datasets.append(dataset)

        return xr.concat(datasets, dim='time')
    
    @staticmethod
    def load_from_disk(file_name, file_path="./data/climate_data/"):
        """
        Load climate data from disk.

        Parameters:
        - file_path (str): Path to the directory containing the file. Default is "data".
        - file_name (str): Name of the file.

        Returns:
        - xr.Dataset: Loaded climate data.
        """
        file_path = file_path + file_name + ".nc"
        return xr.open_dataset(file_path)
    
    
