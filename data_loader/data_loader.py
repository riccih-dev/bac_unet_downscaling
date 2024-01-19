import requests
import xarray as xr
import io
from config import era5_url as era5
from config import cerra_url as cerra

class DataLoader:  
    """
    DataLoader for loading temperature and addtional feature data from different datasets (ERA5, CERRA).
    """

    def __init__(self):
        self.__era5_t2m_paths = era5.t2m_url_small_set
        self.__era5_lsm_geop_path = era5.lsm_geop_url

        self.__cerra_t2m_paths = cerra.t2m_urls_small_set
        self.__cerra_lsm_orog_path = cerra.lsm_orog_url


    def __load_via_url_(self, urls):
        """
        Loads datasets from a list of URLs and concatenates them along the 'time' dimension.

        Args:
            urls (list): List of URLs pointing to dataset files.

        Returns:
            xr.Dataset: Xarray dataset containing concatenated data.
        """
        
        return xr.concat(
            [
                xr.open_dataset(io.BytesIO(requests.get(url, allow_redirects=True).content), engine='h5netcdf')
                for url in urls
            ],
            dim='time'
        )
    
    def __load_via_url(self, urls):
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



    def load_era5_data(self):
        """
        Loads ERA5 data for the time period specified in the config file.
        
        Returns:
        ---------- 
        xr.Dataset: Xarray dataset containing temperature (t2m) data from ERA5.
        xr.Dataset: Xarray dataset containing land surface model (LSM) data from ERA5.
        xr.Dataset: Xarray dataset containing geopotential height (z) data from ERA5.
        """
        era5 = self.__load_via_url(self.__era5_t2m_paths)
        era5_lsm_orog = self.__load_via_url(self.__era5_lsm_geop_path)

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
        cerra = self.__load_via_url(self.__cerra_t2m_paths)
        cerra_lsm_orog = self.__load_via_url(self.__cerra_lsm_orog_path)

        return cerra, cerra_lsm_orog
    
