from utility.Loader.DataLoader import DataLoader
import os
import xarray as xr

class Era5DataLoader(DataLoader):
    def __init__(self):
        DataLoader.__init__(self)
        self.era5_base_url = self.base_url+"netCDF_era5_2019_2021/"    
        self.base_file = "t2m_era5_"
        self.suffix = "_025deg.nc"
        self.filenames = []
        self.urls = []

        self.additional_data_url = self.base_url+"t2m_era5_lsm_geop_201801_025deg.nc"
        self.additional_data_filename = "t2m_era5_geop.nc"


    def load_data(self):
        '''
        loads era5 data 

        Returns:
        ----------
        array with xarray objects of the cerra data
        '''
        self.__generateFileNames()

        # check which files do not exist & load them from the cloud
        non_existing_filenames, non_existing_urls = DataLoader._get_non_existing_filenames_urls(self, self.filenames, self.file_path, self.urls)
        
        if non_existing_filenames:
            DataLoader._load_via_url(self, non_existing_urls, non_existing_filenames, self.file_path)

        data = DataLoader._load_from_file_system(self, self.filenames, self.file_path)

        return data
            
    
    def __generateFileNames(self):
        self.filenames, self.urls = DataLoader._generate_filenames_url(self, self.era5_base_url, self.base_file, self.suffix)


    def enrich_data(self, era5_data):
        '''
        adds additional information to the loaded cerra data. Additional informations is given as further dater variables
        (lsm and z).

        Parameters:
        ----------
        era5_data: list with xarrays, containing era5_data for different time periods

        
        Returns:
        ----------
        array with xarray objects of the enriched cerra data objects

        '''
        file_path =self.file_path+self.additional_data_filename

        # check if file with additonal data variables exists already
        if not os.path.isfile(file_path):
            DataLoader._load_via_url(self, [self.additional_data_url], [self.additional_data_filename], self.file_path)
        
        additional_data = xr.open_dataset(file_path)
        
        # drop time dimension (not needed since only one time period)
        additional_data = additional_data.isel(time=0)

        # extract z and lsm data variables
        z_data = additional_data['z'].drop_vars('time')
        lsm_data = additional_data['lsm'].drop_vars('time')

        enriched_data = []
        for data in era5_data:
            z = z_data.expand_dims(time=data['time'])
            lsm = lsm_data.expand_dims(time=data['time'])

            data = data.assign(z=(["time", "latitude", "longitude", ], z.data), lsm=(["time", "latitude", "longitude"], lsm.data))

            enriched_data.append(data)

        return enriched_data



        
