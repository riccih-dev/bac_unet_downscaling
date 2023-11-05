import requests
import xarray as xr
import os

class DataLoader:
    def __init__(self):
        #TODO : add these values to config file
        self.base_url = "https://storage.ecmwf.europeanweather.cloud/Code4Earth/"
        self.file_path = './././data/'
        self.lower_year_range = 2019
        self.upper_year_range = 2022 #exclusive

        
    # TODO test this method by generating new file (also use this file for loading from cloud)
    def _generate_filenames_url(self, base_url, base_file, suffix):
        '''
        generate filenames and urls using the given base url, file and suffix for the specificed time 
        period in the config file

        Parameters
        ----------
        base_url: string, base url for loading the data
        base_file: string, base structure for file naming 
        suffix: string, suffix for the file

        Returns:
        ----------
        array with filenames, array with urls
        '''
        filenames = []
        urls = []

        for year in range(self.lower_year_range, self.upper_year_range):
            stop = False

            for month in range(1,13):
                # cerra and era5 data exist until 06-2021
                if (year == 2021 and month >6):
                    stop = True
                    break

                # add a leading zero for each month before october
                month = f"0{month}" if month < 10 else str(month)

                file = base_file+str(year)+str(month)+suffix
                filenames.append(file)

                url = base_url+file
                urls.append(url)
            
            if stop:
                break

        return filenames, urls
    
    def _load_via_url(self, urls, filenames, file_location):
        '''
        loads data from the given urls and stores them in fhe file system.

        Parameters
        ----------
        urls: array with strings, contains urls to load data 
        filenames: array with strings, contains filenames to for storing the  data
        location: string, specifieds location in filesystem for storing the data
        
        '''
        for i in range(0,len(urls)):
            response = requests.get(urls[i], allow_redirects=True)

            path =file_location+filenames[i]
            open(path, 'wb').write(response.content)


    def _load_from_file_system(self, filenames, file_location):
        '''
        loads data from filesystem

        Parameters
        ----------
        filenames: array with strings, specifies the filenames of the data which should be loaded
        location: string, specifies location in file system 

        Returns:
        ----------
        array with xarray objects of the loaded data 
        '''

        data = []

        for i in range(0, len(filenames)):
            file_path =file_location+filenames[i]

            ds = xr.open_dataset(file_path)
            data.append(ds)

        return data
    
    def _get_non_existing_filenames_urls(self, filenames, file_location, urls):
        '''
        checks whichs files do not exist in the filesystem and should therefore be downloaded from the cloud

        Parameters:
        ----------       
        filenames: array with string, filenames which should be checked
        urls: array with string, corresponding urls for the given filenames 

        Returns:
        ----------
        array with non existing filenames, array with non existing urls
        '''
        non_existing_filenames = []
        non_existing_urls = []
        
        for i, filename in enumerate(filenames):
            path = file_location+filename
            if not os.path.isfile(path):
                non_existing_filenames.append(filename)
                non_existing_urls.append(urls[i])
        
        return non_existing_filenames, non_existing_urls