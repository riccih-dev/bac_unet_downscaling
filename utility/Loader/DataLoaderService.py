from utility.Loader.CerraDataLoader import CerraDataLoader
from utility.Loader.Era5DataLoader import Era5DataLoader

from IPython.display import display


def load_data():
    '''
    loads era5 and cerra data for the time period specified in the config file
    
    Returns:
    ---------- 
    array with xarray objects of the loaded era5 data 
    array with xarray objects of the loaded cerra data 
    '''
    era5Loader= Era5DataLoader()
    era5_ds = era5Loader.load_data()
    era5_ds = era5Loader.enrich_data(era5_ds)

    cerraLoader = CerraDataLoader()
    cerra_ds = cerraLoader.load_data()
    cerra_ds = cerraLoader.enrich_data(cerra_ds)

    return era5_ds, cerra_ds


def test_data_load():
    era5_ds, cerra_ds = load_data()
    for data in era5_ds:
        display(data)

    for data in cerra_ds:
        display(data)