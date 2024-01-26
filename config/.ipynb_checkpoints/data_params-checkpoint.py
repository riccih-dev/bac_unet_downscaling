##### ------ PARAMS for cda data loading ------
# Define time params
TIME_INTERVALS =  [
            '00:00', '03:00', '06:00',
            '09:00', '12:00', '15:00',
            '18:00', '21:00',
        ]

MONTHS = [
            '01', '02', '03',
            #'04', '05', '06',
            #'07', '08', '09',
            #'10', '11', '12',
        ]


YEARS = [
            #'2017',
            #'2018', 
            #'2019', 
            '2020' # seems like request for 2020 cerra data kills the process
        ]

DAYS = [
            '01', '02', '03',
            #'04', '05', '06',
            #'07', '08', '09',
            #'10', '11', '12',
            #'13', '14', '15',
            #'16', '17', '18',
            #'19', '20', '21',
            #'22', '23', '24',
            #'25', '26', '27',
            #'28', '29', '30',
            #'31',
        ]


# Define subregion bounds
LONGITUDE_WEST, LONGITUDE_EAST =  -15, 30 #0, 359.99 the coordiantes of cerra but cover to much (incl. north america, asia)
LATITUDE_SOUTH, LATITUDE_NORTH = 34, 74 #20.29, 75.35 

##### --- Defining Params -----------
# Define params for CERRA dataset
CERRA_PARAMS = {
        'format': 'netcdf', #grib
        'variable': [
            '2m_temperature', 'land_sea_mask', 'orography',
        ],
        'level_type': 'surface_or_atmosphere',
        'data_type': 'reanalysis',
        'product_type': 'analysis',
        'year': YEARS,
        'month': MONTHS,
        'day': DAYS,
        'time': TIME_INTERVALS,
}


# Define params for ERA5 dataset
ERA5_PARAMS = {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': [
            '2m_temperature', 'geopotential', 'land_sea_mask',
        ],
        'year': YEARS,
        'month': MONTHS,
        'day': DAYS,
        'time': TIME_INTERVALS,
        'area': [
            LATITUDE_NORTH, LONGITUDE_WEST, LATITUDE_SOUTH, LONGITUDE_EAST,
        ],
}

### --- Defining Dataset Names ------
# CERRA
DATASET_CERRA = 'reanalysis-cerra-single-levels'

# ERA5 
DATASET_ERA5 = 'reanalysis-era5-single-levels'