import numpy as np
from sklearn.preprocessing import StandardScaler

class StandardizedAnomalies:
    """Calculate standardized anomalies for input data."""
    def __init__(self):
        self.scaler = StandardScaler()
        self.__mu = 0
        self.__sigma = 0

    def calculate_climatology(self):
        '''
        calculates climatology (long-term average) of high-resolution data

        Parameters:
        ----------
        - hr_data: array high resolution data
        '''
        self.__mu = np.mean(self.__hr_data)
        self.__sigma = np.std(self.__hr_data)


    def calculate_standardized_anomalies(self, data):
        '''
        calculates anomalies and standardizes them by using climatology 
        
        Parameters:
        ----------
        - data: observed data
        '''
        if self.__mu == 0  or self.__sigma == 0 :
            raise Exception("Climatology not caclulated")
        
        # data - climatology (mean) => anomalies
        # anomalies / deviation => standardized anomalies
        return (data - self.__mu)/self.__sigma
    

    def inverse_standardization(self, data):
        '''
        converts the anomalies into absolute temperatur evalues, 
        by adding back the climatology to the orginal data

        Parameters:
        ----------
        - data: anomalies

        '''
        if self.__mu == 0  or self.__sigma == 0 :
            raise Exception("Climatology not caclulated")
        
        return data * self.__sigma * self.__mu