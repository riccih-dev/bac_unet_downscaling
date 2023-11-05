# -------------- IMPORTS --------------
import tensorflow as tf
from keras.layers import (Activation, BatchNormalization, Concatenate, Conv2D,
                                     Conv2DTranspose, Input, MaxPool2D)
from keras.models import Model


class Encoder:
    def __init__(self, num_filters, kernel_size, padding = 'same'):
        # TODO: set these values via init or with getEncoder ? 
        self.__num_filters = num_filters
        self.__kernel_size = kernel_size
        self.__padding = padding

    def getEncoder(self, inputData, num_blocks):
        # TODO: what do return = last pool, each conv (maybe in an array)
        convs = []
        pool = inputData

        for _ in range(num_blocks): 
            conv, pool = self.__encoderBlock(pool)
            convs.append(conv)
        
        return convs, pool

    def __encoderBlock(self, input_data):
        '''
        peforms the work of one encoder-block, by reducing the spatial resoultion while increasing the 
        number of feature channels.

        Parameters
        ----------
        - input: xarray, input data 

        Returns:
        ----------
        - feature map of convolutional layer
        - downsampled version of input

        '''

        # TODO: how many convolutional layers? 2 or more, or let it be choosen via parameters?
        conv = Conv2D(self.__num_filters, self.__kernel_size, activation='relu', padding = self.__padding)(input_data)
        conv = Conv2D(self.__num_filters, self.__kernel_size, activation='relu', padding = self.__padding)(conv)
        pool = MaxPool2D(pool_size=(2, 2))(conv)
        
        return conv, pool