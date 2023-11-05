# -------------- IMPORTS --------------
import tensorflow as tf
from keras.layers import (Activation, BatchNormalization, Concatenate, Conv2D,
                                     Conv2DTranspose, Input, MaxPool2D)
from keras.models import Model


class Decoder:
    def __init__(self, num_filters, kernel_size, strides, padding = 'same'):
        # TODO: set these values via init or with getEncoder ? 
        self.__num_filters = num_filters
        self.__kernel_size = kernel_size
        self.__padding = padding
        self.__strides = strides

    def getDecoder (self, encoder_convs, pool_conv, num_blocks):
        # TODO: returns only last conv
        conv = pool_conv

        for i in range(num_blocks): 
            conv = self.__decoderBlock(conv, encoder_convs[i])
        
        return conv

    def __decoderBlock(self, input_data, encoder_conv):
        # TODO: input data = last conv, enco
        '''
        peforms the work of one decoder-block, by increasing the spatial resolution whille reducing the number of feature
        channels.

        Parameters:
        ----------
        - input: xarray, encoded representation obtained from the encoder
        - num_filter: int, number of filters, which determines the number of generated feature maps
        - kernel_size: int, specifies the spatial size of the filters
        - stride:  Specifies the step size used to slide the filters across the input feature maps
        - padding: int, determines whether the input is padded, default = 'same'

        '''

        deconv = Conv2DTranspose(self.__num_filters, self.__kernel_size, self.__strides, self.__padding)(input_data)
        up = Concatenate()([deconv, encoder_conv])
        
        conv = Conv2D(self.__num_filters, self.__kernel_size, activation='relu', padding=self.__padding)(up)
        conv = Conv2D(self.__num_filters, self.__kernel_size, activation='relu', padding=self.__padding)(conv)

        return conv