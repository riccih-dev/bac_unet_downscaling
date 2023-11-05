# -------------- IMPORTS --------------
import tensorflow as tf
from keras.layers import (Activation, BatchNormalization, Concatenate, Conv2D,
                                     Conv2DTranspose, Input, MaxPool2D)
from keras.models import Model
from model.Encoder import Encoder
from model.Decoder import Decoder

#TODO: add Batch normalization 


class UNetModel:
    """Create and train a UNet model for downscaling."""
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = self._create_model()
        
    def create_model(self):
        """Create the UNet model architecture."""

        # TODO: auslagern in eigene config File oder aehnliches
        n= 1
        inputs = 1
        num_filters = 1
        kernel_size = 3
        strides = 0
        num_blocks = 3

        # Encoder 
        encoder = Encoder(num_filters, kernel_size)
        encoder_convs, pool = encoder.getEncoder(inputs, num_blocks)

        # Bridge between Encoder and Decoder
        bridge_conv = self.__bridge_block(n, pool, num_filters, kernel_size)

        # Decoder
        decoder = Decoder(num_filters, kernel_size, strides)
        decoder_conv = decoder.getDecoder(bridge_conv, encoder_convs, num_blocks)

        # Output
        outputs = Conv2D(1, (1,1), activation='linear')(decoder_conv)
        
        model = Model(inputs, outputs, name="downscaling_t2m_UNet")

        return model
    

    def __bridge_block(input, num_filters, kernel_size, padding='same'):
        # TODO: fuer die vollständigkeit in eigene Class geben
        conv = Conv2D(num_filters, kernel_size, activation='relu', padding=padding)(input)
        conv = Conv2D(num_filters, kernel_size, activation='relu', padding=padding)(conv)
        return conv