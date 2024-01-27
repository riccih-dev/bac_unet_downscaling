# -------------- IMPORTS --------------
import tensorflow as tf
from keras.layers import (Activation, BatchNormalization, Concatenate, Conv2D,
                          Conv2DTranspose, Input, MaxPool2D, UpSampling2D)
from keras.models import Model

class UNetModel:
    """Create and train a UNet model for downscaling."""

    def create_model(self, input_shape, additional_features=True):
        """
        Create the UNet model architecture.

        Parameters:
        ----------
        - input_shape: Tuple, shape of the input data 

        Returns:
        ----------
        - Keras Model: The UNet model for downscaling temperature.
        """
        num_blocks = 4
        filters = [56, 112, 224, 448, 896]

        #input tensor to the entire U-Net model.
        inputs = Input(input_shape)

        # Encoder - Extracting hierarchical features from the LR-Data (ERA5)
        encoder_feature_maps, downsampled_input = self.__encode(inputs,filters, num_blocks)

        # Bridge between Encoder and Decoder
        bridge_features = self.__conv_block(downsampled_input, filters[4])

        # Decoder - reconstruct HR from the learned features of Encoder
        decoder_conv = self.__decode(encoder_features=encoder_feature_maps, bridge_features=bridge_features, filters=filters, num_blocks=num_blocks)


        if additional_features: 
            output_temp = Conv2D(1, (1,1), activation='linear', kernel_initializer="he_normal", name="output_temp")(decoder_conv)
            output_orog = Conv2D(1, (1, 1), activation='linear', kernel_initializer="he_normal", name="output_orog")(decoder_conv)
            output_lsm = Conv2D(1, (1, 1), activation='linear', kernel_initializer="he_normal", name="output_lsm")(decoder_conv)

            model = Model(inputs, [output_temp, output_lsm, output_orog], name="t2m_downscaling_unet_with_z")
        else: 
            outputs = Conv2D(1, (1, 1), activation='linear')(decoder_conv)
            model = Model(inputs, outputs, name="downscaling_t2m_UNet")

        return model
    

    def __encode(self, x,  filters: list, num_blocks: int = 3 , num_conv_layers: int = 2, pool_size: tuple = (2, 2)):
        """
        Encode the input tensor through multiple encoder blocks.

        Parameters:
        ----------
        - x: Input tensor
        - filters: List with the number of filters (channels) for each block
        - num_blocks: Number of encoder blocks
        - num_conv_layers: Number of convolutional layers per block
        - pool_size: Tuple, size of the pooling window

        Returns:
        ----------
        - Tuple: List of feature maps from each block and the downsampled input tensor.
        """

        # List to store feature maps of each block (later used in decoder)
        feature_maps = []
        downsampled_input = None

        # generate multiple Encoder blocks (at least 3)
        for i in range(num_blocks):

            # each block consists of at least two convolutional layers and a Max Pooling Step 
            # Conv Layer - caputes pattern and relationsships in the spatial and temporal data (extracts features)
            for _ in range(num_conv_layers):
                x = self.__conv_block(x, filters[i])
            
            feature_maps.append(x)
            # MaxPooling - reduces spatial dimension in the data 
            downsampled_input = MaxPool2D(pool_size)(x)

        return feature_maps, downsampled_input
    
    
    def __decode(self, encoder_features:list, bridge_features, filters: list, num_blocks: int = 3 , num_conv_layers: int = 2,
                 kernel_size: tuple = (3,3), pool_size: tuple =(2,2), padding: str = 'same'):
        """
        Decode the downsampled input and concatenate with encoder features.

        Parameters:
        ----------
        - encoder_convs: List of feature maps from encoder blocks
        - downsampled_input: Downsampled input tensor of the bridge
        - filters: List with the number of filters (channels) for each block
        - num_blocks: Number of decoder blocks
        - num_conv_layers: Number of convolutional layers per block
        - kernel_size: Tuple, size of the convolutional kernel
        - pool_size: Tuple, size of transpose stride, expected to tbe same as in encoder
        - padding: Padding strategy for Conv2DTranspose

        Returns:
        ----------
        - Keras Tensor: The decoded tensor after the final convolutional layers.
        """
        conv = None
        filters.reverse()
        encoder_features.reverse()

        print(encoder_features)

        for i in range(num_blocks): 
            # Upsampling or Transpose Convolution - increases the spatial resolution of the feature maps 
            deconv = Conv2DTranspose(filters[i+1], kernel_size = kernel_size , strides = (2,2), padding = padding)(bridge_features)        
            
            # Skip Connections - conncets correpsonding encoder and decoder layer - to obtain spatial information
            conv = Concatenate(axis=3)([deconv, encoder_features[i]])

            #[(None, 110, 140, 448), (None, 221, 281, 448)] # stride = (1,1)
            # input_shape=[(None, 220, 280, 448), (None, 221, 281, 448)] # stride (2,2)
            # input_shape=[(None, 330, 420, 448), (None, 221, 281, 448)] # stride = (3,3)
            
            # there are at least two convolutional Layers - these capture fine grained features
            for _ in range(num_conv_layers):
                conv = self.__conv_block(conv, filters[i+1])

        return conv


    

    def __conv_block(self, conv, num_filters: int, kernel_size: tuple = (3, 3), activation: str = 'relu', padding: str = 'same',
                       kernel_initializer: str = 'he_normal', use_batch_normalization: bool = True):
        """
        Convolutional block with optional batch normalization.

        Parameters:
        ----------
        - conv: Input tensor
        - num_filters: Number of filters (channels) for the convolutional layer
        - kernel_size: Tuple, size of the convolutional kernel
        - activation: Activation function
        - padding: Padding strategy for Conv2D
        - kernel_initializer: Initialization method for kernel weights
        - use_batch_normalization: Whether to use batch normalization

        Returns:
        ----------
        - Keras Tensor: The tensor after convolution, batch normalization, and activation.
        """
        conv = Conv2D(num_filters, kernel_size, padding=padding,kernel_initializer=kernel_initializer)(conv)

        # Optionally apply batch normalization
        if use_batch_normalization:
            conv = BatchNormalization()(conv)

        # Activiation step - introduces non-linearity to the model, enables the unet to learn complex relationsships in the data
        conv = Activation(activation)(conv)

        return conv
