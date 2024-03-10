from keras.layers import (Activation, BatchNormalization, Concatenate, Conv2D,
                          Conv2DTranspose, Input, MaxPool2D)
from keras.models import Model
from keras.regularizers import l2
from torch import tensor

class UNetModel:
    """Creation of UNet model for downscaling."""

    def create_model(self, input_shape, filters):
        """
        Create the UNet model architecture.

        Parameters:
        ----------
        - input_shape: Tuple[int], shape of the input data 
        - filters: List[int], the number of filters (channels) for each block

        Returns:
        ----------
        - Keras Model: The UNet model for downscaling temperature.
        """
        num_blocks = 4
        inputs = Input(input_shape)

        # Encoder - Extracting hierarchical features from the LR-Data (ERA5)
        encoder_feature_maps, downsampled_input = self.__encode(inputs,filters, num_blocks)

        # Bridge between Encoder and Decoder
        bridge_features = self.__conv_block(downsampled_input, filters[4])

        # Decoder - reconstruct HR from the learned features of Encoder
        decoder_conv = self.__decode(encoder_features=encoder_feature_maps, conv=bridge_features, filters=filters, num_blocks=num_blocks)
        
        outputs = Conv2D(1, (1, 1), activation='linear')(decoder_conv)
        model = Model(inputs, outputs, name="downscaling_t2m_UNet")

        return model
    

    def __encode(self, x,  filters: list, num_blocks: int = 3 , num_conv_layers: int = 2, pool_size: tuple = (2, 2)):
        """
        Encode the input tensor through multiple encoder blocks.

        Parameters:
        ----------
        - x: Input tensor
        - filters: List[int], the number of filters (channels) for each block
        - num_blocks: Int, number of encoder blocks
        - num_conv_layers: Int, number of convolutional layers per block
        - pool_size: Tuple[int], size of the pooling window

        Returns:
        ----------
        - Tuple: List of feature maps from each block and the downsampled input tensor.
        """
        # List to store feature maps of each block (later used in decoder)
        feature_maps = []
        downsampled_input = None

        # Generate multiple Encoder blocks
        for i in range(num_blocks):
            # each block consists of at least two convolutional layers and a Max Pooling Step 
            # Conv Layer - caputes pattern and relationsships in the spatial and temporal data (extracts features)
            for j in range(num_conv_layers):
              use_batch = j < num_conv_layers - 1
              x = self.__conv_block(x, filters[i], use_batch_normalization=use_batch)

            feature_maps.append(x)

            # MaxPooling - reduces spatial dimension in the data 
            downsampled_input = MaxPool2D(pool_size)(x)
            x  = downsampled_input

        return feature_maps, downsampled_input
    
    
    def __decode(self, encoder_features:list, conv, filters: list, num_blocks: int = 3 , num_conv_layers: int = 2,
                 kernel_size: tuple = (3,3), padding: str = 'same'):
        """
        Decode the downsampled input and concatenate with encoder features.

        Parameters:
        ----------
        - encoder_features: List of feature maps from encoder blocks
        - conv: Downsampled input tensor of the bridge
        - filters: List[int], the number of filters (channels) for each block
        - num_blocks: Int, number of decoder blocks
        - num_conv_layers: Int, number of convolutional layers per block
        - kernel_size: Tuple[int], size of the convolutional kernel
        - padding: Str, padding strategy for Conv2DTranspose

        Returns:
        ----------
        - Keras Tensor: The decoded tensor after the final convolutional layers.
        """
        filters.reverse()
        encoder_features.reverse()

        for i in range(num_blocks): 
            # Upsampling or Transpose Convolution - increases the spatial resolution of the feature maps 
            conv = Conv2DTranspose(filters[i+1], kernel_size = kernel_size , strides = (2,2), padding = padding)(conv)   
            
            # Skip Connections - conncets correpsonding encoder and decoder layer - to obtain spatial information
            conv = Concatenate(axis=3)([conv, encoder_features[i]])

            for j in range(num_conv_layers):
              use_batch = j < num_conv_layers - 1
              conv = self.__conv_block(conv, filters[i+1], use_batch_normalization=use_batch)

        return conv


    
    def __conv_block(self, conv, num_filters: int, kernel_size: tuple = (3, 3), activation: str = 'tanh', padding: str = 'same',
                       kernel_initializer: str = 'he_normal', reg_val = 0.001, use_batch_normalization: bool = True):
        """
        Convolutional block with optional batch normalization.

        Parameters:
        ----------
        - conv: Input tensor
        - num_filters: Int, number of filters (channels) for the convolutional layer
        - kernel_size: Tuple[int], size of the convolutional kernel
        - activation: Str, activation function
        - padding: Str, padding strategy for Conv2D
        - kernel_initializer: Str, initialization method for kernel weights
        - use_batch_normalization: Bool, whether to use batch normalization

        Returns:
        ----------
        - Keras Tensor: The tensor after convolution, batch normalization, and activation.
        """
        conv = Conv2D(num_filters, kernel_size, padding=padding,kernel_initializer=kernel_initializer, 
                      #kernel_regularizer=l2(reg_val), bias_regularizer=l2(reg_val)
                      )(conv)

        # Optionally apply batch normalization
        if use_batch_normalization:
            conv = BatchNormalization()(conv)

        # Activiation step - introduces non-linearity to the model, enables the unet to learn complex relationsships in the data
        conv = Activation(activation)(conv)

        return conv
