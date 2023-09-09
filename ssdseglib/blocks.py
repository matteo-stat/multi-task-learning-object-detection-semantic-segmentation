import tensorflow as tf
from typing import Tuple

def deeplabv3plus_encoder(layer: tf.keras.layers.Layer, filters: int = 256, dilation_rates: Tuple[int, int, int] = (6, 12, 18), relu_max_value: float = 0.0) -> tf.keras.layers.Layer:
    """
    create a deeplabv3+ encoder for semantic segmentation, which should be more efficient regarding resources usage compared to deeplabv3\n
    this encoder block apply to the input layer the aspp (atrous spatial pyramid pooling) block and the pooling block\n
    the aspp block it's implemented with atrous separable convolution (depthwise convolution with dilation rate followed by a pointwise convolution)\n
    the aspp block it's composed by a pointwise convolution and three atrous separable convolution with different dilation rates\n
    the pooling block apply global average pooling to the height-width dimension, followed by a pointwise convolution\n
    the outputs from aspp block and pooling block are concatenated along the axis channel and processed by a pointwise convolution\n
    all convolutions operations are followed by batch normalization and relu6 activation

    Args:
        layer (tf.keras.layers.Layer): input layer for this block
        filters (int, optional): number of filters to use in the pointwise convolutions. Defaults to 256.
        dilation_rates (Tuple[int, int, int], optional): a tuple with three different dilation rates for the atrous convolutions. Defaults to (6, 12, 18).
        relu_max_value (float, optional): relu max value. Defaults to 0.

    Returns:
        tf.keras.layers.Layer: output from the deeplabv3plus encoder
    """
    
    # set the prefix for layers names (aspp)
    name_prefix = f'mask-encoder-aspp-'

    # aspp block - pointwise convolution branch    
    layer_pointwise = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, padding='same', use_bias=False, name=f'{name_prefix}pointwise-conv')(layer)
    layer_pointwise = tf.keras.layers.BatchNormalization(name=f'{name_prefix}pointwise-batchnorm')(layer_pointwise)
    layer_pointwise = tf.keras.layers.ReLU(max_value=relu_max_value, name=f'{name_prefix}pointwise-relu6')(layer_pointwise)

    # aspp block - 1st atrous separable convolution branch   
    layer_atrous_1 = tf.keras.layers.SeparableConv2D(filters=filters, kernel_size=3, padding='same', dilation_rate=dilation_rates[0], depth_multiplier=1, use_bias=False, name=f'{name_prefix}atrous1-sepconv')(layer)
    layer_atrous_1 = tf.keras.layers.BatchNormalization(name=f'{name_prefix}atrous1-batchnorm')(layer_atrous_1)
    layer_atrous_1 = tf.keras.layers.ReLU(max_value=relu_max_value, name=f'{name_prefix}atrous1-relu6')(layer_atrous_1)

    # aspp block - 2nd atrous separable convolution branch
    layer_atrous_2 = tf.keras.layers.SeparableConv2D(filters=filters, kernel_size=3, padding='same', dilation_rate=dilation_rates[1], depth_multiplier=1, use_bias=False, name=f'{name_prefix}atrous2-sepconv')(layer)
    layer_atrous_2 = tf.keras.layers.BatchNormalization(name=f'{name_prefix}atrous2-batchnorm')(layer_atrous_2)
    layer_atrous_2 = tf.keras.layers.ReLU(max_value=relu_max_value, name=f'{name_prefix}atrous2-relu6')(layer_atrous_2)

    # aspp block - 3rd atrous separable convolution branch
    layer_atrous_3 = tf.keras.layers.SeparableConv2D(filters=filters, kernel_size=3, padding='same', dilation_rate=dilation_rates[2], depth_multiplier=1, use_bias=False, name=f'{name_prefix}atrous3-sepconv')(layer)
    layer_atrous_3 = tf.keras.layers.BatchNormalization(name=f'{name_prefix}atrous3-batchnorm')(layer_atrous_3)
    layer_atrous_3 = tf.keras.layers.ReLU(max_value=relu_max_value, name=f'{name_prefix}atrous3-relu6')(layer_atrous_3)

    # set the prefix for layers names (pooling)
    name_prefix = f'mask-encoder-pooling-'

    # pooling block - apply global average pooling (height-width dimensions squeezed to 1x1 shape), pointwise convolution and then upsample back to the input layer height-width dimensions
    # this branch completely squeeze height-width dimensions with global average pooling down to 1x1 resolution
    # then recover the original height-width dimensions back simply using bilinear upsampling
    # this is what i understood reading the paper, but honestly i found it kind of strange, especially the bilinear upsampling
    # but.. i don't know, let's simply try it out
    # (note for me -> the upsampling rate it's fine to be equal to shape because the resolution after global average pooling is 1x1)
    upsampling_size_pooling = tuple(layer.shape[1:3])
    layer_pooling = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_last', keepdims=True, name=f'{name_prefix}globalavgpool')(layer)
    layer_pooling = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, padding='same', use_bias=False, name=f'{name_prefix}conv')(layer_pooling)
    layer_pooling = tf.keras.layers.BatchNormalization(name=f'{name_prefix}batchnorm')(layer_pooling)
    layer_pooling = tf.keras.layers.ReLU(max_value=relu_max_value, name=f'{name_prefix}relu6')(layer_pooling)
    layer_pooling = tf.keras.layers.UpSampling2D(size=upsampling_size_pooling, interpolation='bilinear', name=f'{name_prefix}upsampling')(layer_pooling)

    # set the prefix for layers names (output)
    name_prefix = f'mask-encoder-'

    # concatenate layer - concantenate the outputs from aspp block and pooling block along channel dimension
    layer_concat = tf.keras.layers.Concatenate(axis=-1, name=f'{name_prefix}concat')([layer_pointwise, layer_atrous_1, layer_atrous_2, layer_atrous_3, layer_pooling])

    # output layer - apply a pointwise convolution to the previous concatenated layer
    layer_output = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, padding='same', use_bias=False, name=f'{name_prefix}output-conv')(layer_concat)
    layer_output = tf.keras.layers.BatchNormalization(name=f'{name_prefix}output-batchnorm')(layer_output)
    layer_output = tf.keras.layers.ReLU(max_value=relu_max_value, name=f'{name_prefix}output-relu6')(layer_output)

    return layer_output

def deeplabv3plus_decoder(layer_encoder: tf.keras.layers.Layer, layer_backbone: tf.keras.layers.Layer, filters_backbone: int , filters_decoder: int, output_height_width: Tuple[int, int], output_channels: int, relu_max_value: float = 0.0) -> tf.keras.layers.Layer:
    """
    create a deeplabv3+ decoder for semantic segmentation, which consist in a simple yet effective decoder\n
    this decoder use an intermediate feature map layer from the backbone to sharpen the resolution of the encoder's output\n
    the encoder's output size it's upsampled to the size of the backbone layer with bilinear interpolation\n
    the number of channels of the backbone layer are reduced with a pointwise convolution, in order to  don't overcome informations coming to the encoder's output\n
    at this point the upsampled encoder's output and backbone layer with reduced channels are concatenated along channel dimension\n
    the concatenated layers are further processed by some convolutions\n
    the output segmentation mask it's obtained with a final convolution followed by upsampling and softmax activation

    Args:
        layer_encoder (tf.keras.layers.Layer): output layer from the encoder branch
        layer_backbone (tf.keras.layers.Layer): an intermediate feature map layer from the backbone network
        filters_backbone (int): number of convolutional filters applied to the backbone layer (this should be less than number of channels from encoder's output layer), the paper suggest 32 or 48
        filters_decoder (int): number of convolutional filters applied in the decoder after the concatenation of encoder and backbone layers
        output_height_width (Tuple[int]): height and width resolution for the output semantic segmentation mask
        output_channels (int): channels for the output semantic segmentation mask
        relu_max_value (float, optional): relu max value. Defaults to 0.

    Returns:
        tf.keras.layers.Layer: output layer for the semantic segmentation, the number of channels are equal to the number of classes    
    """
    
    # set the prefix for layers names
    name_prefix = 'mask-decoder-'

    # the encoder layer height-width dimensions are upsampled up to the backbone layer height-width dimensions
    upsampling_size_encoder = (int(layer_backbone.shape[1] / layer_encoder.shape[1]), int(layer_backbone.shape[2] / layer_encoder.shape[2]))
    layer_encoder = tf.keras.layers.UpSampling2D(size=upsampling_size_encoder, interpolation='bilinear', name=f'{name_prefix}-upsampling-encoder-output')(layer_encoder)

    # reduce the number of channels from the backbone layer using pointwise convolution
    # the idea is to sharpen the mask coming from the encoder, but we don't want to give too much weight to the backbone layer, so we reduce the number of channels
    layer_backbone = tf.keras.layers.Conv2D(filters=filters_backbone, kernel_size=1, padding='same', use_bias=False, name=f'{name_prefix}backbone-conv')(layer_backbone)
    layer_backbone = tf.keras.layers.BatchNormalization(name=f'{name_prefix}backbone-batchnorm')(layer_backbone)
    layer_backbone = tf.keras.layers.ReLU(max_value=relu_max_value, name=f'{name_prefix}backbone-relu6')(layer_backbone)

    # concatenate the upsampled encoder layer and the backbone layer along channel dimension
    layer_concat = tf.keras.layers.Concatenate(axis=-1, name=f'{name_prefix}concat')([layer_encoder, layer_backbone])

    # first refinement step - convolution
    layer = tf.keras.layers.Conv2D(filters=filters_decoder, kernel_size=3, padding='same', use_bias=False, name=f'{name_prefix}conv')(layer_concat)
    layer = tf.keras.layers.BatchNormalization(name=f'{name_prefix}conv-batchnorm')(layer)
    layer = tf.keras.layers.ReLU(max_value=relu_max_value, name=f'{name_prefix}conv-relu6')(layer)

    # second refinement step - depthwise separable convolution
    layer = tf.keras.layers.SeparableConv2D(filters=filters_decoder, kernel_size=3, padding='same', depth_multiplier=1, use_bias=False, name=f'{name_prefix}sepconv')(layer)
    layer = tf.keras.layers.BatchNormalization(name=f'{name_prefix}sepconv-batchnorm')(layer)
    layer = tf.keras.layers.ReLU(max_value=relu_max_value, name=f'{name_prefix}sepconv-relu6')(layer)

    # create the ouput semantic segmentation mask, where the number of channels are equal to the number of classes
    layer = tf.keras.layers.Conv2D(filters=output_channels, kernel_size=3, padding='same', use_bias=False, name=f'{name_prefix}output-conv')(layer)
    upsampling_size_decoder = (int(output_height_width[0] / layer.shape[1]), int(output_height_width[1] / layer.shape[2]))
    layer = tf.keras.layers.UpSampling2D(size=upsampling_size_decoder, interpolation='bilinear', name=f'{name_prefix}output-upsampling')(layer)
    layer_output = tf.keras.layers.Softmax(name='output-mask')(layer)

    return layer_output

def ssdlite(layer: tf.keras.layers.Layer, filters: int, output_channels: int, name_prefix: str, relu_max_value: float = 0.0) -> tf.keras.layers.Layer:
    """
    single-shot-detector-lite (ssdlite) block, that consinsts in a depthwise separable convolution and a reshape operation\n
    the number of filters applied should be equal to the number of default bounding boxes, multiplied by number of classes (for classification) or by number of coordinates (4 for regression)\n
    the processed feature map it's reshaped in order to match the given output channels, which should be equal to number of classes (for classification) or number of coordinates (4 for regression)\n
    this reshape operation conceptually concatenate all the prediction points on the channel dimension

    Args:
        layer (tf.keras.layers.Layer): input layer for this block
        filters (int): number of filter applied by the pointwise convolution, which should be equal to the number of default bounding boxes,
        multiplied by number of classes (for classification) or by number of coordinates (4 for regression)
        output_channels (int): should be equal to number of classes (for classification) or number of coordinates (4 for regression)
        name_prefix (str): prefix for layers names
        relu_max_value (float, optional): relu max value. Defaults to 0.

    Returns:
        tf.keras.layers.Layer: output from a ssdlite block
    """       
    layer = tf.keras.layers.SeparableConv2D(filters=filters, kernel_size=3, padding='same', depth_multiplier=1, use_bias=False, name=f'{name_prefix}sepconv')(layer)
    layer = tf.keras.layers.BatchNormalization(name=f'{name_prefix}batchnorm')(layer)
    layer = tf.keras.layers.ReLU(max_value=relu_max_value, name=f'{name_prefix}relu6')(layer)   
    layer_output = tf.keras.layers.Reshape(target_shape=(-1, output_channels), name=f'{name_prefix}reshape')(layer)

    return layer_output
