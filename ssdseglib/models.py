from typing import Tuple, Union, List
import tensorflow as tf

def mobilenetv2_block_expand(layer: tf.keras.layers.Layer, channels: int, counter_blocks: int, kernel_size: Union[int, Tuple[int, int]] = 1, strides: Union[int, Tuple[int, int]] = 1) -> tf.keras.layers.Layer:
    """
    mobilenet-v2 expand block, that consists in a separable convolution followed by batch normalization and relu6 activation\n
    this block increase the input channels by an expansion factor (channels argument expect that you pass the output channels, so input channels * expansion factor)\n

    Args:
        layer (tf.keras.layers.Layer): input layer for this block
        channels (int): number of filters applied by the pointwise convolution, you should pass the output channels you want (input channels * expansion factor)
        counter_blocks (int): counter for the the blocks, used to give proper names to the layers
        kernel_size (Union[int, Tuple[int, int]], optional): for a standard mobilenet-v2 expand block should be 1. Defaults to 1.
        strides (Union[int, Tuple[int, int]], optional): for a standard mobilenet-v2 expand block should be 1. Defaults to 1.

    Returns:
        tf.keras.layers.Layer: output layer from the mobilenet-v2 expand block
    """
    # set the prefix for layers names
    name_prefix = f'backbone-block{counter_blocks}-expand-'

    # apply in sequence pointwise convolution, batch normalization and relu6
    layer = tf.keras.layers.Conv2D(filters=channels, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False, name=f'{name_prefix}conv')(layer)
    layer = tf.keras.layers.BatchNormalization(name=f'{name_prefix}batchnorm')(layer)
    layer = tf.keras.layers.ReLU(max_value=6.0, name=f'{name_prefix}relu6')(layer)

    return layer

def mobilenetv2_block_depthwise(layer: tf.keras.layers.Layer, strides: Union[int, Tuple[int, int]], counter_blocks: int) -> tf.keras.layers.Layer:
    """
    mobilenet-v2 depthwise block, that consists in a depthwise convolution followed by batch normalization and relu6 activation\n
    this block apply the depthwise convolution, that consists of indipendent convolutions for each input channel\n

    Args:
        layer (tf.keras.layers.Layer): input layer for this block
        strides (Union[int, Tuple[int, int]]): strides for the depthwise convolution, usually in mobilenet-v2 can be 1 or 2
        counter_blocks (int): counter for the the blocks, used to give proper names to the layers

    Returns:
        tf.keras.layers.Layer: output layer from the mobilenet-v2 depthwise block
    """
    # set the prefix for layers names
    name_prefix = f'backbone-block{counter_blocks}-depthwise-'

    # apply in sequence depthwise convolution, batch normalization and relu6
    # note that mobilenet-v2 use depthwise pointwise convolution, so we have always one filter per channel in the depthwise convolution (depth_multiplier=1)
    layer = tf.keras.layers.DepthwiseConv2D(depth_multiplier=1, kernel_size=3, strides=strides, padding='same', use_bias=False, name=f'{name_prefix}conv')(layer)        
    layer = tf.keras.layers.BatchNormalization(name=f'{name_prefix}batchnorm')(layer)
    layer = tf.keras.layers.ReLU(max_value=6.0, name=f'{name_prefix}relu6')(layer)

    return layer

def mobilenetv2_block_project(layer: tf.keras.layers.Layer, channels: int, counter_blocks: int) -> tf.keras.layers.Layer:
    """
    mobilenet-v2 project block, that consist in a pointwise convolution followed by batch normalization\n
    this block reduce the number of channels, which in mobilenet-v2 are previously increased by the expand block\n

    Args:
        layer (tf.keras.layers.Layer): input layer for this block
        channels (int): number of filters applied by the pointwise convolution
        counter_blocks (int): counter for the the blocks, used to give proper names to the layers

    Returns:
        tf.keras.layers.Layer: output layer from the mobilenet-v2 project block
    """
    # set the prefix for layers names
    name_prefix = f'backbone-block{counter_blocks}-project-'

    # apply in sequence pointwise convolution and batch normalization
    layer = tf.keras.layers.Conv2D(filters=channels, kernel_size=1, padding='same', use_bias=False, name=f'{name_prefix}conv')(layer)
    layer = tf.keras.layers.BatchNormalization(name=f'{name_prefix}batchnorm')(layer)

    return layer

def mobilenetv2_block_sequence(layer: tf.keras.layers.Layer, expansion_factor: int, channels_output: int, n_repeat: int, strides: Union[int, Tuple[int, int]], counter_blocks: int) -> tf.keras.layers.Layer:
    """
    mobilenet-v2 sequence block, which consists in a sequence of expand, depthwise and project blocks\n
    the add operation (residual connection) it's applied only when a sequence of expand-depwthise-project it's repeated more than 1 time, as described by the paper\n

    Args:
        layer (tf.keras.layers.Layer): input layer for this block
        expansion_factor (int): expansion factor used for the expand block (input channels are increased by a factor equal to expansion_factor)
        channels_output (int): number of channels in the output layer of this block
        n_repeat (int): number of times that the residual bottlenec layers (expand/depthwise/project) it's repeated
        strides (Union[int, Tuple[int, int]]): following the paper architecture should be 1 or 2, but only the first depthwise convolution in a sequence will use strides greater than 1
        counter_blocks (int): counter for the the blocks, used to give proper names to the layers

    Returns:
        tf.keras.layers.Layer: output layer from the mobilenet-v2 residual bottleneck layers sequence (expand-depwthise-project)
    """

    # last layer in this function code it's intended to be the last layer valid for a residual connection
    # note that the input layer given by argument for this sequence will never be valid for a residual connection, because it won't have a compatible shape for the add operation
    # here it's set equal to the last layer only for convenience in the for loop cycle, avoiding some if and conditions..
    # maybe i should have came up with better naming, but.. let's think about it in the future :)
    layer_last = layer

    # repeat the building blocks as requested
    for n in range(n_repeat):

        # increment the blocks counter, used for give proper names to the layers
        counter_blocks += 1

        # input channels
        channels_input = layer_last.shape[-1]

        # expanded channels are the input channels increased by a factor equal to expansion_factor 
        channels_expand = channels_input * expansion_factor

        # create an expand block
        layer = mobilenetv2_block_expand(layer=layer_last, channels=channels_expand, counter_blocks=counter_blocks)

        # create a depthwise block (as described by the paper, only the first depthwise convolution can apply strides > 1)
        layer = mobilenetv2_block_depthwise(layer=layer, strides=1 if n > 0 else strides, counter_blocks=counter_blocks)

        # create a project block
        layer = mobilenetv2_block_project(layer=layer, channels=channels_output, counter_blocks=counter_blocks)

        # it's possible to create a residual connection with the add operation starting from the second sequence of blocks
        # so if we are in the first sequence of blocks, the ouput from the project block become the last layer valid for a residual connection and it's also the input for the next sequence
        # instead from the second sequence of blocks, we create a residual connection with the add operation and the concatenated output it's also the input for the next sequence
        if n > 0:
            layer_last = tf.keras.layers.Add(name=f'backbone-block{counter_blocks}-add')([layer_last, layer])
        else:
            layer_last = layer

    # return the output layer and the increased counter blocks
    return layer_last, counter_blocks

def deeplabv3plus_encoder(layer: tf.keras.layers.Layer, filters: int = 256, dilation_rates: Tuple[int, int, int] = (6, 12, 18)) -> tf.keras.layers.Layer:
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

    Returns:
        tf.keras.layers.Layer: output from the deeplabv3plus encoder
    """
    
    # set the prefix for layers names
    name_prefix = f'segmentation-encoder-aspp-'

    # aspp block - pointwise convolution branch    
    layer_pointwise = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, padding='same', use_bias=False, name=f'{name_prefix}conv')(layer)
    layer_pointwise = tf.keras.layers.BatchNormalization(name=f'{name_prefix}conv-batchnorm')(layer_pointwise)
    layer_pointwise = tf.keras.layers.ReLU(name=f'{name_prefix}conv-relu')(layer_pointwise)

    # aspp block - 1st atrous separable convolution branch
    layer_atrous_1 = tf.keras.layers.DepthwiseConv2D(depth_multiplier=1, kernel_size=3, padding='same', dilation_rate=dilation_rates[0], use_bias=False, name=f'{name_prefix}atrous1-depthconv')(layer)
    layer_atrous_1 = tf.keras.layers.BatchNormalization(name=f'{name_prefix}atrous1-depthconv-batchnorm')(layer_atrous_1)
    layer_atrous_1 = tf.keras.layers.ReLU(name=f'{name_prefix}atrous1-depthconv-relu')(layer_atrous_1)
    layer_atrous_1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, padding='same', use_bias=False, name=f'{name_prefix}atrous1-conv')(layer_atrous_1)
    layer_atrous_1 = tf.keras.layers.BatchNormalization(name=f'{name_prefix}atrous1-conv-batchnorm')(layer_atrous_1)
    layer_atrous_1 = tf.keras.layers.ReLU(name=f'{name_prefix}atrous1-conv-relu')(layer_atrous_1)

    # aspp block - 2nd atrous separable convolution branch
    layer_atrous_2 = tf.keras.layers.DepthwiseConv2D(depth_multiplier=1, kernel_size=3, padding='same', dilation_rate=dilation_rates[1], use_bias=False, name=f'{name_prefix}atrous2-depthconv')(layer)
    layer_atrous_2 = tf.keras.layers.BatchNormalization(name=f'{name_prefix}atrous2-depthconv-batchnorm')(layer_atrous_2)
    layer_atrous_2 = tf.keras.layers.ReLU(name=f'{name_prefix}atrous2-depthconv-relu')(layer_atrous_2)
    layer_atrous_2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, padding='same', use_bias=False, name=f'{name_prefix}atrous2-conv')(layer_atrous_2)
    layer_atrous_2 = tf.keras.layers.BatchNormalization(name=f'{name_prefix}atrous2-conv-batchnorm')(layer_atrous_2)
    layer_atrous_2 = tf.keras.layers.ReLU(name=f'{name_prefix}atrous2-conv-relu')(layer_atrous_2)

    # aspp block - 3rd atrous separable convolution branch
    layer_atrous_3 = tf.keras.layers.DepthwiseConv2D(depth_multiplier=1, kernel_size=3, padding='same', dilation_rate=dilation_rates[2], use_bias=False, name=f'{name_prefix}atrous3-depthconv')(layer)
    layer_atrous_3 = tf.keras.layers.BatchNormalization(name=f'{name_prefix}atrous3-depthconv-batchnorm')(layer_atrous_3)
    layer_atrous_3 = tf.keras.layers.ReLU(name=f'{name_prefix}atrous3-depthconv-relu')(layer_atrous_3)
    layer_atrous_3 = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, padding='same', use_bias=False, name=f'{name_prefix}atrous3-conv')(layer_atrous_3)
    layer_atrous_3 = tf.keras.layers.BatchNormalization(name=f'{name_prefix}atrous3-conv-batchnorm')(layer_atrous_3)
    layer_atrous_3 = tf.keras.layers.ReLU(name=f'{name_prefix}atrous3-conv-relu')(layer_atrous_3)

    # set the prefix for layers names
    name_prefix = f'segmentation-encoder-'    

    # pooling block - apply global average pooling (height-width dimensions squeezed to 1x1 shape), pointwise convolution and then upsample back to the input layer height-width dimensions
    # this branch completely squeeze height-width dimensions with global average pooling down to 1x1 resolution
    # then recover the original height-width dimensions back simply using bilinear upsampling
    # this is what i understood reading the paper, but honestly i found it kind of strange, especially the bilinear upsampling
    # but.. i don't know, let's simply try it out
    # (note for me -> the upsampling rate it's fine to be equal to shape because the resolution after global average pooling is 1x1)
    upsampling_size_pooling = tuple(layer.shape[1:3])
    layer_pooling = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_last', keepdims=True, name=f'{name_prefix}pooling')(layer)
    layer_pooling = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, padding='same', use_bias=False, name=f'{name_prefix}pooling-conv')(layer_pooling)
    layer_pooling = tf.keras.layers.BatchNormalization(name=f'{name_prefix}pooling-batchnorm')(layer_pooling)
    layer_pooling = tf.keras.layers.ReLU(name=f'{name_prefix}pooling-relu')(layer_pooling)
    layer_pooling = tf.keras.layers.UpSampling2D(size=upsampling_size_pooling, interpolation='bilinear', name=f'{name_prefix}pooling-upsampling')(layer_pooling)

    # concatenate layer - concantenate the outputs from aspp block and pooling block along channel dimension
    layer_concat = tf.keras.layers.Concatenate(axis=-1, name=f'{name_prefix}concat')([layer_pointwise, layer_atrous_1, layer_atrous_2, layer_atrous_3, layer_pooling])

    # output layer - apply a pointwise convolution to the previous concatenated layer
    layer_output = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, padding='same', use_bias=False, name=f'{name_prefix}output-conv')(layer_concat)
    layer_output = tf.keras.layers.BatchNormalization(name=f'{name_prefix}output-batchnorm')(layer_output)
    layer_output = tf.keras.layers.ReLU(name=f'{name_prefix}output-relu')(layer_output)

    return layer_output

def deeplabv3plus_decoder(layer_encoder: tf.keras.layers.Layer, layer_backbone: tf.keras.layers.Layer, layer_output_height_width: Tuple[int], filters_backbone: int, filters_decoder: int, filters_output: int) -> tf.keras.layers.Layer:
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
        layer_output_height_width (Tuple[int]): height and width resolution for the output semantic segmentation mask
        filters_backbone (int): number of convolutional filters applied to the backbone layer (this should be less than number of channels from encoder's output layer), the paper suggest 32 or 48
        filters_decoder (int): number of convolutional filters applied in the decoder after the concatenation of encoder and backbone layers
        filters_output (int): number of convolutional filter applied to the output layer, this should be equal to the number of classes of the semantic segmentation problem

    Returns:
        tf.keras.layers.Layer: output layer for the semantic segmentation, the number of channels are equal to the number of classes    
    """
    
    # set the prefix for layers names
    name_prefix = 'segmentation-decoder-'

    # the encoder layer height-width dimensions are upsampled up to the backbone layer height-width dimensions
    upsampling_size_encoder = (int(layer_backbone.shape[1] / layer_encoder.shape[1]), int(layer_backbone.shape[2] / layer_encoder.shape[2]))
    layer_encoder = tf.keras.layers.UpSampling2D(size=upsampling_size_encoder, interpolation='bilinear', name=f'{name_prefix}pooling-upsampling')(layer_encoder)

    # reduce the number of channels from the backbone layer using pointwise convolution
    # the idea is to sharpen the mask coming from the encoder, but we don't want to give too much weight to the backbone layer, so we reduce the number of channels
    layer_backbone = tf.keras.layers.Conv2D(filters=filters_backbone, kernel_size=1, padding='same', use_bias=False, name=f'{name_prefix}backbone-conv')(layer_backbone)
    layer_backbone = tf.keras.layers.BatchNormalization(name=f'{name_prefix}backbone-batchnorm')(layer_backbone)
    layer_backbone = tf.keras.layers.ReLU(max_value=6.0, name=f'{name_prefix}backbone-relu6')(layer_backbone)

    # concatenate the upsampled encoder layer and the backbone layer along channel dimension
    layer_concat = tf.keras.layers.Concatenate(axis=-1, name=f'{name_prefix}concat')([layer_backbone, layer_encoder])

    # process the concatenated information with a sequence of two depthwise separable convolutions
    # probably the standard convolution will give better results, but it's more expensive in terms of resources
    # note for me -> if there are enough resources switch to standard convolution

    # first depthwise separable convolution
    layer = tf.keras.layers.DepthwiseConv2D(depth_multiplier=1, kernel_size=3, padding='same', use_bias=False, name=f'{name_prefix}depthconv1')(layer_concat)
    layer = tf.keras.layers.BatchNormalization(name=f'{name_prefix}depthconv1-batchnorm')(layer)
    layer = tf.keras.layers.ReLU(name=f'{name_prefix}depthconv1-relu')(layer)
    layer = tf.keras.layers.Conv2D(filters=filters_decoder, kernel_size=1, padding='same', use_bias=False, name=f'{name_prefix}conv1')(layer)
    layer = tf.keras.layers.BatchNormalization(name=f'{name_prefix}conv1-batchnorm')(layer)
    layer = tf.keras.layers.ReLU(name=f'{name_prefix}conv1-relu')(layer)

    # second depthwise separable convolution
    layer = tf.keras.layers.DepthwiseConv2D(depth_multiplier=1, kernel_size=3, padding='same', use_bias=False, name=f'{name_prefix}depthconv2')(layer_concat)
    layer = tf.keras.layers.BatchNormalization(name=f'{name_prefix}depthconv2-batchnorm')(layer)
    layer = tf.keras.layers.ReLU(name=f'{name_prefix}depthconv2-relu')(layer)
    layer = tf.keras.layers.Conv2D(filters=filters_decoder, kernel_size=1, padding='same', use_bias=False, name=f'{name_prefix}conv2')(layer)
    layer = tf.keras.layers.BatchNormalization(name=f'{name_prefix}conv2-batchnorm')(layer)
    layer = tf.keras.layers.ReLU(name=f'{name_prefix}conv2-relu')(layer)

    # create the ouput semantic segmentation mask, where the number of channels are equal to the number of classes
    upsampling_size_decoder = (int(layer_output_height_width[0] / layer.shape[1]), int(layer_output_height_width[1] / layer.shape[2]))
    layer_output = tf.keras.layers.Conv2D(filters=filters_output, kernel_size=3, padding='same', use_bias=False, name=f'{name_prefix}output-conv')(layer)
    layer_output = tf.keras.layers.UpSampling2D(size=upsampling_size_decoder, interpolation='bilinear', name=f'{name_prefix}output-upsampling')(layer_output)
    layer_output = tf.keras.layers.Softmax(name='output-mask')(layer_output)

    return layer_output

def ssdlite_block(layer: tf.keras.layers.Layer, filters: int, output_channels: int, name_prefix: str) -> tf.keras.layers.Layer:
    """
    single-shot-detector-lite (ssdlite) block, that consinsts in a depthwise separable convolution and a reshape operation\n
    the depthwise convolution process the input layer, followed by batch normalization and relu6\n
    then a pointwise convolution reduce the input channels applying a given number of filters\n
    the number of filters applied should be equal to the number of default bounding boxes, multiplied by number of classes (for classification) or by number of coordinates (4 for regression)\n
    finally the processed feature map it's reshaped in order to match the given output channels, which should be equal to number of classes (for classification) or number of coordinates (4 for regression)\n
    this reshape operation conceptually concatenate all the prediction points, leaving classes or boxes predictions on the channel dimension

    Args:
        layer (tf.keras.layers.Layer): input layer for this block
        filters (int): number of filter applied by the pointwise convolution, which should be equal to the number of default bounding boxes,
        multiplied by number of classes (for classification) or by number of coordinates (4 for regression)
        output_channels (int): should be equal to number of classes (for classification) or number of coordinates (4 for regression)
        name_prefix (str): prefix for layers names

    Returns:
        tf.keras.layers.Layer: output from a ssdlite block
    """
    layer = tf.keras.layers.DepthwiseConv2D(depth_multiplier=1, kernel_size=3, padding='same', use_bias=False, name=f'{name_prefix}-depthconv')(layer)
    layer = tf.keras.layers.BatchNormalization(name=f'{name_prefix}-batchnorm')(layer)
    layer = tf.keras.layers.ReLU(max_value=6.0, name=f'{name_prefix}-relu6')(layer)
    layer = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, padding='same', use_bias=True, name=f'{name_prefix}-pointconv')(layer)
    layer = tf.keras.layers.Reshape(target_shape=(-1, output_channels), name=f'{name_prefix}-reshape')(layer)

    return layer

def build_mobilenetv2_ssdseg(input_image_shape: Tuple[int, int, int], number_of_boxes_per_point: Union[int, List[int]], number_of_classes: int):

    if isinstance(number_of_boxes_per_point, int):
        number_of_boxes_per_point = (number_of_boxes_per_point,) * 4

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------
    # -> backbone input
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------
    # define the input
    layer_input_backbone = tf.keras.Input(shape=input_image_shape, dtype=tf.float32, name='input')

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------
    # -> backbone preprocessing
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------
    # rescale input values from [0, 255] to [-1, 1] (use scale=1./255 and offset=0.0 for rescaling to [0, 1])
    layer = tf.keras.layers.Rescaling(scale=1./127.5, offset=-1, name='input-rescaling')(layer_input_backbone)

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------
    # -> backbone architecture
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------
    # the initial mobilenet-v2 sequence of blocks it's slightly different from the rest, so let's build it using the basic blocks (expand, depthwise and project)
    # all the other sequences of blocks follow the same exact scheme, so the architecture can be built very easily

    # this is a counter used to give proper names to the network components
    counter_blocks = 0

    # the initial expand block apply 32 convolutional filters to the input, using kernel size 3 and strides 2
    layer = mobilenetv2_block_expand(layer=layer, channels=32, kernel_size=3, strides=2, counter_blocks=counter_blocks)

    # the initial depthwise block apply a depthwise convolution (one filter per channel) with strides 1
    layer = mobilenetv2_block_depthwise(layer=layer, strides=1, counter_blocks=counter_blocks)

    # the initial project block apply 16 pointwise convolutional filters
    layer = mobilenetv2_block_project(layer=layer, channels=16, counter_blocks=counter_blocks)

    # sequences of blocks (expand-depthwise-project)
    layer, counter_blocks = mobilenetv2_block_sequence(layer=layer, expansion_factor=6, channels_output=24, n_repeat=2, strides=2, counter_blocks=counter_blocks)
    layer, counter_blocks = mobilenetv2_block_sequence(layer=layer, expansion_factor=6, channels_output=32, n_repeat=3, strides=2, counter_blocks=counter_blocks)
    layer, counter_blocks = mobilenetv2_block_sequence(layer=layer, expansion_factor=6, channels_output=64, n_repeat=4, strides=2, counter_blocks=counter_blocks)
    layer, counter_blocks = mobilenetv2_block_sequence(layer=layer, expansion_factor=6, channels_output=96, n_repeat=3, strides=1, counter_blocks=counter_blocks)
    layer, counter_blocks = mobilenetv2_block_sequence(layer=layer, expansion_factor=6, channels_output=160, n_repeat=3, strides=2, counter_blocks=counter_blocks)
    layer, counter_blocks = mobilenetv2_block_sequence(layer=layer, expansion_factor=6, channels_output=320, n_repeat=1, strides=1, counter_blocks=counter_blocks)

    # additional deptwhise separable convolution blocks for further reduce the feature map size
    # these additional feature maps will be used for the object detection
    counter_blocks += 1
    name_prefix = f'backbone-block{counter_blocks}-'
    layer = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=2, padding='same', depth_multiplier=1, use_bias=False, name=f'{name_prefix}depthconv')(layer)        
    layer = tf.keras.layers.BatchNormalization(name=f'{name_prefix}depthconv-batchnorm')(layer)
    layer = tf.keras.layers.ReLU(max_value=6.0, name=f'{name_prefix}depthconv-relu6')(layer)
    layer = tf.keras.layers.Conv2D(filters=320, kernel_size=1, padding='same', use_bias=False, name=f'{name_prefix}pointconv')(layer)
    layer = tf.keras.layers.BatchNormalization(name=f'{name_prefix}pointconv-batchnorm')(layer)
    layer = tf.keras.layers.ReLU(max_value=6.0, name=f'{name_prefix}pointconv-relu6')(layer)

    counter_blocks += 1
    name_prefix = f'backbone-block{counter_blocks}-'
    layer = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=2, padding='same', depth_multiplier=1, use_bias=False, name=f'{name_prefix}depthconv')(layer)        
    layer = tf.keras.layers.BatchNormalization(name=f'{name_prefix}depthconv-batchnorm')(layer)
    layer = tf.keras.layers.ReLU(max_value=6.0, name=f'{name_prefix}depthconv-relu6')(layer)
    layer = tf.keras.layers.Conv2D(filters=320, kernel_size=1, padding='same', use_bias=False, name=f'{name_prefix}pointconv')(layer)
    layer = tf.keras.layers.BatchNormalization(name=f'{name_prefix}pointconv-batchnorm')(layer)
    layer_output_backbone = tf.keras.layers.ReLU(max_value=6.0, name=f'{name_prefix}pointconv-relu6')(layer)

    # this is not the final model, just a temporary one, but it's very useful to extract layers by name from the backbone
    model = tf.keras.Model(inputs=layer_input_backbone, outputs=layer_output_backbone)

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------
    # -> semantic segmentation head (deeplabv3+)
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------
    # the semantic segmentation encoder block starts from the latest backbone layer with output stride 16
    layer_input_segmentation_encoder_ = model.get_layer(name='backbone-block13-expand-relu6').output

    # the semantic segmentation decoder use an intermediate feature map for sharpen the output coming from the encoder
    layer_input_segmentation_decoder_from_backbone = model.get_layer(name='backbone-block3-expand-relu6').output

    # semantic segmentation encoder output
    layer_input_segmentation_decoder_from_encoder = deeplabv3plus_encoder(layer=layer_input_segmentation_encoder_, filters=256, dilation_rates=(6, 12, 18))

    # semantic segmentation decoder output (mask)
    layer_output_segmentation_mask = deeplabv3plus_decoder(
        layer_encoder=layer_input_segmentation_decoder_from_encoder,
        layer_backbone=layer_input_segmentation_decoder_from_backbone,
        layer_output_height_width=tuple(layer_input_backbone.shape[1:3]),
        filters_backbone=48,
        filters_decoder=256,
        filters_output=number_of_classes
    )

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------
    # -> object detection head (ssdlite)
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------
    # retrieve all the layers on which the ssd framework for object detection will be applied
    # these feature maps have different shapes, for better handling multi scale objects
    layer_input_object_detection_1 = model.get_layer(name='backbone-block13-expand-relu6').output
    layer_input_object_detection_2 = model.get_layer(name='backbone-block16-project-batchnorm').output
    layer_input_object_detection_3 = model.get_layer(name='backbone-block17-pointconv-relu6').output
    layer_input_object_detection_4 = model.get_layer(name='backbone-block18-pointconv-relu6').output

    # object detection regression branch (boxes)
    layer_output_object_detection_boxes_1 = ssdlite_block(layer=layer_input_object_detection_1, filters=number_of_boxes_per_point[0]*number_of_classes, output_channels=number_of_classes, name_prefix='object-detection-boxes1-')
    layer_output_object_detection_boxes_2 = ssdlite_block(layer=layer_input_object_detection_2, filters=number_of_boxes_per_point[1]*number_of_classes, output_channels=number_of_classes, name_prefix='object-detection-boxes2-')
    layer_output_object_detection_boxes_3 = ssdlite_block(layer=layer_input_object_detection_3, filters=number_of_boxes_per_point[2]*number_of_classes, output_channels=number_of_classes, name_prefix='object-detection-boxes3-')
    layer_output_object_detection_boxes_4 = ssdlite_block(layer=layer_input_object_detection_4, filters=number_of_boxes_per_point[3]*number_of_classes, output_channels=number_of_classes, name_prefix='object-detection-boxes4-')
    layer_output_object_detection_boxes = tf.keras.layers.Concatenate(axis=1, name=f'output-boxes')([layer_output_object_detection_boxes_1, layer_output_object_detection_boxes_2, layer_output_object_detection_boxes_3, layer_output_object_detection_boxes_4])


    # object detection classification branch (labels)
    layer_output_object_detection_labels_1 = ssdlite_block(layer=layer_input_object_detection_1, filters=number_of_boxes_per_point[0]*number_of_classes, output_channels=number_of_classes, name_prefix='object-detection-labels1-')
    layer_output_object_detection_labels_2 = ssdlite_block(layer=layer_input_object_detection_2, filters=number_of_boxes_per_point[1]*number_of_classes, output_channels=number_of_classes, name_prefix='object-detection-labels2-')
    layer_output_object_detection_labels_3 = ssdlite_block(layer=layer_input_object_detection_3, filters=number_of_boxes_per_point[2]*number_of_classes, output_channels=number_of_classes, name_prefix='object-detection-labels3-')
    layer_output_object_detection_labels_4 = ssdlite_block(layer=layer_input_object_detection_4, filters=number_of_boxes_per_point[3]*number_of_classes, output_channels=number_of_classes, name_prefix='object-detection-labels4-')
    layer_output_object_detection_labels_concat = tf.keras.layers.Concatenate(axis=1, name=f'object-detection-labels-concat')([layer_output_object_detection_labels_1, layer_output_object_detection_labels_2, layer_output_object_detection_labels_3, layer_output_object_detection_labels_4])
    layer_output_object_detection_labels = tf.keras.layers.Softmax(name='output-labels')(layer_output_object_detection_labels_concat)

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------
    # -> model with multiple outputs
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------
    model = tf.keras.Model(inputs=layer_input_backbone, outputs=[layer_output_segmentation_mask, layer_output_object_detection_labels, layer_output_object_detection_boxes])

    return model
