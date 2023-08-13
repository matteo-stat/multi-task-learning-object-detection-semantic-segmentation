import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf

def mobilenetv2_block_expand(layer: tf.keras.layers.Layer, channels: int, counter_blocks: int, kernel_size: int | tuple[int] = 1, strides: int | tuple[int] = 1) -> tf.keras.layers.Layer:
    """
    mobilenet-v2 expand block, that consists in a separable convolution followed by batch normalization and relu6 activation
    this block increase the input channels by an expansion factor (channels argument expect that you pass the output channels, so input channels * expansion factor)

    Args:
        layer (tf.keras.layers.Layer): input layer for this block
        channels (int): number of filters applied by the pointwise convolution, you should pass the output channels you want (input channels * expansion factor)
        counter_blocks (int): counter for the the blocks, used to give proper names to the layers
        kernel_size (int | tuple[int], optional): for a standard mobilenet-v2 expand block should be 1. Defaults to 1.
        strides (int | tuple[int], optional): for a standard mobilenet-v2 expand block should be 1. Defaults to 1.

    Returns:
        tf.keras.layers.Layer: output layer from the mobilenet-v2 expand block
    """
    # set the prefix for layers names
    name_prefix = f'block{counter_blocks}_expand_'

    # apply in sequence pointwise convolution, batch normalization and relu6
    layer = tf.keras.layers.Conv2D(filters=channels, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False, name=f'{name_prefix}conv')(layer)
    layer = tf.keras.layers.BatchNormalization(name=f'{name_prefix}batchnorm')(layer)
    layer = tf.keras.layers.ReLU(max_value=6.0, name=f'{name_prefix}relu6')(layer)

    return layer

def mobilenetv2_block_depthwise(layer: tf.keras.layers.Layer, strides: int | tuple[int], counter_blocks: int) -> tf.keras.layers.Layer:
    """
    mobilenet-v2 depthwise block, that consists in a depthwise convolution followed by batch normalization and relu6 activation
    this block apply the depthwise convolution, that consists of indipendent convolutions for each input channel    

    Args:
        layer (tf.keras.layers.Layer): input layer for this block
        strides (int | tuple[int]): strides for the depthwise convolution, usually in mobilenet-v2 can be 1 or 2
        counter_blocks (int): counter for the the blocks, used to give proper names to the layers

    Returns:
        tf.keras.layers.Layer: output layer from the mobilenet-v2 depthwise block
    """
    # set the prefix for layers names
    name_prefix = f'block{counter_blocks}_depthwise_'

    # apply in sequence depthwise convolution, batch normalization and relu6
    # note that mobilenet-v2 use depthwise pointwise convolution, so we have always one filter per channel in the depthwise convolution (depth_multiplier=1)
    layer = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=strides, padding='same', depth_multiplier=1, use_bias=False, name=f'{name_prefix}conv')(layer)        
    layer = tf.keras.layers.BatchNormalization(name=f'{name_prefix}batchnorm')(layer)
    layer = tf.keras.layers.ReLU(max_value=6.0, name=f'{name_prefix}relu6')(layer)

    return layer

def mobilenetv2_block_project(layer: tf.keras.layers.Layer, channels: int, counter_blocks: int) -> tf.keras.layers.Layer:
    """
    mobilenet-v2 project block, that consist in a pointwise convolution followed by batch normalization
    this block reduce the number of channels, which in mobilenet-v2 are previously increased by the expand block

    Args:
        layer (tf.keras.layers.Layer): input layer for this block
        channels (int): number of filters applied by the pointwise convolution
        counter_blocks (int): counter for the the blocks, used to give proper names to the layers

    Returns:
        tf.keras.layers.Layer: output layer from the mobilenet-v2 project block
    """
    # set the prefix for layers names
    name_prefix = f'block{counter_blocks}_project_'

    # apply in sequence pointwise convolution and batch normalization
    layer = tf.keras.layers.Conv2D(filters=channels, kernel_size=1, strides=1, padding='same', use_bias=False, name=f'{name_prefix}conv')(layer)
    layer = tf.keras.layers.BatchNormalization(name=f'{name_prefix}batchnorm')(layer)

    return layer

def mobilenetv2_block_sequence(layer: tf.keras.layers.Layer, expansion_factor: int, channels_output: int, n_repeat: int, strides: int | tuple[int], counter_blocks: int) -> tf.keras.layers.Layer:
    """
    mobilenet-v2 sequence block, which consists in a sequence of expand, depthwise and project blocks
    in the original paper this kind of sequences are called residual bottleneck layers
    the add operation (residual connection) it's applied only when a sequence of expand-depwthise-project it's repeated more than 1 time, as described by the paper

    Args:
        layer (tf.keras.layers.Layer): input layer for this block
        expansion_factor (int): expansion factor used for the expand block (input channels are increased by a factor equal to expansion_factor)
        channels_output (int): number of channels in the output layer of this block
        n_repeat (int): number of times that the residual bottlenec layers (expand/depthwise/project) it's repeated
        strides (int | tuple[int]): following the paper architecture should be 1 or 2, but only the first depthwise convolution in a sequence will use strides greater than 1
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
            layer_last = tf.keras.layers.Add(name=f'block{counter_blocks}_add')([layer_last, layer])
        else:
            layer_last = layer

    # return the output layer and the increased counter blocks
    return layer_last, counter_blocks

layer_last = tf.keras.Input(shape=(480, 640, 3), dtype=tf.float32)

# the initial mobilenetv2 block sequence it's slightly different
# -> expansion block use a convolution with kernel size 3 and strides 2
# -> depthwise block use a convolution with strides 1
# -> projection block it's no different from the other ones
counter_blocks = 0
layer = mobilenetv2_block_expand(layer=layer_last, channels=32, kernel_size=3, strides=2, counter_blocks=counter_blocks)
layer = mobilenetv2_block_depthwise(layer=layer, strides=1, counter_blocks=counter_blocks)
layer = mobilenetv2_block_project(layer=layer, channels=16, counter_blocks=counter_blocks)

# create a mobilenetv2 blocks sequence as described in the paper
layer, counter_blocks = mobilenetv2_block_sequence(layer_last=layer, expansion_factor=6, channels_output=24, n_repeat=2, strides=2, counter_blocks=counter_blocks)
layer, counter_blocks = mobilenetv2_block_sequence(layer_last=layer, expansion_factor=6, channels_output=32, n_repeat=3, strides=2, counter_blocks=counter_blocks)
layer, counter_blocks = mobilenetv2_block_sequence(layer_last=layer, expansion_factor=6, channels_output=64, n_repeat=4, strides=2, counter_blocks=counter_blocks)
layer, counter_blocks = mobilenetv2_block_sequence(layer_last=layer, expansion_factor=6, channels_output=96, n_repeat=3, strides=1, counter_blocks=counter_blocks)
layer, counter_blocks = mobilenetv2_block_sequence(layer_last=layer, expansion_factor=6, channels_output=160, n_repeat=3, strides=2, counter_blocks=counter_blocks)
layer, counter_blocks = mobilenetv2_block_sequence(layer_last=layer, expansion_factor=6, channels_output=320, n_repeat=1, strides=1, counter_blocks=counter_blocks)


# ssdlite (object detection)
# simply replace standard convolution operations with depthwise separable convolution
# mobilenetv2 paper suggest to connect first ssd layer to the expansion block with output stride = 16 (30x40, block13_expand_relu6)
# the rest of ssd layer are connected on top of the last layer with output stride = 32 (15x20)

# deeplabv3 (semantic segmentation)
# mobilenetv2 paper says that deeplabv3 builds 5 parallel heads
# aspp, atrous spatial pyramid pooling, containing 3 convolutions with 3x3 kernel and different atrous rates
# 1 convolution with 1x1
# 1 image level features

# model
model = tf.keras.Model(inputs=layer_last, outputs=layer)
model.summary()

# deeplabv3 snippet
layer_input = layer
dims = layer_input.shape

# image features (global average pooling + upsampling)
x = tf.keras.layers.GlobalAveragePooling2D()(layer_input)
x = tf.keras.layers.Conv2D(filters=256, kernel_size=1, use_bias=False, padding='same')(x) # + batchnorm + relu
out_pool = tf.keras.layers.UpSampling2D(size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear")(x)

# aspp
out_aspp1 = tf.keras.layers.Conv2D(filters=256, kernel_size=1, padding='same', use_bias=False, dilation_rate=1)(layer_input) # + batchnorm + relu
out_aspp2 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', use_bias=False, dilation_rate=6)(layer_input) # + batchnorm + relu
out_aspp3 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', use_bias=False, dilation_rate=12)(layer_input) # + batchnorm + relu
out_aspp4 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', use_bias=False, dilation_rate=18)(layer_input) # + batchnorm + relu

# concatenate
concat = tf.keras.layers.Concatenate(axis=-1)([out_pool, out_aspp1, out_aspp2, out_aspp3, out_aspp4])

# output
out_deeplab = tf.keras.layers.Conv2D(filters=256, kernel_size=1, padding='same', use_bias=False)(concat) # + batchnorm + relu

# upsample to desired resolution..
