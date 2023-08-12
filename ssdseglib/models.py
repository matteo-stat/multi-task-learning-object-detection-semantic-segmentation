import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf

def mobilenetv2_block_expand(layer, channels, kernel_size=1, strides=1, counter_blocks=0):
    name_prefix = f'block{counter_blocks}_expand_'
    layer = tf.keras.layers.Conv2D(filters=channels, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False, name=f'{name_prefix}conv')(layer)
    layer = tf.keras.layers.BatchNormalization(name=f'{name_prefix}batchnorm')(layer)
    layer = tf.keras.layers.ReLU(max_value=6.0, name=f'{name_prefix}relu6')(layer)

    return layer

def mobilenetv2_block_depthwise(layer, strides, counter_blocks=0):
    name_prefix = f'block{counter_blocks}_depthwise_'
    layer = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=strides, padding='same', depth_multiplier=1, use_bias=False, name=f'{name_prefix}conv')(layer)        
    layer = tf.keras.layers.BatchNormalization(name=f'{name_prefix}batchnorm')(layer)
    layer = tf.keras.layers.ReLU(max_value=6.0, name=f'{name_prefix}relu6')(layer)

    return layer

def mobilenetv2_block_project(layer, channels, counter_blocks=0):
    name_prefix = f'block{counter_blocks}_project_'
    layer = tf.keras.layers.Conv2D(filters=channels, kernel_size=1, strides=1, padding='same', use_bias=False, name=f'{name_prefix}conv')(layer)
    layer = tf.keras.layers.BatchNormalization(name=f'{name_prefix}batchnorm')(layer)

    return layer

def mobilenetv2_block_sequence(layer_last, expansion_factor, channels_output, n_repeat, strides, counter_blocks):

    for n in range(n_repeat):
        # increment the blocks counter
        counter_blocks += 1

        # channels
        channels_input = layer_last.shape[-1]
        channels_expand = channels_input * expansion_factor

        # block expand
        layer = mobilenetv2_block_expand(layer=layer_last, channels=channels_expand, counter_blocks=counter_blocks)

        # block depthwise (only the first depthwise convolution apply strides > 1)
        layer = mobilenetv2_block_depthwise(layer=layer, strides=1 if n > 0 else strides, counter_blocks=counter_blocks)

        # block project
        layer = mobilenetv2_block_project(layer=layer, channels=channels_output, counter_blocks=counter_blocks)

        # overwrite layer input with the new output
        if n > 0:
            layer_last = tf.keras.layers.Add(name=f'block{counter_blocks}_add')([layer_last, layer])
        else:
            layer_last = layer

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
