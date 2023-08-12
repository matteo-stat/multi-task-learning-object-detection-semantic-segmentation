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

model = tf.keras.Model(inputs=layer_last, outputs=layer)
model.summary()
