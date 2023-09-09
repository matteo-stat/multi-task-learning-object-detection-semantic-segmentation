from typing import Tuple, Union, List, Literal
from numpy import ndarray
import tensorflow as tf
import ssdseglib

class MobileNetV2SsdSegBuilder():
    def __init__(
            self,
            input_image_shape: Tuple[int, int, int],
            number_of_boxes_per_point: Union[int, List[int]],
            number_of_classes: int,
            center_x_boxes_default: ndarray[float],
            center_y_boxes_default: ndarray[float],
            width_boxes_default: ndarray[float],
            height_boxes_default: ndarray[float],
            standard_deviations_centroids_offsets: tuple[float],
        ) -> None:
        """
        initialize the mobilenet-v2 builder\n
        this class can build a mobilenet-v2 backbone with a semantic segmentation head and an object detection head

        Args:
            input_image_shape (Tuple[int, int, int]): the input image shape as (height, width, channels)
            number_of_boxes_per_point (Union[int, List[int]]): number of default bounding boxes for each point in default grids
            number_of_classes (int): number of classes for the object detection head
            
            center_x_boxes_default (ndarray[float]): array of coordinates for center x (centroids coordinates)
            center_y_boxes_default (ndarray[float]): array of coordinates for center y (centroids coordinates)
            width_boxes_default (ndarray[float]): array of coordinates for width (centroids coordinates)
            height_boxes_default (ndarray[float]): array of coordinates for heigh (centroids coordinates)
            standard_deviations_centroids_offsets (tuple[float]): standard deviations for offsets between ground truth and default bounding boxes, expected as (standard_deviation_center_x_offsets, standard_deviation_center_y_offsets, standard_deviation_width_offsets, standard_deviation_height_offsets).
        """

        self.input_image_shape = input_image_shape
        self.number_of_boxes_per_point = (number_of_boxes_per_point,) * 4 if isinstance(number_of_boxes_per_point, int) else  number_of_boxes_per_point
        self.number_of_classes = number_of_classes
        self._center_x_boxes_default = center_x_boxes_default
        self._center_y_boxes_default = center_y_boxes_default
        self._width_boxes_default = width_boxes_default
        self._height_boxes_default = height_boxes_default
        self._standard_deviation_center_x_offsets, self._standard_deviation_center_y_offsets, self._standard_deviation_width_offsets, self._standard_deviation_height_offsets = standard_deviations_centroids_offsets
        
        # this is a counter used to give proper names to the network components
        self._counter_blocks = 0
        self._layers = {}

    def _mobilenetv2_block_expand(self, layer: tf.keras.layers.Layer, channels: int, kernel_size: Union[int, Tuple[int, int]] = 1, strides: Union[int, Tuple[int, int]] = 1) -> tf.keras.layers.Layer:
        """
        mobilenet-v2 expand block, that consists in a separable convolution followed by batch normalization and relu6 activation\n
        this block increase the input channels by an expansion factor (channels argument expect that you pass the output channels, so input channels * expansion factor)\n

        Args:
            layer (tf.keras.layers.Layer): input layer for this block
            channels (int): number of filters applied by the pointwise convolution, you should pass the output channels you want (input channels * expansion factor)
            kernel_size (Union[int, Tuple[int, int]], optional): for a standard mobilenet-v2 expand block should be 1. Defaults to 1.
            strides (Union[int, Tuple[int, int]], optional): for a standard mobilenet-v2 expand block should be 1. Defaults to 1.

        Returns:
            tf.keras.layers.Layer: output layer from the mobilenet-v2 expand block
        """
        # set the prefix for layers names
        name_prefix = f'backbone-block{self._counter_blocks}-expand-'

        # apply in sequence pointwise convolution, batch normalization and relu6
        layer = tf.keras.layers.Conv2D(filters=channels, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False, name=f'{name_prefix}conv')(layer)
        layer = tf.keras.layers.BatchNormalization(name=f'{name_prefix}batchnorm')(layer)
        layer = tf.keras.layers.ReLU(max_value=6.0, name=f'{name_prefix}relu6')(layer)

        return layer

    def _mobilenetv2_block_depthwise(self, layer: tf.keras.layers.Layer, strides: Union[int, Tuple[int, int]]) -> tf.keras.layers.Layer:
        """
        mobilenet-v2 depthwise block, that consists in a depthwise convolution followed by batch normalization and relu6 activation\n
        this block apply the depthwise convolution, that consists of indipendent convolutions for each input channel\n

        Args:
            layer (tf.keras.layers.Layer): input layer for this block
            strides (Union[int, Tuple[int, int]]): strides for the depthwise convolution, usually in mobilenet-v2 can be 1 or 2

        Returns:
            tf.keras.layers.Layer: output layer from the mobilenet-v2 depthwise block
        """
        # set the prefix for layers names
        name_prefix = f'backbone-block{self._counter_blocks}-depthwise-'

        # apply in sequence depthwise convolution, batch normalization and relu6
        # note that mobilenet-v2 use depthwise pointwise convolution, so we have always one filter per channel in the depthwise convolution (depth_multiplier=1)
        layer = tf.keras.layers.DepthwiseConv2D(depth_multiplier=1, kernel_size=3, strides=strides, padding='same', use_bias=False, name=f'{name_prefix}conv')(layer)        
        layer = tf.keras.layers.BatchNormalization(name=f'{name_prefix}batchnorm')(layer)
        layer = tf.keras.layers.ReLU(max_value=6.0, name=f'{name_prefix}relu6')(layer)

        return layer

    def _mobilenetv2_block_project(self, layer: tf.keras.layers.Layer, channels: int) -> tf.keras.layers.Layer:
        """
        mobilenet-v2 project block, that consist in a pointwise convolution followed by batch normalization\n
        this block reduce the number of channels, which in mobilenet-v2 are previously increased by the expand block\n

        Args:
            layer (tf.keras.layers.Layer): input layer for this block
            channels (int): number of filters applied by the pointwise convolution

        Returns:
            tf.keras.layers.Layer: output layer from the mobilenet-v2 project block
        """
        # set the prefix for layers names
        name_prefix = f'backbone-block{self._counter_blocks}-project-'

        # apply in sequence pointwise convolution and batch normalization
        layer = tf.keras.layers.Conv2D(filters=channels, kernel_size=1, padding='same', use_bias=False, name=f'{name_prefix}conv')(layer)
        layer = tf.keras.layers.BatchNormalization(name=f'{name_prefix}batchnorm')(layer)

        return layer

    def _mobilenetv2_block_sequence(self, layer: tf.keras.layers.Layer, expansion_factor: int, channels_output: int, n_repeat: int, strides: Union[int, Tuple[int, int]]) -> tf.keras.layers.Layer:
        """
        mobilenet-v2 sequence block, which consists in a sequence of expand, depthwise and project blocks\n
        the add operation (residual connection) it's applied only when a sequence of expand-depwthise-project it's repeated more than 1 time, as described by the paper\n

        Args:
            layer (tf.keras.layers.Layer): input layer for this block
            expansion_factor (int): expansion factor used for the expand block (input channels are increased by a factor equal to expansion_factor)
            channels_output (int): number of channels in the output layer of this block
            n_repeat (int): number of times that the residual bottlenec layers (expand/depthwise/project) it's repeated
            strides (Union[int, Tuple[int, int]]): following the paper architecture should be 1 or 2, but only the first depthwise convolution in a sequence will use strides greater than 1

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
            self._counter_blocks += 1

            # input channels
            channels_input = layer_last.shape[-1]

            # expanded channels are the input channels increased by a factor equal to expansion_factor 
            channels_expand = channels_input * expansion_factor

            # create an expand block
            layer = self._mobilenetv2_block_expand(layer=layer_last, channels=channels_expand)

            # create a depthwise block (as described by the paper, only the first depthwise convolution can apply strides > 1)
            layer = self._mobilenetv2_block_depthwise(layer=layer, strides=1 if n > 0 else strides)

            # create a project block
            layer = self._mobilenetv2_block_project(layer=layer, channels=channels_output)

            # it's possible to create a residual connection with the add operation starting from the second sequence of blocks
            # so if we are in the first sequence of blocks, the ouput from the project block become the last layer valid for a residual connection and it's also the input for the next sequence
            # instead from the second sequence of blocks, we create a residual connection with the add operation and the concatenated output it's also the input for the next sequence
            if n > 0:
                layer_last = tf.keras.layers.Add(name=f'backbone-block{self._counter_blocks}-add')([layer_last, layer])
            else:
                layer_last = layer

        # return the output layer and the increased counter blocks
        return layer_last

    def _mobilenetv2_backbone(self) -> tf.keras.layers.Layer:
        """
        create a mobilenet-v2 backbone\n
        some additional final layers are added for object detection, but they may not be required depending on your input image resolution

        Returns:
            tf.keras.layers.Layer: the input layer of the backbone
        """
        # ----------------------------------------------------------------------------------------------------------------------------------------------------------
        # -> backbone input
        # ----------------------------------------------------------------------------------------------------------------------------------------------------------
        # define the input
        layer_input = tf.keras.Input(shape=self.input_image_shape, dtype=tf.float32, name='backbone-input')

        # ----------------------------------------------------------------------------------------------------------------------------------------------------------
        # -> backbone preprocessing
        # ----------------------------------------------------------------------------------------------------------------------------------------------------------
        # rescale input values from [0, 255] to [-1, 1] (use scale=1./255 and offset=0.0 for rescaling to [0, 1])
        layer = tf.keras.layers.Rescaling(scale=1./127.5, offset=-1, name='backbone-input-rescaling')(layer_input)

        # ----------------------------------------------------------------------------------------------------------------------------------------------------------
        # -> backbone architecture
        # ----------------------------------------------------------------------------------------------------------------------------------------------------------
        # the initial mobilenet-v2 sequence of blocks it's slightly different from the rest, so let's build it using the basic blocks (expand, depthwise and project)
        # all the other sequences of blocks follow the same exact scheme, so the architecture can be built very easily

        # the initial expand block apply 32 convolutional filters to the input, using kernel size 3 and strides 2
        layer = self._mobilenetv2_block_expand(layer=layer, channels=32, kernel_size=3, strides=2)

        # the initial depthwise block apply a depthwise convolution (one filter per channel) with strides 1
        layer = self._mobilenetv2_block_depthwise(layer=layer, strides=1)

        # the initial project block apply 16 pointwise convolutional filters
        layer = self._mobilenetv2_block_project(layer=layer, channels=16)

        # sequences of blocks (expand-depthwise-project)
        layer = self._mobilenetv2_block_sequence(layer=layer, expansion_factor=6, channels_output=24, n_repeat=2, strides=2)
        layer = self._mobilenetv2_block_sequence(layer=layer, expansion_factor=6, channels_output=32, n_repeat=3, strides=2)
        layer = self._mobilenetv2_block_sequence(layer=layer, expansion_factor=6, channels_output=64, n_repeat=4, strides=2)
        layer = self._mobilenetv2_block_sequence(layer=layer, expansion_factor=6, channels_output=96, n_repeat=3, strides=1)
        layer = self._mobilenetv2_block_sequence(layer=layer, expansion_factor=6, channels_output=160, n_repeat=3, strides=2)
        layer_output = self._mobilenetv2_block_sequence(layer=layer, expansion_factor=6, channels_output=320, n_repeat=1, strides=1)

        # create a dictionary with all the backbone layers, using layers names as keys
        self._layers = {layer.name: layer.output for layer in tf.keras.Model(inputs=layer_input, outputs=layer_output).layers}

        return layer_input

    def _object_detection_head_ssdlite(self) -> Tuple[tf.keras.layers.Layer, tf.keras.layers.Layer]:
        """
        create an object detection head using ssd framework for object detection
        
        Returns:
            Tuple[tf.keras.layers.Layer, tf.keras.layers.Layer]: output layers for labels and boxes
        """
        # ----------------------------------------------------------------------------------------------------------------------------------------------------------
        # -> object detection inputs
        # ----------------------------------------------------------------------------------------------------------------------------------------------------------
        # retrieve all the layers on which the ssd framework for object detection will be applied
        # these feature maps have different shapes, for better handling multi scale objects
        layer_input_1 = self._layers['backbone-block13-expand-relu6']
        layer_input_2 = self._layers['backbone-block16-project-batchnorm']

        # add to mobilenetv2 backbone other depthwise separable convolutions to further reduce feature map size
        # these additional feature maps will be inputs for ssd
        self._counter_blocks += 1
        name_prefix = f'backbone-block{self._counter_blocks}-'
        layer = tf.keras.layers.SeparableConv2D(filters=320, strides=2, kernel_size=3, padding='same', depth_multiplier=1, use_bias=False, name=f'{name_prefix}sepconv')(layer_input_2)
        layer = tf.keras.layers.BatchNormalization(name=f'{name_prefix}batchnorm')(layer)
        layer_input_3 = tf.keras.layers.ReLU(max_value=6.0, name=f'{name_prefix}relu6')(layer)

        self._counter_blocks += 1
        name_prefix = f'backbone-block{self._counter_blocks}-'
        layer = tf.keras.layers.SeparableConv2D(filters=360, strides=2, kernel_size=3, padding='same', depth_multiplier=1, use_bias=False, name=f'{name_prefix}sepconv')(layer_input_3)
        layer = tf.keras.layers.BatchNormalization(name=f'{name_prefix}batchnorm')(layer)
        layer_input_4 = tf.keras.layers.ReLU(max_value=6.0, name=f'{name_prefix}relu6')(layer)

        # ----------------------------------------------------------------------------------------------------------------------------------------------------------
        # -> object detection classification
        # ----------------------------------------------------------------------------------------------------------------------------------------------------------
        # object detection classification branches at different feature maps scales
        layer_labels_1 = ssdseglib.blocks.ssdlite(layer=layer_input_1, filters=self.number_of_boxes_per_point[0]*self.number_of_classes, output_channels=self.number_of_classes, name_prefix='labels1-', relu_max_value=6.0)
        layer_labels_2 = ssdseglib.blocks.ssdlite(layer=layer_input_2, filters=self.number_of_boxes_per_point[1]*self.number_of_classes, output_channels=self.number_of_classes, name_prefix='labels2-', relu_max_value=6.0)
        layer_labels_3 = ssdseglib.blocks.ssdlite(layer=layer_input_3, filters=self.number_of_boxes_per_point[2]*self.number_of_classes, output_channels=self.number_of_classes, name_prefix='labels3-', relu_max_value=6.0)
        layer_labels_4 = ssdseglib.blocks.ssdlite(layer=layer_input_4, filters=self.number_of_boxes_per_point[3]*self.number_of_classes, output_channels=self.number_of_classes, name_prefix='labels4-', relu_max_value=6.0)

        # concatenate along boxes dimension
        layer_labels_concat = tf.keras.layers.Concatenate(axis=1, name=f'labels-concat')([layer_labels_1, layer_labels_2, layer_labels_3, layer_labels_4])

        # softmax for outputting classes probabilities
        layer_output_labels = tf.keras.layers.Softmax(name='output-labels')(layer_labels_concat)

        # ----------------------------------------------------------------------------------------------------------------------------------------------------------
        # -> object detection regression
        # ----------------------------------------------------------------------------------------------------------------------------------------------------------
        # object detection regression branches at different feature maps scales
        layer_boxes_1 = ssdseglib.blocks.ssdlite(layer=layer_input_1, filters=self.number_of_boxes_per_point[0]*self.number_of_classes, output_channels=self.number_of_classes, name_prefix='boxes1-', relu_max_value=6.0)
        layer_boxes_2 = ssdseglib.blocks.ssdlite(layer=layer_input_2, filters=self.number_of_boxes_per_point[1]*self.number_of_classes, output_channels=self.number_of_classes, name_prefix='boxes2-', relu_max_value=6.0)
        layer_boxes_3 = ssdseglib.blocks.ssdlite(layer=layer_input_3, filters=self.number_of_boxes_per_point[2]*self.number_of_classes, output_channels=self.number_of_classes, name_prefix='boxes3-', relu_max_value=6.0)
        layer_boxes_4 = ssdseglib.blocks.ssdlite(layer=layer_input_4, filters=self.number_of_boxes_per_point[3]*self.number_of_classes, output_channels=self.number_of_classes, name_prefix='boxes4-', relu_max_value=6.0)

        # concatenate along boxes dimension
        layer_output_boxes = tf.keras.layers.Concatenate(axis=1, name=f'output-boxes')([layer_boxes_1, layer_boxes_2, layer_boxes_3, layer_boxes_4])

        return layer_output_labels, layer_output_boxes

    def _semantic_segmentation_head_deeplabv3plus(self, dilation_rates: Tuple[int, int, int] = (6, 12, 18)) -> tf.keras.layers.Layer:
        """
        create a semantic segmentation head
        
        Args:
            dilation_rates (Tuple[int, int, int], optional): a tuple with three different dilation rates for the atrous convolutions. Defaults to (6, 12, 18).

        Returns:
            tf.keras.layers.Layer: output segmentation mask
        """
        
        # ----------------------------------------------------------------------------------------------------------------------------------------------------------
        # -> semantic segmentation encoder
        # ----------------------------------------------------------------------------------------------------------------------------------------------------------
        # input for encoder it's the backbone layer with output stride 16
        layer_input_encoder = self._layers['backbone-block13-expand-relu6']

        # encoder output it's one of the input for the decoder
        layer_input_decoder_from_encoder = ssdseglib.blocks.deeplabv3plus_encoder(layer=layer_input_encoder, filters=256, dilation_rates=dilation_rates, relu_max_value=6.0)

        # ----------------------------------------------------------------------------------------------------------------------------------------------------------
        # -> semantic segmentation decoder
        # ----------------------------------------------------------------------------------------------------------------------------------------------------------        
        # the semantic segmentation decoder use an intermediate feature map for sharpen the output coming from the encoder
        layer_input_decoder_from_backbone = self._layers['backbone-block3-expand-relu6']

        # semantic segmentation decoder output (mask)
        layer_output = ssdseglib.blocks.deeplabv3plus_decoder(
            layer_encoder=layer_input_decoder_from_encoder,
            layer_backbone=layer_input_decoder_from_backbone,
            filters_backbone=48,
            filters_decoder=256,
            output_height_width=self.input_image_shape[0:2],
            output_channels=self.number_of_classes,
            relu_max_value=6.0
        )

        return layer_output

    def get_model_for_training(self, segmentation_architecture: Literal['deeplabv3plus'], object_detection_architecture: Literal['ssdlite'], segmentation_dilation_rates: Tuple[int, int, int] = (6, 12, 18)) -> tf.keras.Model:
        """
        create a mobilenet-v2 backbone with a segmentation head and an object detection head\n
        this model performs simultaneously semantic segmentation, classification and regression tasks\n
        the ouputs of this architecture are valid for the training phase

        Args:
            segmentation_architecture (Literal[''deeplabv3plus']): specify a valid semantic segmentation architecture
            object_detection_architecture (Literal['ssdlite']): specify a valid object detection architecture
            segmentation_dilation_rates (Tuple[int, int, int], optional): a tuple with three different dilation rates for the atrous convolutions. Defaults to (6, 12, 18).

        Returns:
            tf.keras.Model: the keras model
        """                
        # create backbone, segmentation head and object detection head
        self._counter_blocks = 0
        layer_input = self._mobilenetv2_backbone()
        if segmentation_architecture == 'deeplabv3plus':
            layer_output_mask = self._semantic_segmentation_head_deeplabv3plus(dilation_rates=segmentation_dilation_rates)
        
        if object_detection_architecture == 'ssdlite':
            layer_output_labels, layer_output_boxes = self._object_detection_head_ssdlite()

        # create a model for training, with 3 outputs
        model = tf.keras.Model(inputs=layer_input, outputs=[layer_output_mask, layer_output_labels, layer_output_boxes])

        # update the internal list of available layers
        self._layers = {layer.name: layer.output for layer in model.layers}
        
        return model

    def get_model_for_inference(
            self,
            model_trained: tf.keras.Model,
            max_number_of_boxes_per_class: int,
            max_number_of_boxes_per_sample: int,
            boxes_iou_threshold: float,
            labels_probability_threshold: float,
            suppress_background_boxes: bool,
            use_segmentation_suppression: bool
        ) -> tf.keras.Model:
        """
        transform a trained model to an inference one\b
        it adds a layer for decoding boxes coordinates predictions and apply non-maximum-suppression to the object detection outputs, to keep only relevant boxes\n
        the learned weights are kept from the trained model

        Args:
            model_trained (tf.keras.Model): a trained model
            max_number_of_boxes_per_class (int): maximum number of boxes to keep per class
            max_number_of_boxes_per_sample (int): maximum number of boxes per sample
            boxes_iou_threshold (float): threshold for deciding whether boxes overlap too much with respect to iou
            labels_probability_threshold (float): threshold for deciding when to remove boxes based on class probabilities

        Returns:
            tf.keras.Model: a model for performing inference
        """
        
        # extract layers from trained model
        layer_input = model_trained.get_layer('backbone-input').output
        layer_output_mask = model_trained.get_layer('output-mask').output
        layer_boxes = model_trained.get_layer('output-boxes').output
        layer_labels = model_trained.get_layer('output-labels').output

        # layer for decoding predictions from the object detection regression branch
        decode_boxes_centroids_offsets = ssdseglib.layers.DecodeBoxesCentroidsOffsets(
            center_x_boxes_default=self._center_x_boxes_default,
            center_y_boxes_default=self._center_y_boxes_default,
            width_boxes_default=self._width_boxes_default,
            height_boxes_default=self._height_boxes_default,
            standard_deviation_center_x_offsets=self._standard_deviation_center_x_offsets,
            standard_deviation_center_y_offsets=self._standard_deviation_center_y_offsets,
            standard_deviation_width_offsets=self._standard_deviation_width_offsets,
            standard_deviation_height_offsets=self._standard_deviation_height_offsets,
            name='decode-output-boxes'
        )
        decode_boxes_centroids_offsets.trainable = False

        # layer for processing the object detection heads outputs with non maximum suppression
        non_maximum_suppression = ssdseglib.layers.NonMaximumSuppression(
            max_number_of_boxes_per_class=max_number_of_boxes_per_class,
            max_number_of_boxes_per_sample=max_number_of_boxes_per_sample,
            boxes_iou_threshold=boxes_iou_threshold,
            labels_probability_threshold=labels_probability_threshold,
            suppress_background_boxes=suppress_background_boxes,
            name='output-object-detection'
        )
        non_maximum_suppression.trainable = False

        if use_segmentation_suppression:
            # segmentation suppression
            segmentation_suppression = ssdseglib.layers.SegmentationSuppression(name='segmentation-suppression')
            segmentation_suppression.trainable = False

            # process probabilities by segmentation suppression
            layer_labels = segmentation_suppression(segmentation_mask=layer_output_mask, labels_probabilities=layer_labels)

        # convert predicted centroids offsets to boxes corners coordinates
        layer_boxes_decoded = decode_boxes_centroids_offsets(layer_boxes)

        # keep only relevant boxes
        layer_output_object_detection = non_maximum_suppression(boxes_corners_coordinates=layer_boxes_decoded, labels_probabilities=layer_labels)

        # layers for inference model
        model_inference = tf.keras.Model(inputs=layer_input, outputs=[layer_output_mask, layer_output_object_detection])

        # transfer weights from the trained model to the inference model        
        for layer_trained in model_trained.layers:
            model_inference.get_layer(layer_trained.name).set_weights(layer_trained.get_weights())            

        return model_inference

class ShuffleNetV2SsdSegBuilder():
    def __init__(
            self,
            input_image_shape: Tuple[int, int, int],
            model_size: Literal['0.5x', '1x', '1.5x', '2x'],
            use_additional_depthwise_convolution: bool,
            use_residual_connections: bool,
            number_of_boxes_per_point: Union[int, List[int]],
            number_of_classes: int,
            center_x_boxes_default: ndarray[float],
            center_y_boxes_default: ndarray[float],
            width_boxes_default: ndarray[float],
            height_boxes_default: ndarray[float],
            standard_deviations_centroids_offsets: tuple[float],
        ) -> None:
        """
        initialize the shufflenet-v2 builder\n
        this class can build a shufflenet-v2 backbone with a semantic segmentation head and an object detection head

        Args:
            input_image_shape (Tuple[int, int, int]): the input image shape as (height, width, channels)
            model_size (Literal['0.5x', '1x', '1.5x', '2x']): the shufflenet-v2 model size
            use_additional_depthwise_convolution (bool): if True add an additional depthwise convolution before the first pointwise convolution in each building block, the paper suggests that may improved object detection            
            use_residual_connections (bool): if True add a residual connection in shufflenet-v2 basic units
            number_of_boxes_per_point (Union[int, List[int]]): number of default bounding boxes for each point in default grids
            number_of_classes (int): number of classes for the object detection head            
            center_x_boxes_default (ndarray[float]): array of coordinates for center x (centroids coordinates)
            center_y_boxes_default (ndarray[float]): array of coordinates for center y (centroids coordinates)
            width_boxes_default (ndarray[float]): array of coordinates for width (centroids coordinates)
            height_boxes_default (ndarray[float]): array of coordinates for heigh (centroids coordinates)
            standard_deviations_centroids_offsets (tuple[float]): standard deviations for offsets between ground truth and default bounding boxes, expected as (standard_deviation_center_x_offsets, standard_deviation_center_y_offsets, standard_deviation_width_offsets, standard_deviation_height_offsets).
        """

        self.input_image_shape = input_image_shape
        if model_size == '0.5x':
            self.output_channels_stages = {2: 48, 3: 96, 4: 192}
        elif model_size == '1x':
            self.output_channels_stages = {2: 116, 3: 232, 4: 464}
        elif model_size == '1.5x':
            self.output_channels_stages = {2: 176, 3: 352, 4: 704}
        elif model_size == '2x':
            self.output_channels_stages = {2: 244, 3: 488, 4: 976}
        else:
            raise ValueError('invalid "model_size" value! available values are "0.5x", "1x", "1.5x", "2x"')
        self.use_additional_depthwise_convolution = use_additional_depthwise_convolution
        self.use_residual_connections = use_residual_connections
        self.number_of_boxes_per_point = (number_of_boxes_per_point,) * 4 if isinstance(number_of_boxes_per_point, int) else  number_of_boxes_per_point
        self.number_of_classes = number_of_classes
        self._center_x_boxes_default = center_x_boxes_default
        self._center_y_boxes_default = center_y_boxes_default
        self._width_boxes_default = width_boxes_default
        self._height_boxes_default = height_boxes_default
        self._standard_deviation_center_x_offsets, self._standard_deviation_center_y_offsets, self._standard_deviation_width_offsets, self._standard_deviation_height_offsets = standard_deviations_centroids_offsets        
        self._layers = {}    

    def _shufflenetv2_block_channels_shuffle(self, layer: tf.keras.layers.Layer, name_prefix: str, groups: int = 2) -> tf.keras.layers.Layer:
        """
        shufflenet-v2 channels shuffle block

        Args:
            layer (tf.keras.layers.Layer): input layer for this block
            name_prefix (str): name prefix to use for layers in the block
            groups (int, optional): number of groups to shuffle. Defaults to 2.

        Returns:
            tf.keras.layers.Layer: a layer with shuffled channels
        """

        # get height, width and channels
        _, height, width, channels = layer.get_shape().as_list()

        # reshape the input layer adding a new group dimension
        layer = tf.keras.layers.Reshape(target_shape=(height, width, groups, channels // groups), name=f'{name_prefix}reshape-pre-channels-shuffle')(layer)

        # permute group and channel dimensions, shuffling values along groups and channels 
        layer = tf.keras.layers.Permute(dims=(1, 2, 4, 3), name=f'{name_prefix}channels-shuffle')(layer)

        # reshape to original shape
        layer_output = tf.keras.layers.Reshape(target_shape=(height, width, channels), name=f'{name_prefix}reshape-post-channels-shuffle')(layer)

        return layer_output

    def _shufflenetv2_block_downsampling_unit(self, layer: tf.keras.layers.Layer, output_channels: int, name_prefix: str) -> tf.keras.layers.Layer:
        """
        shufflenet-v2 downsampling unit, used for spatial down sampling

        Args:
            layer (tf.keras.layers.Layer): input layer for this block
            output_channels (int): number of output channels
            name_prefix (str): name prefix to use for layers in the block

        Returns:
            tf.keras.layers.Layer: output layer from shufflenet-v2 downsampling unit
        """
        
        # number of filters to apply in each branch it's equal to half the output channels
        filters = output_channels // 2

        # left downsampling branch as described in the paper
        branch_left = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=2, padding='same', depth_multiplier=1, use_bias=False, name=f'{name_prefix}branch-left-depthconv1')(layer)
        branch_left = tf.keras.layers.BatchNormalization(name=f'{name_prefix}branch-left-batchnorm1')(branch_left)
        
        branch_left = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, padding='same', use_bias=False, name=f'{name_prefix}branch-left-conv2')(branch_left)        
        branch_left = tf.keras.layers.BatchNormalization(name=f'{name_prefix}branch-left-batchnorm2')(branch_left)
        branch_left = tf.keras.layers.ReLU(name=f'{name_prefix}branch-left-relu2')(branch_left)

        # right downsampling branch as described in the paper
        if self.use_additional_depthwise_convolution:
            branch_right = tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding='same', depth_multiplier=1, use_bias=False, name=f'{name_prefix}branch-right-depthconv0')(layer)
            branch_right = tf.keras.layers.BatchNormalization(name=f'{name_prefix}branch-right-batchnorm0')(branch_right)
            branch_right = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, padding='same', use_bias=False, name=f'{name_prefix}branch-right-conv1')(branch_right)
        else:
            branch_right = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, padding='same', use_bias=False, name=f'{name_prefix}branch-right-conv1')(layer)

        branch_right = tf.keras.layers.BatchNormalization(name=f'{name_prefix}branch-right-batchnorm1')(branch_right)
        branch_right = tf.keras.layers.ReLU(name=f'{name_prefix}branch-right-relu1')(branch_right)

        branch_right = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=2, padding='same', depth_multiplier=1, use_bias=False, name=f'{name_prefix}branch-right-depthconv2')(branch_right)
        branch_right = tf.keras.layers.BatchNormalization(name=f'{name_prefix}branch-right-batchnorm2')(branch_right)
        
        branch_right = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, padding='same', use_bias=False, name=f'{name_prefix}branch-right-conv3')(branch_right)
        branch_right = tf.keras.layers.BatchNormalization(name=f'{name_prefix}branch-right-batchnorm3')(branch_right)
        branch_right = tf.keras.layers.ReLU(name=f'{name_prefix}branch-right-relu3')(branch_right)

        # concat branches along channels dimension
        layer_concat = tf.keras.layers.Concatenate(axis=-1, name=f'{name_prefix}-concat')([branch_left, branch_right])

        # channels shuffle
        layer_output = self._shufflenetv2_block_channels_shuffle(layer_concat, name_prefix=f'{name_prefix}')

        return layer_output
    
    def _shufflenetv2_block_basic_unit(self, layer: tf.keras.layers.Layer, output_channels: int, name_prefix: str) -> tf.keras.layers.Layer:
        """
        shufflenet-v2 basic unit

        Args:
            layer (tf.keras.layers.Layer): input layer for this block
            output_channels (int): number of output channels
            name_prefix (str): name prefix to use for layers in the block
        Returns:
            tf.keras.layers.Layer: output layer from shufflenet-v2 basic unit
        """

        # number of filters to apply in each branch it's equal to half the output channels
        filters = output_channels // 2

        # split input channels evenly in two branches
        layer_identity, layer_conv = ssdseglib.layers.Split(num_or_size_splits=2, axis=-1, name=f'{name_prefix}channels-split')(layer)
        
        # branch with convolutions, as described in the paper
        if self.use_additional_depthwise_convolution:
            branch_conv = tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding='same', depth_multiplier=1, use_bias=False, name=f'{name_prefix}branch-conv-depthconv0')(layer_conv)
            branch_conv = tf.keras.layers.BatchNormalization(name=f'{name_prefix}branch-conv-batchnorm0')(branch_conv)
            branch_conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, padding='same', use_bias=False, name=f'{name_prefix}branch-conv-conv1')(branch_conv)
        else:            
            branch_conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, padding='same', use_bias=False, name=f'{name_prefix}branch-conv-conv1')(layer_conv)

        branch_conv = tf.keras.layers.BatchNormalization(name=f'{name_prefix}branch-conv-batchnorm1')(branch_conv)
        branch_conv = tf.keras.layers.ReLU(name=f'{name_prefix}branch-conv-relu1')(branch_conv)

        branch_conv = tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding='same', depth_multiplier=1, use_bias=False, name=f'{name_prefix}branch-conv-depthconv2')(branch_conv)
        branch_conv = tf.keras.layers.BatchNormalization(name=f'{name_prefix}branch-conv-batchnorm2')(branch_conv)
        
        branch_conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, padding='same', use_bias=False, name=f'{name_prefix}branch-conv-conv3')(branch_conv)
        branch_conv = tf.keras.layers.BatchNormalization(name=f'{name_prefix}branch-conv-batchnorm3')(branch_conv)

        if self.use_residual_connections:
            branch_conv = tf.keras.layers.Add(name=f'{name_prefix}branch-conv-add')([branch_conv, layer_conv])

        branch_conv = tf.keras.layers.ReLU(name=f'{name_prefix}branch-conv-relu3')(branch_conv)

        # concat branches
        layer_concat = tf.keras.layers.Concatenate(axis=-1, name=f'{name_prefix}concat')([layer_identity, branch_conv])

        # channels shuffle
        layer_output = self._shufflenetv2_block_channels_shuffle(layer_concat, name_prefix=f'{name_prefix}')

        return layer_output    

    def _shufflenetv2_backbone(self) -> tf.keras.layers.Layer:
        """
        create a shufflenet-v2 backbone

        Returns:
            tf.keras.layers.Layer: the input layer of the backbone
        """
        # ----------------------------------------------------------------------------------------------------------------------------------------------------------
        # -> backbone input
        # ----------------------------------------------------------------------------------------------------------------------------------------------------------
        # define the input
        layer_input = tf.keras.Input(shape=self.input_image_shape, dtype=tf.float32, name='backbone-input')

        # ----------------------------------------------------------------------------------------------------------------------------------------------------------
        # -> backbone preprocessing
        # ----------------------------------------------------------------------------------------------------------------------------------------------------------
        # rescale input values from [0, 255] to [-1, 1] (use scale=1./255 and offset=0.0 for rescaling to [0, 1])
        layer = tf.keras.layers.Rescaling(scale=1./127.5, offset=-1, name='backbone-input-rescaling')(layer_input)

        # ----------------------------------------------------------------------------------------------------------------------------------------------------------
        # -> backbone architecture
        # ----------------------------------------------------------------------------------------------------------------------------------------------------------
        # the initial shufflenet-v2 sequence of blocks consists in a pointwise convolution and max pooling, both use kernel size 3 and stride 2
        layer = tf.keras.layers.Conv2D(filters=24, kernel_size=3, strides=2, padding='same', use_bias=True, name='backbone-stage1-conv')(layer)
        layer = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='backbone-stage1-maxpool')(layer)

        # shufflenet stage 2
        layer = self._shufflenetv2_block_downsampling_unit(layer, output_channels=self.output_channels_stages[2], name_prefix='backbone-stage2-downblock-')
        name_prefix='backbone-stage2-block'
        for block_counter in range(3):
            layer = self._shufflenetv2_block_basic_unit(layer, output_channels=self.output_channels_stages[2], name_prefix=f'{name_prefix}{block_counter + 1}-')

        # shufflenet stage 3
        layer = self._shufflenetv2_block_downsampling_unit(layer, output_channels=self.output_channels_stages[3], name_prefix='backbone-stage3-downblock-')
        name_prefix='backbone-stage3-block'
        for block_counter in range(7):
            layer = self._shufflenetv2_block_basic_unit(layer, output_channels=self.output_channels_stages[3], name_prefix=f'{name_prefix}{block_counter + 1}-')

        # shufflenet stage 4
        layer = self._shufflenetv2_block_downsampling_unit(layer, output_channels=self.output_channels_stages[4], name_prefix='backbone-stage4-downblock-')
        name_prefix='backbone-stage4-block'
        for block_counter in range(3):
            layer = self._shufflenetv2_block_basic_unit(layer, output_channels=self.output_channels_stages[4], name_prefix=f'{name_prefix}{block_counter + 1}-')

        # create a dictionary with all the backbone layers, using layers names as keys
        self._layers = {layer.name: layer.output for layer in tf.keras.Model(inputs=layer_input, outputs=layer).layers}

        return layer_input
    
    def _object_detection_head_ssdlite(self) -> Tuple[tf.keras.layers.Layer, tf.keras.layers.Layer]:
        """
        create an object detection head using ssd framework for object detection
        
        Returns:
            Tuple[tf.keras.layers.Layer, tf.keras.layers.Layer]: output layers for labels and boxes
        """
        # ----------------------------------------------------------------------------------------------------------------------------------------------------------
        # -> object detection inputs
        # ----------------------------------------------------------------------------------------------------------------------------------------------------------
        # retrieve all the layers on which the ssd framework for object detection will be applied
        # these feature maps have different shapes, for better handling multi scale objects
        layer_input_1 = self._layers['backbone-stage3-block7-reshape-post-channels-shuffle']
        layer_input_2 = self._layers['backbone-stage4-block3-reshape-post-channels-shuffle']

        # add to shufflenet-v2 backbone other depthwise separable convolutions to further reduce feature map size
        # these additional feature maps will be inputs for ssd
        name_prefix = 'backbone-stage5-block1-'
        layer = tf.keras.layers.SeparableConv2D(filters=self.output_channels_stages[4], strides=2, kernel_size=3, padding='same', depth_multiplier=1, use_bias=False, name=f'{name_prefix}sepconv')(layer_input_2)
        layer = tf.keras.layers.BatchNormalization(name=f'{name_prefix}batchnorm')(layer)
        layer_input_3 = tf.keras.layers.ReLU(name=f'{name_prefix}relu')(layer)

        name_prefix = 'backbone-stage5-block2-'
        layer = tf.keras.layers.SeparableConv2D(filters=self.output_channels_stages[4], strides=2, kernel_size=3, padding='same', depth_multiplier=1, use_bias=False, name=f'{name_prefix}sepconv')(layer_input_3)
        layer = tf.keras.layers.BatchNormalization(name=f'{name_prefix}batchnorm')(layer)
        layer_input_4 = tf.keras.layers.ReLU(name=f'{name_prefix}relu')(layer)

        # ----------------------------------------------------------------------------------------------------------------------------------------------------------
        # -> object detection classification
        # ----------------------------------------------------------------------------------------------------------------------------------------------------------
        # object detection classification branches at different feature maps scales
        layer_labels_1 = ssdseglib.blocks.ssdlite(layer=layer_input_1, filters=self.number_of_boxes_per_point[0]*self.number_of_classes, output_channels=self.number_of_classes, name_prefix='labels1-')
        layer_labels_2 = ssdseglib.blocks.ssdlite(layer=layer_input_2, filters=self.number_of_boxes_per_point[1]*self.number_of_classes, output_channels=self.number_of_classes, name_prefix='labels2-')
        layer_labels_3 = ssdseglib.blocks.ssdlite(layer=layer_input_3, filters=self.number_of_boxes_per_point[2]*self.number_of_classes, output_channels=self.number_of_classes, name_prefix='labels3-')
        layer_labels_4 = ssdseglib.blocks.ssdlite(layer=layer_input_4, filters=self.number_of_boxes_per_point[3]*self.number_of_classes, output_channels=self.number_of_classes, name_prefix='labels4-')

        # concatenate along boxes dimension
        layer_labels_concat = tf.keras.layers.Concatenate(axis=1, name=f'labels-concat')([layer_labels_1, layer_labels_2, layer_labels_3, layer_labels_4])

        # softmax for outputting classes probabilities
        layer_output_labels = tf.keras.layers.Softmax(name='output-labels')(layer_labels_concat)

        # ----------------------------------------------------------------------------------------------------------------------------------------------------------
        # -> object detection regression
        # ----------------------------------------------------------------------------------------------------------------------------------------------------------
        # object detection regression branches at different feature maps scales
        layer_boxes_1 = ssdseglib.blocks.ssdlite(layer=layer_input_1, filters=self.number_of_boxes_per_point[0]*self.number_of_classes, output_channels=self.number_of_classes, name_prefix='boxes1-')
        layer_boxes_2 = ssdseglib.blocks.ssdlite(layer=layer_input_2, filters=self.number_of_boxes_per_point[1]*self.number_of_classes, output_channels=self.number_of_classes, name_prefix='boxes2-')
        layer_boxes_3 = ssdseglib.blocks.ssdlite(layer=layer_input_3, filters=self.number_of_boxes_per_point[2]*self.number_of_classes, output_channels=self.number_of_classes, name_prefix='boxes3-')
        layer_boxes_4 = ssdseglib.blocks.ssdlite(layer=layer_input_4, filters=self.number_of_boxes_per_point[3]*self.number_of_classes, output_channels=self.number_of_classes, name_prefix='boxes4-')

        # concatenate along boxes dimension
        layer_output_boxes = tf.keras.layers.Concatenate(axis=1, name=f'output-boxes')([layer_boxes_1, layer_boxes_2, layer_boxes_3, layer_boxes_4])

        return layer_output_labels, layer_output_boxes

    def _semantic_segmentation_head_deeplabv3plus(self, dilation_rates: Tuple[int, int, int] = (6, 12, 18)) -> tf.keras.layers.Layer:
        """
        create a semantic segmentation head
        
        Args:
            dilation_rates (Tuple[int, int, int], optional): a tuple with three different dilation rates for the atrous convolutions. Defaults to (6, 12, 18).

        Returns:
            tf.keras.layers.Layer: output segmentation mask
        """
        
        # ----------------------------------------------------------------------------------------------------------------------------------------------------------
        # -> semantic segmentation encoder
        # ----------------------------------------------------------------------------------------------------------------------------------------------------------
        # input for encoder it's the backbone layer with output stride 16
        layer_input_encoder = self._layers['backbone-stage3-block7-reshape-post-channels-shuffle']

        # encoder output it's one of the input for the decoder
        layer_input_decoder_from_encoder = ssdseglib.blocks.deeplabv3plus_encoder(layer=layer_input_encoder, filters=256, dilation_rates=dilation_rates)

        # ----------------------------------------------------------------------------------------------------------------------------------------------------------
        # -> semantic segmentation decoder
        # ----------------------------------------------------------------------------------------------------------------------------------------------------------        
        # the semantic segmentation decoder use an intermediate feature map for sharpen the output coming from the encoder
        # given the architecture of shufflenet-v2, i decided to concatenate two feature maps, with output stride 8 and 4
        
        # backbone layer with output stride 4
        layer_backbone_output_stride_4 = self._layers['backbone-stage1-maxpool']
        layer_backbone_output_stride_4 = tf.keras.layers.BatchNormalization(name='backbone-output-stride4-batchnorm')(layer_backbone_output_stride_4)

        # backbone layer with output stride 8, upsampled by deconvolution to output stride 4
        layer_backbone_output_stride_8 = self._layers['backbone-stage2-block3-reshape-post-channels-shuffle']
        layer_backbone_output_stride_8_upsampled = tf.keras.layers.Conv2DTranspose(filters=self.output_channels_stages[2], kernel_size=3, strides=2, padding='same', use_bias=False, name='backbone-output-stride8-upsampled-deconv')(layer_backbone_output_stride_8)
        layer_backbone_output_stride_8_upsampled = tf.keras.layers.BatchNormalization(name='backbone-output-stride8-upsampled-batchnorm')(layer_backbone_output_stride_8_upsampled)

        # concatenate
        layer_input_decoder_from_backbone = tf.keras.layers.Concatenate(name='backbone-intermediate-output-for-mask-decoder')([layer_backbone_output_stride_4, layer_backbone_output_stride_8_upsampled])

        # semantic segmentation decoder output (mask)
        layer_output = ssdseglib.blocks.deeplabv3plus_decoder(
            layer_encoder=layer_input_decoder_from_encoder,
            layer_backbone=layer_input_decoder_from_backbone,
            filters_backbone=48,
            filters_decoder=256,
            output_height_width=self.input_image_shape[0:2],
            output_channels=self.number_of_classes
        )

        return layer_output

    def get_model_for_training(self, segmentation_architecture: Literal['deeplabv3plus'], object_detection_architecture: Literal['ssdlite'], segmentation_dilation_rates: Tuple[int, int, int] = (6, 12, 18)) -> tf.keras.Model:
        """
        create a mobilenet-v2 backbone with a segmentation head and an object detection head\n
        this model performs simultaneously semantic segmentation, classification and regression tasks\n
        the ouputs of this architecture are valid for the training phase

        Args:
            segmentation_architecture (Literal[''deeplabv3plus']): specify a valid semantic segmentation architecture
            object_detection_architecture (Literal['ssdlite']): specify a valid object detection architecture
            segmentation_dilation_rates (Tuple[int, int, int], optional): a tuple with three different dilation rates for the atrous convolutions. Defaults to (6, 12, 18).

        Returns:
            tf.keras.Model: the keras model
        """                
        # create backbone, segmentation head and object detection head
        self._counter_blocks = 0
        layer_input = self._shufflenetv2_backbone()
        if segmentation_architecture == 'deeplabv3plus':
            layer_output_mask = self._semantic_segmentation_head_deeplabv3plus(dilation_rates=segmentation_dilation_rates)
        
        if object_detection_architecture == 'ssdlite':
            layer_output_labels, layer_output_boxes = self._object_detection_head_ssdlite()

        # create a model for training, with 3 outputs
        model = tf.keras.Model(inputs=layer_input, outputs=[layer_output_mask, layer_output_labels, layer_output_boxes])

        # update the internal list of available layers
        self._layers = {layer.name: layer.output for layer in model.layers}
        
        return model

    def get_model_for_inference(
            self,
            model_trained: tf.keras.Model,
            max_number_of_boxes_per_class: int,
            max_number_of_boxes_per_sample: int,
            boxes_iou_threshold: float,
            labels_probability_threshold: float,
            suppress_background_boxes: bool,
            use_segmentation_suppression: bool
        ) -> tf.keras.Model:
        """
        transform a trained model to an inference one\b
        it adds a layer for decoding boxes coordinates predictions and apply non-maximum-suppression to the object detection outputs, to keep only relevant boxes\n
        the learned weights are kept from the trained model

        Args:
            model_trained (tf.keras.Model): a trained model
            max_number_of_boxes_per_class (int): maximum number of boxes to keep per class
            max_number_of_boxes_per_sample (int): maximum number of boxes per sample
            boxes_iou_threshold (float): threshold for deciding whether boxes overlap too much with respect to iou
            labels_probability_threshold (float): threshold for deciding when to remove boxes based on class probabilities

        Returns:
            tf.keras.Model: a model for performing inference
        """
        
        # extract layers from trained model
        layer_input = model_trained.get_layer('backbone-input').output
        layer_output_mask = model_trained.get_layer('output-mask').output
        layer_boxes = model_trained.get_layer('output-boxes').output
        layer_labels = model_trained.get_layer('output-labels').output

        # layer for decoding predictions from the object detection regression branch
        decode_boxes_centroids_offsets = ssdseglib.layers.DecodeBoxesCentroidsOffsets(
            center_x_boxes_default=self._center_x_boxes_default,
            center_y_boxes_default=self._center_y_boxes_default,
            width_boxes_default=self._width_boxes_default,
            height_boxes_default=self._height_boxes_default,
            standard_deviation_center_x_offsets=self._standard_deviation_center_x_offsets,
            standard_deviation_center_y_offsets=self._standard_deviation_center_y_offsets,
            standard_deviation_width_offsets=self._standard_deviation_width_offsets,
            standard_deviation_height_offsets=self._standard_deviation_height_offsets,
            name='decode-output-boxes'
        )
        decode_boxes_centroids_offsets.trainable = False

        # layer for processing the object detection heads outputs with non maximum suppression
        non_maximum_suppression = ssdseglib.layers.NonMaximumSuppression(
            max_number_of_boxes_per_class=max_number_of_boxes_per_class,
            max_number_of_boxes_per_sample=max_number_of_boxes_per_sample,
            boxes_iou_threshold=boxes_iou_threshold,
            labels_probability_threshold=labels_probability_threshold,
            suppress_background_boxes=suppress_background_boxes,
            name='output-object-detection'
        )
        non_maximum_suppression.trainable = False

        if use_segmentation_suppression:
            # segmentation suppression
            segmentation_suppression = ssdseglib.layers.SegmentationSuppression(name='segmentation-suppression')
            segmentation_suppression.trainable = False

            # process probabilities by segmentation suppression
            layer_labels = segmentation_suppression(segmentation_mask=layer_output_mask, labels_probabilities=layer_labels)

        # convert predicted centroids offsets to boxes corners coordinates
        layer_boxes_decoded = decode_boxes_centroids_offsets(layer_boxes)

        # keep only relevant boxes
        layer_output_object_detection = non_maximum_suppression(boxes_corners_coordinates=layer_boxes_decoded, labels_probabilities=layer_labels)

        # layers for inference model
        model_inference = tf.keras.Model(inputs=layer_input, outputs=[layer_output_mask, layer_output_object_detection])

        # transfer weights from the trained model to the inference model        
        for layer_trained in model_trained.layers:
            model_inference.get_layer(layer_trained.name).set_weights(layer_trained.get_weights())            

        return model_inference
