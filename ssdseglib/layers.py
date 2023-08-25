import tensorflow as tf

@tf.keras.saving.register_keras_serializable(name='DecodeBoxesCentroidsOffsets')
class DecodeBoxesCentroidsOffsets(tf.keras.layers.Layer):
    def __init__(
            self,
            center_x_boxes_default: tf.Tensor,
            center_y_boxes_default: tf.Tensor,
            width_boxes_default: tf.Tensor,
            height_boxes_default: tf.Tensor,
            standard_deviation_center_x_offsets: float,
            standard_deviation_center_y_offsets: float,
            standard_deviation_width_offsets: float,
            standard_deviation_height_offsets: float,
            **kwargs
        ) -> None:
        """
        decode the centroids offsets predicted by the network

        Args:
            center_x_boxes_default (tf.Tensor): center_x coordinates for default bounding boxes, expected shape it's (total boxes,)
            center_y_boxes_default (tf.Tensor): center_y coordinates for default bounding boxes, expected shape it's (total boxes,)
            width_boxes_default (tf.Tensor): width coordinates for default bounding boxes, expected shape it's (total boxes,)
            height_boxes_default (tf.Tensor): height coordinates for default bounding boxes, expected shape it's (total boxes,)
            standard_deviation_center_x_offsets (float): standard deviation for center_x offsets
            standard_deviation_center_y_offsets (float): standard deviation for center_y offsets
            standard_deviation_width_offsets (float): standard deviation for width offsets
            standard_deviation_height_offsets (float): standard deviation for height offsets
        """

        # init from parent class
        super().__init__(**kwargs)

        # set attributes
        self.center_x_boxes_default = tf.constant(center_x_boxes_default, dtype=tf.float32)
        self.center_y_boxes_default = tf.constant(center_y_boxes_default, dtype=tf.float32)
        self.width_boxes_default = tf.constant(width_boxes_default, dtype=tf.float32)
        self.height_boxes_default = tf.constant(height_boxes_default, dtype=tf.float32)
        self.standard_deviation_center_x_offsets = tf.constant(standard_deviation_center_x_offsets, dtype=tf.float32)
        self.standard_deviation_center_y_offsets = tf.constant(standard_deviation_center_y_offsets, dtype=tf.float32)
        self.standard_deviation_width_offsets = tf.constant(standard_deviation_width_offsets, dtype=tf.float32)
        self.standard_deviation_height_offsets = tf.constant(standard_deviation_height_offsets, dtype=tf.float32)

    def call(self, boxes_centroids_offsets: tf.Tensor) -> tf.Tensor:
        """
        decode the centroids offsets predicted by the network, returning corners coordinates (ymin, xmin, ymax, xmax)\n
        note that corners coordinates are returned in the order required by the non maximum suppression functionality available tensorflow

        Args:
            boxes_centroids_offsets (tf.Tensor): centroids offsets predicted by the network (center_x_offsets, center_y_offsets, width_offsets, height_offsets)

        Returns:
            tf.Tensor: decoded corners coordinates (ymin, xmin, ymax, xmax)
        """

        # split the offsets centroids coordinates
        center_x_offsets, center_y_offsets, width_offsets, height_offsets = tf.split(value=boxes_centroids_offsets, num_or_size_splits=4, axis=-1)
        
        # remove last dimension
        center_x_offsets = tf.squeeze(center_x_offsets, axis=-1)
        center_y_offsets = tf.squeeze(center_y_offsets, axis=-1)
        width_offsets = tf.squeeze(width_offsets, axis=-1)
        height_offsets = tf.squeeze(height_offsets, axis=-1)

        # decode centroids offsets to centroids coordinates
        center_x = center_x_offsets * self.standard_deviation_center_x_offsets * self.width_boxes_default + self.center_x_boxes_default
        center_y = center_y_offsets * self.standard_deviation_center_y_offsets * self.height_boxes_default + self.center_y_boxes_default
        width = (tf.math.exp(width_offsets * self.standard_deviation_width_offsets) - 1.0) * self.width_boxes_default
        height = (tf.math.exp(height_offsets * self.standard_deviation_height_offsets) - 1.0) * self.height_boxes_default

        # convert centroids to corners coordinates
        xmin = center_x - (width - 1.0) / 2.0
        ymin = center_y - (height - 1.0) / 2.0
        xmax = center_x + (width - 1.0) / 2.0
        ymax = center_y + (height - 1.0) / 2.0

        # concatenate corners coordinates as required by non maximum suppression built-in functionality in tensorflow
        corners = tf.stack([ymin, xmin, ymax, xmax], axis=2)
        
        return corners

    def get_config(self):
        return {
            'center_x_boxes_default': self.center_x_boxes_default,
            'center_y_boxes_default': self.center_y_boxes_default,
            'width_boxes_default': self.width_boxes_default,
            'height_boxes_default': self.height_boxes_default,
            'standard_deviation_center_x_offsets': self.standard_deviation_center_x_offsets,
            'standard_deviation_center_y_offsets': self.standard_deviation_center_y_offsets,
            'standard_deviation_width_offsets': self.standard_deviation_width_offsets,
            'standard_deviation_height_offsets': self.standard_deviation_height_offsets,
        }

@tf.keras.saving.register_keras_serializable(name='NonMaximumSuppression')
class NonMaximumSuppression(tf.keras.layers.Layer):
    def __init__(
            self,
            max_number_of_boxes_per_class: int,
            max_number_of_boxes_per_sample: int,
            boxes_iou_threshold: float,
            labels_probability_threshold: float,
            **kwargs
        ):
        """
        process the object detection outputs with non maximum suppression\n
        only a subset of boxes that meets the given criteria will be returned

        Args:
            max_number_of_boxes_per_class (int): maximum number of boxes to keep per class
            max_number_of_boxes_per_sample (int): maximum number of boxes per sample
            boxes_iou_threshold (float): threshold for deciding whether boxes overlap too much with respect to iou
            labels_probability_threshold (float): threshold for deciding when to remove boxes based on class probabilities
        """

        # init from parent class
        super().__init__(**kwargs)

        # set attributes
        self.max_number_of_boxes_per_class = max_number_of_boxes_per_class
        self.max_number_of_boxes_per_sample = max_number_of_boxes_per_sample
        self.boxes_iou_threshold = boxes_iou_threshold
        self.labels_probability_threshold = labels_probability_threshold

    def call(self, boxes_corners_coordinates: tf.Tensor, labels_probabilities: tf.Tensor) -> tf.Tensor:
        """
        process the object detection outputs with non maximum suppression\n
        only a subset of boxes that meets the given criteria will be returned        

        Args:
            boxes_corners_coordinates (tf.Tensor): predicted boxes corners coordinates with the following order (ymin, xmin, ymax, xmax), expected shape it's (batch, total boxes, 4)
            labels_probabilities (tf.Tensor): predicted labels probabilities, expected shape it's (batch, total boxes)

        Returns:
            tf.Tensor: _description_
        """

        # non maximum suppression
        boxes_corners_coordinates, labels_probabilities, labels, _ = tf.image.combined_non_max_suppression(
            boxes=tf.expand_dims(boxes_corners_coordinates, axis=2),
            scores=labels_probabilities,
            max_output_size_per_class=self.max_number_of_boxes_per_class,
            max_total_size=self.max_number_of_boxes_per_sample,
            iou_threshold=self.boxes_iou_threshold,
            score_threshold=self.labels_probability_threshold,
            clip_boxes=False
        )
        
        # we will keep only boxes related to classes that are not background
        not_background = tf.math.greater(labels, 0.)

        # reorder corners coordinates with the more standard style (xmin, ymin, xmax, ymax)
        boxes_corners_coordinates = tf.gather(boxes_corners_coordinates, indices=[1, 0, 3, 2], axis=-1)

        # expand dimension, needed for concatenate with 
        labels = tf.expand_dims(labels, axis=2)
        labels_probabilities = tf.expand_dims(labels_probabilities, axis=2)

        # concatenated classes labels, probabilities and boxes corners coordinates, output shape it's (batch, selected boxes, 6)
        object_detection_output = tf.concat([labels, labels_probabilities, boxes_corners_coordinates], axis=-1)

        # keep only output related to classes that are not background
        object_detection_output = tf.boolean_mask(tensor=object_detection_output, mask=not_background)

        return object_detection_output

    def get_config(self):
        return {
            'max_number_of_boxes_per_class': self.max_number_of_boxes_per_class,
            'max_number_of_boxes_per_sample': self.max_number_of_boxes_per_sample,
            'boxes_iou_threshold': self.boxes_iou_threshold,
            'labels_probability_threshold': self.labels_probability_threshold,
        }