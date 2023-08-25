from typing import Tuple, List, Callable
import tensorflow as tf

@tf.keras.saving.register_keras_serializable(name="jaccard_iou_segmentation_masks")
def jaccard_iou_segmentation_masks(classes_weights: List[float]) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """
    jaccard iou metric, for one-hot encoded semantic segmentation masks with shape (batch, height, width, number of classes)\n
    you must pass some weights for you classes, and they must sum up to 1 (otherwise the calculation of the loss won't be right)\n
    predictions must be passed as probabilities

    Args:
        classes_weights (List[float]): weights for your classes, they must sum up to 1 (otherwise the calculation of the loss won't be right)

    Returns:
        Callable[[tf.Tensor, tf.Tensor], tf.Tensor]: the function for calculating the weighted jaccard iou segmentation masks metric
    """

    classes_weights = tf.constant(classes_weights, dtype=tf.float32, shape=(1, len(classes_weights)))

    def jaccard_iou_segmentation_masks_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        jaccard iou metric, for one-hot encoded semantic segmentation masks with shape (batch, height, width, number of classes)\n
        it return a single scalar metric value per batch item, which is a weighted average of the classes metrics

        Args:
            y_true (tf.Tensor): ground truth
            y_pred (tf.Tensor): predictions, expressed as probabilities

        Returns:
            tf.Tensor: a tensor with a single scalar metric value per batch item, output shape it's (batch,)
        """
        
        # intersection area, along height and width dimensions, output shape it's (batch, number of classes)
        intersection = tf.math.reduce_sum(y_true * y_pred, axis=(1, 2))

        # total area, along height and width dimensions, output shape it's (batch, number of classes)
        total = tf.math.reduce_sum(y_true + y_pred, axis=(1, 2))

        # jaccard iou metric, with laplace smoothing for managing missing class values
        # union can be calculated easily as difference between total and intersection
        metric_value = (intersection + 1.0) / (total - intersection + 1.0)

        # weighted average along classes dimension, output shape it's (batch,)
        metric_value = metric_value * classes_weights
        metric_value = tf.reduce_sum(metric_value, axis=-1)

        return metric_value
    
    return jaccard_iou_segmentation_masks_metric

@tf.keras.saving.register_keras_serializable(name="jaccard_iou_bounding_boxes")
def jaccard_iou_bounding_boxes(
        center_x_boxes_default: tf.Tensor,
        center_y_boxes_default: tf.Tensor,
        width_boxes_default: tf.Tensor,
        height_boxes_default: tf.Tensor,
        standard_deviation_center_x_offsets: float,
        standard_deviation_center_y_offsets: float,
        standard_deviation_width_offsets: float,
        standard_deviation_height_offsets: float
    ) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """
    jaccard iou metric, for object detection regression data with shape (batch, total boxes, 4)\n
    predictions are expected to be standardized offsets centroids coordinates

    Args:
        center_x_boxes_default (tf.Tensor): center-x for centroids coordinates of default bounding boxes
        center_y_boxes_default (tf.Tensor): center-y for centroids coordinates of default bounding boxes
        width_boxes_default (tf.Tensor): width for centroids coordinates of default bounding boxes
        height_boxes_default (tf.Tensor): height for centroids coordinates of default bounding boxes
        standard_deviation_center_x_offsets (float): standard deviation for center-x offsets between ground truth and default bounding boxes
        standard_deviation_center_y_offsets (float): standard deviation for center-y offsets between ground truth and default bounding boxes
        standard_deviation_width_offsets (float): standard deviation for width offsets between ground truth and default bounding boxes
        standard_deviation_height_offsets (float): standard deviation for height offsets between ground truth and default bounding boxes

    Returns:
        Callable[[tf.Tensor, tf.Tensor], tf.Tensor]: the function for calculating the weighted jaccard iou bounding boxes metric
    """

    def _decode_standardized_offsets(offsets_centroids: tf.Tensor, not_background: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        decode bounding boxes standardized offsets expressed in centroids coordinates (center_x_offsets, center_y_offsets, width_offsets, height_offsets)\n

        Args:
            offsets_centroids (tf.Tensor): centroids standardized offsets (center_x_offsets, center_y_offsets, width_offsets, height_offsets) with shape (batch, total boxes, 4)
            not_background (tf.Tensor): a tensor with shape (batch, total boxes) indicating if that box it's background or not

        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]: a tuple with the decoded corners coordinates plus width and height of boxes
        """
        # split offsets centroids for easier calculations
        center_x_offsets, center_y_offsets, width_offsets, height_offsets = [
            tf.squeeze(offsets_coordinates)
            for offsets_coordinates in tf.split(value=offsets_centroids, num_or_size_splits=4, axis=-1)
        ]

        # decode offsets centroids to centroids coordinates
        center_x = center_x_offsets * standard_deviation_center_x_offsets * width_boxes_default + center_x_boxes_default
        center_y = center_y_offsets * standard_deviation_center_y_offsets * height_boxes_default + center_y_boxes_default
        width = (tf.math.exp(width_offsets * standard_deviation_width_offsets) - 1.0) * width_boxes_default
        height = (tf.math.exp(height_offsets * standard_deviation_height_offsets) - 1.0) * height_boxes_default
        
        # keep only relevant decoded centroids coordinates (not background), this is consinstent with the localization loss calculation
        # the maximum operation applied to width and height it's not right from a conversion point of view, but in this case will ensure iou <= 1.0
        # the problem here it's that we can receive offsets from a network that's still learning/training, and once decoded, they results in invalid boxes (negative width or height)
        # the maximum operation is irrelevant for ground truth data
        center_x = center_x * not_background
        center_y = center_y * not_background
        width = tf.math.maximum(0.0, width) * not_background
        height = tf.math.maximum(0.0, height) * not_background

        # convert centroids coordinates to corners coordinates
        xmin = center_x - (width - 1.0) / 2.0
        ymin = center_y - (height - 1.0) / 2.0
        xmax = center_x + (width - 1.0) / 2.0
        ymax = center_y + (height - 1.0) / 2.0

        # keep only relevant decoded corners coordinates (not background), this is consinstent with the localization loss calculation
        xmin = xmin * not_background
        ymin = ymin * not_background
        xmax = xmax * not_background
        ymax = ymax * not_background

        return xmin, ymin, xmax, ymax, width, height
        
    def jaccard_iou_bounding_boxes_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        jaccard iou metric, for object detection regression data with shape (batch, total boxes, 4)\n
        it return a single scalar metric value per batch item, which is the average iou for the non-background boxes

        Args:
            y_true (tf.Tensor): ground truth
            y_pred (tf.Tensor): predictions, expressed as probabilities

        Returns:
            tf.Tensor: a tensor with a single scalar metric value per batch item, output shape it's (batch,)
        """

        # identify classes that are not background class (encoded boxes offsets for background class are all equal to zero)
        # output shape for this tensor will be (batch, total boxes)
        sum_of_coordinates_abs_value = tf.math.reduce_sum(tf.math.abs(y_true), axis=-1)
        not_background = tf.cast(tf.math.greater(sum_of_coordinates_abs_value, 0.0), dtype=tf.float32)
        
        # decode predicted offsets and get corners coordinates, plus width and height of boxes
        xmin_pred, ymin_pred, xmax_pred, ymax_pred, width_pred, height_pred = _decode_standardized_offsets(offsets_centroids=y_pred, not_background=not_background)

        # decode predicted offsets and get corners coordinates, plus width and height of boxes
        xmin_true, ymin_true, xmax_true, ymax_true, width_true, height_true = _decode_standardized_offsets(offsets_centroids=y_true, not_background=not_background)

        # calculate corners coordinates of intersections between ground truth and predicted boxes
        # it selects the maximum for xmin and ymin coordinates, the minimum for xmax and ymax coordinates
        # also calculate width and height for intersection boxes
        xmin_intersection = tf.math.maximum(xmin_true, xmin_pred)
        ymin_intersection = tf.math.maximum(ymin_true, ymin_pred)
        xmax_intersection = tf.math.minimum(xmax_true, xmax_pred)
        ymax_intersection = tf.math.minimum(ymax_true, ymax_pred)
        width_intersection = tf.math.maximum(0.0, xmax_intersection - xmin_intersection + 1.0) * not_background
        heigth_intersection = tf.math.maximum(0.0, ymax_intersection - ymin_intersection + 1.0) * not_background

        # boxes areas
        boxes_area_true = width_true * height_true
        boxes_area_pred = width_pred * height_pred
        boxes_area_intersection = width_intersection * heigth_intersection

        # iou for all the boxes, output shape it's (batch, total boxes)
        # use a small value at denominator for avoid division by zero when dealing with boxes assigned to background class
        epsilon = tf.keras.backend.epsilon()
        metric_value = (boxes_area_intersection) / (boxes_area_pred + boxes_area_true - boxes_area_intersection + epsilon)

        # reduce by taking the average iou for each batch sample along boxes dimension, output shape it's (batch,)
        metric_value = tf.reduce_sum(metric_value, axis=-1) / tf.reduce_sum(not_background, axis=-1)

        return metric_value
    
    return jaccard_iou_bounding_boxes_metric

@tf.keras.saving.register_keras_serializable(name="categorical_accuracy")
def categorical_accuracy(classes_weights: List[float]) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """
    categorical accuracy metric, for object detection classification one-hot encoded data with shape (batch, total boxes, number of classes)\n
    you must pass some weights for you classes, and they must sum up to 1 (otherwise the calculation of the loss won't be right)\n
    predictions must be passed as probabilities, argmax it's internally applied to get classes predictions

    Args:
        classes_weights (List[float]): weights for your classes, they must sum up to 1 (otherwise the calculation of the loss won't be right)

    Returns:
        Callable[[tf.Tensor, tf.Tensor], tf.Tensor]: the function for calculating the weighted categorical accuracy metric
    """

    classes_weights = tf.constant(classes_weights, dtype=tf.float32, shape=(1, len(classes_weights)))

    def categorical_accuracy_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        categorical accuracy metric, for object detection classification one-hot encoded data with shape (batch, total boxes, number of classes)\n
        it return a single scalar metric value per batch item, which is a weighted average of the classes metrics

        Args:
            y_true (tf.Tensor): ground truth
            y_pred (tf.Tensor): predictions, expressed as probabilities

        Returns:
            tf.Tensor: a tensor with a single scalar metric value per batch item, output shape it's (batch,)
        """
        
        # convert predicted classes probabilities to one-hot classes predictions
        y_pred = tf.one_hot(tf.math.argmax(y_pred, axis=-1), depth=tf.shape(y_pred)[-1], axis=-1, dtype=tf.float32)

        # calculate the true positives for each class along boxes dimension, output shape it's (batch, number of classes)
        true_positives = tf.cast(tf.math.equal(y_pred, y_true), dtype=tf.float32)
        true_positives = tf.math.reduce_sum(true_positives, axis=1)

        # the total number of boxes it's equal to the number of samples for each class
        number_of_samples_per_class = tf.cast(tf.shape(y_true)[1], dtype=tf.float32)

        # calculate weighted average accuracy for each batch, output shape it's (batch,)
        metric_value = true_positives / number_of_samples_per_class * classes_weights 
        metric_value = tf.reduce_sum(metric_value, axis=-1)

        return metric_value
    
    return categorical_accuracy_metric
