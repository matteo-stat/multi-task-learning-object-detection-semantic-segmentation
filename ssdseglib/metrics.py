from typing import List, Callable
import tensorflow as tf

def jaccard_iou_segmentation_masks(classes_weights: List[float]) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """
    jaccard iou metric, for one-hot encoded semantic segmentation masks with shape (batch, height, width, number of classes)\n
    you must pass some weights for you classes, and they must sum up to 1 (otherwise the calculation of the loss won't be right)\n
    predictions must be passed as probabilities

    Args:
        classes_weights (List[float]): weights for your classes, they must sum up to 1 (otherwise the calculation of the loss won't be right)

    Returns:
        Callable[[tf.Tensor, tf.Tensor], tf.Tensor]: the function for calculating the weighted jaccard iou metric
    """

    classes_weights = tf.constant(classes_weights, dtype=tf.float32, shape=(1, len(classes_weights)))

    def metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        jaccard iou metric, for one-hot encoded semantic segmentation masks with shape (batch, height, width, number of classes)\n
        it return a single scalar loss value per batch item, which is a weighted average of the classes losses

        Args:
            y_true (tf.Tensor): ground truth
            y_pred (tf.Tensor): predictions, expressed as probabilities

        Returns:
            tf.Tensor: a tensor with a single scalar loss value per batch item, output shape it's (batch,)
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
    
    return metric
