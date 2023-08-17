import tensorflow as tf

def localization_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    localization loss, which is a slightly modified smooth l1 loss\n
    this loss it's calculated only for classes that are different background, because there are no boxes for background\n
    if no classes are present then the loss will be zero\n
    note that this loss it's averaged by the number of non background classes\n
    in the last step the loss values per batch are multiplied by the batch size to avoid batch averaging

    Args:
        y_true (tf.Tensor): ground truth
        y_pred (tf.Tensor): predictions

    Returns:
        tf.Tensor: a tensor with a single scalar loss value per batch item, output shape it's (batch,)
    """    

    # identify classes that are not background class (encoded boxes offsets for background class are all equal to zero)
    # output shape for this tensor will be (batch, total boxes)
    sum_of_coordinates_abs_value = tf.math.reduce_sum(tf.math.abs(y_true), axis=-1)
    not_background = tf.cast(tf.math.greater(sum_of_coordinates_abs_value, 0.0), dtype=tf.float32)

    # calculate prediction error in absolute and squared terms, useful for calculating smooth l1 loss
    prediction_error_absolute = tf.math.abs(y_true - y_pred)
    prediction_error_squared = tf.math.pow(y_true - y_pred, 2)

    # condition for calculating the smooth l1 loss
    smooth_l1_loss_condition = tf.math.less(prediction_error_absolute, 1.0)

    # calculate the smooth l1 loss for each coordinate, output shape it's (batch, total boxes, 4)
    smooth_l1_loss = tf.where(smooth_l1_loss_condition, prediction_error_squared * 0.5, prediction_error_absolute - 0.5)

    # sum up the loss along coordinates dimension, output shape it's (batch, total boxes)
    smooth_l1_loss = tf.math.reduce_sum(smooth_l1_loss, axis=-1)

    # keep smooth l1 loss entries only for background classes different from background
    smooth_l1_loss = smooth_l1_loss * not_background

    # sum up the loss along the boxes dimension, output shape it's (batch,)
    smooth_l1_loss = tf.math.reduce_sum(smooth_l1_loss, axis=-1)

    # divide the loss by the number of classes that are not background in the batch
    # if there are no classes other than background one, then the loss will be simply zero
    smooth_l1_loss = smooth_l1_loss / tf.math.maximum(tf.math.reduce_sum(not_background, axis=-1), 1.0)

    # tensorflow/keras automatically apply a reduction function to the loss along the batch dimension, in order to get a single scalar loss value
    # the reduction function applied by default it's the mean (sum along batch dimension, then divide by batch size)
    # unfortunately we already divided / averaged the loss by the number of classes different from background 
    # to avoid the default averaging along the batch dimension, we can multiply the loss by the batch size
    # in this way we get the sum of the smooth l1 loss along the batch dimension as a single scalar loss value
    batch_size = tf.cast(tf.shape(y_true)[0], dtype=tf.float32)
    smooth_l1_loss = smooth_l1_loss / batch_size

    return smooth_l1_loss
