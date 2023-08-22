from typing import List, Callable
import tensorflow as tf

@tf.keras.saving.register_keras_serializable(package="ssdseglib", name="localization_loss")
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

    # divide the loss by the number of samples that are not background
    # if there are only background samples, then divide by 1
    smooth_l1_loss = smooth_l1_loss / tf.math.maximum(tf.math.reduce_sum(not_background, axis=-1), 1.0)

    return smooth_l1_loss

@tf.keras.saving.register_keras_serializable(package="ssdseglib", name="confidence_loss")
def confidence_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    confidence loss, which is a slightly modified softmax loss\n
    usually most default bounding boxes are encoded and assigned to the background class\n
    this can lead to an unbalanced loss for the classification\n
    following the ssd paper proposal, in order to solve the unbalance problem, this loss try to keep a ratio of 3 between background and other classes samples

    Args:
        y_true (tf.Tensor): ground truth
        y_pred (tf.Tensor): predictions, expressed as probabilities

    Returns:
        tf.Tensor: a tensor with a single scalar loss value per batch item, output shape it's (batch,)
    """   
    # --------------------------------------------------------------------------------------------------
    # background class and not background classes
    # --------------------------------------------------------------------------------------------------
    # indicator tensors for background class and other classes
    is_background = y_true[:, :, 0]
    not_background = tf.math.abs(is_background - 1.0)

    # count total number of samples for background class and other classes
    background_samples = tf.math.count_nonzero(is_background, dtype=tf.int32)
    not_background_samples = tf.math.count_nonzero(not_background, dtype=tf.int32)

    # --------------------------------------------------------------------------------------------------
    # softmax loss for all classes
    # --------------------------------------------------------------------------------------------------
    # here we are gonna calculate the softmax loss for all the classes, but unfortunately we'll have to deal with a problem
    # as the original ssd paper pointed out, most of the default bounding boxes will be assigned to the background class
    # this translates in a class imbalancing problem, where the background class it's dominant, while the other ones are more rare
    # to address this problem the ssd paper propose to keep a 3:1 ratio between background class and the other ones
    # this means that we should return a final loss more elaborated than the simple softmax for all the classes
    # in order to do that, we'll split the calculation in two step, first not background classes, then background
    # finally we'll sum them up and return the loss at the end

    # calculate logarithm of predicted probabilities, accounting for extreme values
    epsilon = tf.keras.backend.epsilon()
    log_y_pred = tf.math.log(tf.clip_by_value(y_pred, clip_value_min=epsilon, clip_value_max=1-epsilon))

    # calculate the softmax loss for all classes, output shape it's (batch, total boxes)
    softmax_loss_all_classes = -tf.math.reduce_sum(y_true * log_y_pred, axis=-1)

    # --------------------------------------------------------------------------------------------------
    # softmax loss for other classes (not background)
    # --------------------------------------------------------------------------------------------------
    # keep only loss data for classes that are different from background
    # then reduce it by taking the sum (total loss) along boxes dimension, output shape it's (batch, )
    softmax_loss_not_background = softmax_loss_all_classes * not_background
    softmax_loss_not_background = tf.math.reduce_sum(softmax_loss_not_background, axis=-1)

    # --------------------------------------------------------------------------------------------------
    # softmax loss for background class
    # --------------------------------------------------------------------------------------------------
    # the idea is to keep a number of background samples 3 times higher than the samples related to other classes
    # however in the extremely rare event that background samples are less than other classes samples, we'll just keep them
    # this is important to keep only the relevant background data when using top_k tensorflow function
    number_of_background_samples_to_keep = tf.math.minimum(3 * not_background_samples, background_samples)

    # keep only loss data for background class, output shape it's (batch, total boxes)
    softmax_loss_background = softmax_loss_all_classes * is_background

    # in the extreme case of no background classes, we'll just return zero as loss for background class
    if tf.math.equal(background_samples, tf.constant(0, dtype=tf.int32)):
        softmax_loss_background = tf.zeros_like(softmax_loss_not_background)
    else:
        # flatten the loss data to 1d tensor, output shape it's (batch * total boxes,)
        # this is the required format by the top_k tensorflow function
        softmax_loss_background_1d = tf.reshape(softmax_loss_background, shape=(-1,))

        # keep the previously calculated number of background samples with the highest loss
        # this will give us the indexes of the top-k background samples with highest loss
        _, background_samples_to_keep = tf.math.top_k(
            input=softmax_loss_background_1d,
            k=number_of_background_samples_to_keep,
            sorted=False
        )

        # create a dummy tensor, with shape (batch * total boxes,)
        # it's equal to 1 at the indexes of the top-k background samples with highest loss, 0 elsewhere
        background_samples_to_keep = tf.scatter_nd(
            indices=tf.expand_dims(background_samples_to_keep, axis=1),
            updates=tf.ones_like(background_samples_to_keep, dtype=tf.int32),
            shape=tf.shape(softmax_loss_background_1d)
        )

        # reshape the dummy tensor, so that output shape it's (batch, total boxes)
        # then convert it to float, because we'll use for selecting relevant softmax loss values for background
        background_samples_to_keep = tf.reshape(background_samples_to_keep, shape=tf.shape(softmax_loss_background))
        background_samples_to_keep = tf.cast(background_samples_to_keep, dtype=tf.float32)

        # now that we have succesfully created a tensor indicator for the background samples to keep,
        # we are able to mask out unwanted samples data for background class
        softmax_loss_background = softmax_loss_background * background_samples_to_keep

        # reduce the loss by taking the sum (total loss) along boxes dimension, output shape it's (batch, )
        softmax_loss_background = tf.math.reduce_sum(softmax_loss_background, axis=-1)
        
    # --------------------------------------------------------------------------------------------------
    # softmax loss, final step
    # --------------------------------------------------------------------------------------------------
    # now we are finally able to compute the final loss, by simply summing up the loss for background class and other classes
    softmax_loss_balanced = softmax_loss_not_background + softmax_loss_background

    # divide the loss by the number of samples that are not background
    # if there are only background samples, then divide by 1
    softmax_loss_balanced = softmax_loss_balanced / tf.math.maximum(tf.math.reduce_sum(not_background, axis=-1), 1.0)

    return softmax_loss_balanced

@tf.keras.saving.register_keras_serializable(package="ssdseglib", name="dice_loss")
def dice_loss(classes_weights: List[float]) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """
    dice loss, for one-hot encoded semantic segmentation masks with shape (batch, height, width, number of classes)\n
    you must pass some weights for you classes, and they must sum up to 1 (otherwise the calculation of the loss won't be right)\n
    predictions must be passed as probabilities

    Args:
        classes_weights (List[float]): weights for your classes, they must sum up to 1 (otherwise the calculation of the loss won't be right)

    Returns:
        Callable[[tf.Tensor, tf.Tensor], tf.Tensor]: the function for calculating the weighted dice loss
    """

    classes_weights = tf.constant(classes_weights, dtype=tf.float32, shape=(1, len(classes_weights)))

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        dice loss, for one-hot encoded semantic segmentation masks with shape (batch, height, width, number of classes)\n
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

        # dice loss, with laplace smoothing for managing missing class values
        loss_value = 1.0 - (2.0 * intersection + 1.0) / (total + 1.0)

        # weighted average along classes dimension, output shape it's (batch,)
        loss_value = loss_value * classes_weights
        loss_value = tf.reduce_sum(loss_value, axis=-1)

        return loss_value
    
    return loss

@tf.keras.saving.register_keras_serializable(package="ssdseglib", name="dice_square_loss")
def dice_square_loss(classes_weights: List[float]) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """
    dice square loss, for one-hot encoded semantic segmentation masks with shape (batch, height, width, number of classes)\n
    you must pass some weights for you classes, and they must sum up to 1 (otherwise the calculation of the loss won't be right)\n
    predictions must be passed as probabilities

    Args:
        classes_weights (List[float]): weights for your classes, they must sum up to 1 (otherwise the calculation of the loss won't be right)

    Returns:
        Callable[[tf.Tensor, tf.Tensor], tf.Tensor]: the function for calculating the weighted dice square loss
    """

    classes_weights = tf.constant(classes_weights, dtype=tf.float32, shape=(1, len(classes_weights)))

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        dice square loss, for one-hot encoded semantic segmentation masks with shape (batch, height, width, number of classes)\n
        it return a single scalar loss value per batch item, which is a weighted average of the classes losses

        Args:
            y_true (tf.Tensor): ground truth
            y_pred (tf.Tensor): predictions, expressed as probabilities

        Returns:
            tf.Tensor: a tensor with a single scalar loss value per batch item, output shape it's (batch,)
        """
        
        # intersection area, along height and width dimensions, output shape it's (batch, number of classes)
        intersection = tf.math.reduce_sum(y_true * y_pred, axis=(1, 2))

        # total of squares, along height and width dimensions, output shape it's (batch, number of classes)
        total_of_squares = tf.math.reduce_sum(tf.math.pow(y_true, 2) + tf.math.pow(y_pred, 2), axis=(1, 2))

        # dice square loss, with laplace smoothing for managing missing class values
        loss_value = 1.0 - (2.0 * intersection + 1.0) / (total_of_squares + 1.0)

        # weighted average along classes dimension, output shape it's (batch,)
        loss_value = loss_value * classes_weights
        loss_value = tf.reduce_sum(loss_value, axis=-1)

        return loss_value
    
    return loss

@tf.keras.saving.register_keras_serializable(package="ssdseglib", name="cross_entropy_loss")
def cross_entropy_loss(classes_weights: List[float]) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """
    cross entropy / softmax loss, for one-hot encoded semantic segmentation masks with shape (batch, height, width, number of classes)\n
    you must pass some weights for you classes, and they must sum up to 1 (otherwise the calculation of the loss won't be right)\n
    predictions must be passed as probabilities

    Args:
        classes_weights (List[float]): weights for your classes, they must sum up to 1 (otherwise the calculation of the loss won't be right)

    Returns:
        Callable[[tf.Tensor, tf.Tensor], tf.Tensor]: the function for calculating the weighted cross entropy loss
    """

    classes_weights = tf.constant(classes_weights, dtype=tf.float32, shape=(1, len(classes_weights)))

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        cross entropy / softmax, for one-hot encoded semantic segmentation masks with shape (batch, height, width, number of classes)\n
        it return a single scalar loss value per batch item, which is a weighted average of the classes losses

        Args:
            y_true (tf.Tensor): ground truth
            y_pred (tf.Tensor): predictions, expressed as probabilities

        Returns:
            tf.Tensor: a tensor with a single scalar loss value per batch item, output shape it's (batch,)
        """
        
        # calculate logarithm of predicted probabilities, accounting for extreme values
        epsilon = tf.keras.backend.epsilon()
        log_y_pred = tf.math.log(tf.clip_by_value(y_pred, clip_value_min=epsilon, clip_value_max=1-epsilon))

        # calculate the softmax loss for all classes, output shape it's (batch, number of classes)
        loss_value = -tf.math.reduce_sum(y_true * log_y_pred, axis=(1, 2))

        # weighted average along classes dimension, output shape it's (batch,)
        loss_value = loss_value * classes_weights
        loss_value = tf.reduce_sum(loss_value, axis=-1)

        return loss_value
    
    return loss
