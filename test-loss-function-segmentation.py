import numpy as np
import tensorflow as tf

def generate_ground_truth_mask(batch_size, height, width, num_classes):
    mask_labels = np.random.randint(0, num_classes, size=(batch_size, height, width))
    one_hot_masks = np.eye(num_classes)[mask_labels]
    one_hot_masks = one_hot_masks.astype(np.float32)
    
    return tf.convert_to_tensor(one_hot_masks)

def generate_predicted_mask(batch_size, height, width, num_classes):
    predicted_mask = np.random.random(size=(batch_size, height, width, num_classes))
    predicted_mask /= np.sum(predicted_mask, axis=-1, keepdims=True)
    predicted_mask = predicted_mask.astype(np.float32)

    return tf.convert_to_tensor(predicted_mask)

batch_size = 8
height = 2
width = 3
num_classes = 4

y_true = generate_ground_truth_mask(batch_size, height, width, num_classes)
y_pred = generate_predicted_mask(batch_size, height, width, num_classes)


classes_weights = tf.constant([0.25, 0.25, 0.25, 0.25], shape=(1, 4), dtype=tf.float32)
intersection = tf.math.reduce_sum(y_true * y_pred, axis=(1, 2))
total = tf.math.reduce_sum(y_true + y_pred, axis=(1, 2))
dice_loss = 1.0 - (2 * intersection + 1.0) / (total + 1.0)
dice_loss = (y_true + y_pred) * classes_weights
dice_loss = tf.reduce_sum(dice_loss, axis=1)

# return 1 - numerator / denominator
