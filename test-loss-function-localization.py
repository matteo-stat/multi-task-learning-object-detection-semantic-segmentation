import ssdseglib
import tensorflow as tf

# params
batch_size = 8
num_boxes_per_batch = 8000
num_channels = 4
num_background = int(0.8 * num_boxes_per_batch)

# simulate y_true
y_true_not_background = tf.random.normal(shape=(batch_size, num_boxes_per_batch - num_background, num_channels))
y_true_background = tf.zeros(shape=(batch_size, num_background, num_channels))
y_true = tf.concat([y_true_background, y_true_not_background], axis=1)
y_true = tf.reshape(y_true, (-1, num_channels))
y_true = tf.random.shuffle(y_true)
y_true = tf.reshape(y_true, (batch_size, num_boxes_per_batch, num_channels))

# simulate y_pred
y_pred = tf.random.normal(shape=(batch_size, num_boxes_per_batch, num_channels))

# --------------------------------------------------------------------------------------------------
print(ssdseglib.losses.localization_loss(y_true, y_pred))
