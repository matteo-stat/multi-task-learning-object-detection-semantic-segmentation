import os
import ssdseglib
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf

# params
batch_size = 8
num_boxes_per_batch = 8000
num_channels = 4

def create_one_hot_tensor(batch, num_boxes, channels):
    # Generate random integer indices for each category (channel)
    random_indices = tf.random.uniform(
        shape=(batch, num_boxes),
        minval=0,
        maxval=channels,
        dtype=tf.int32
    )

    # Create a one-hot encoded tensor
    one_hot_tensor = tf.one_hot(random_indices, depth=channels, axis=-1, dtype=tf.float32)    

    return one_hot_tensor

def create_unbalanced_one_hot_tensor(batch, num_boxes, channels):
    # Generate random integer indices for each category (channel)
    unbalanced_indices = []
    n_most_frequent = int(num_boxes * 0.95)
    n_less_frequent = num_boxes - n_most_frequent
    for i in range(batch):
        # Let's say you want the first channel to occur more frequently
        unbalanced_indices.extend([0] * (n_most_frequent))
        unbalanced_indices.extend(tf.random.uniform(
            shape=(n_less_frequent,),
            minval=1,
            maxval=channels,
            dtype=tf.int32
        ))
    
    unbalanced_indices = tf.convert_to_tensor(unbalanced_indices)

    # Reshape the indices tensor to match the desired shape
    indices_shape = (batch, num_boxes)
    unbalanced_indices = tf.reshape(unbalanced_indices, indices_shape)

    # Create a one-hot encoded tensor
    one_hot_tensor = tf.one_hot(unbalanced_indices, depth=channels, axis=-1, dtype=tf.float32)
    
    return one_hot_tensor


def create_logits_tensor(batch, num_boxes, channels):
    # Generate random logits for each category (channel)
    logits = tf.random.normal(
        shape=(batch, num_boxes, channels),
        mean=0,
        stddev=1
    )

    # Apply softmax along the channels dimension to ensure sum of logits is 1
    softmax_logits = tf.nn.softmax(logits, axis=-1)

    return softmax_logits

# Create the one-hot encoded tensor
y_true = create_unbalanced_one_hot_tensor(batch_size, num_boxes_per_batch, num_channels)
y_pred = create_logits_tensor(batch_size, num_boxes_per_batch, num_channels)

# --------------------------------------------------------------------------------------------------
print(ssdseglib.losses.confidence_loss(y_true, y_pred))
