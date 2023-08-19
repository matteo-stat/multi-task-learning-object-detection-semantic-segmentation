import numpy as np
import tensorflow as tf
import ssdseglib

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

# --------------------------------------------------------------------------------------------------
jaccard_iou_segmentation_masks = ssdseglib.metrics.jaccard_iou_segmentation_masks(classes_weights=[0., 0.4, 0.3, 0.3])
print(jaccard_iou_segmentation_masks(y_true, y_pred))
