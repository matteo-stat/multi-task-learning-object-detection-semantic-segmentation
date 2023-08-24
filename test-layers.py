# data options
INPUT_IMAGE_SHAPE = (480, 640, 3)
NUMBER_OF_CLASSES = 4

# tensorflow options
BATCH_SIZE = 16
SEED = 1993

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
tf.random.set_seed(SEED)
tf.keras.backend.clear_session()

import json
import ssdseglib
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from typing import Union

# create default bounding boxes
boxes_default = ssdseglib.boxes.DefaultBoundingBoxes(
    feature_maps_shapes=((30, 40), (15, 20), (8, 10), (4, 5)),
    centers_padding_from_borders_percentage=0.025,
    boxes_scales=(0.2, 0.9),
    additional_square_box=True,  
)

# rescale default bounding boxes to input image shape
boxes_default.rescale_boxes_coordinates(image_shape=INPUT_IMAGE_SHAPE[:2])

# create a data reader encoder
data_reader_encoder = ssdseglib.datacoder.DataEncoderDecoder(
    num_classes=NUMBER_OF_CLASSES,
    image_shape=INPUT_IMAGE_SHAPE[:2],
    xmin_boxes_default=boxes_default.get_boxes_coordinates_xmin(coordinates_style='ssd'),
    ymin_boxes_default=boxes_default.get_boxes_coordinates_ymin(coordinates_style='ssd'),
    xmax_boxes_default=boxes_default.get_boxes_coordinates_xmax(coordinates_style='ssd'),
    ymax_boxes_default=boxes_default.get_boxes_coordinates_ymax(coordinates_style='ssd'),
    iou_threshold=0.5,
    standard_deviations_centroids_offsets=(0.1, 0.1, 0.2, 0.2),
    augmentation_horizontal_flip=True
)


decode_boxes_centroids_offsets = ssdseglib.layers.DecodeBoxesCentroidsOffsets(
    center_x_boxes_default=data_reader_encoder.center_x_boxes_default,
    center_y_boxes_default=data_reader_encoder.center_y_boxes_default,
    width_boxes_default=data_reader_encoder.width_boxes_default,
    height_boxes_default=data_reader_encoder.height_boxes_default,
    standard_deviation_center_x_offsets=data_reader_encoder.standard_deviation_center_x_offsets,
    standard_deviation_center_y_offsets=data_reader_encoder.standard_deviation_center_y_offsets,
    standard_deviation_width_offsets=data_reader_encoder.standard_deviation_width_offsets,
    standard_deviation_height_offsets=data_reader_encoder.standard_deviation_height_offsets
)
decode_boxes_centroids_offsets.trainable = False

    
non_maximum_suppression = ssdseglib.layers.NonMaximumSuppression(max_number_of_boxes_per_class=5, max_number_of_boxes_per_sample=10, boxes_iou_threshold=0.5, labels_probability_threshold=0.6)
non_maximum_suppression.trainable = False


# test
with open('data/test.json', 'r') as f:
    path_images_test, path_masks_test, path_labels_boxes_test = map(list, zip(*json.load(f)))

# load model
model = tf.keras.models.load_model(
    filepath='data/models/model-saved-66-epoch.keras',
    compile=False
)

shuffled_indices = np.arange(len(path_images_test))
np.random.shuffle(shuffled_indices)

path_images_test = np.array(path_images_test)[shuffled_indices]
path_masks_test = np.array(path_masks_test)[shuffled_indices]
path_labels_boxes_test = np.array(path_labels_boxes_test)[shuffled_indices]

for path_image, path_mask, path_labels_boxes in zip(path_images_test[:10], path_masks_test[:10], path_labels_boxes_test[:10]):
    image = Image.open(path_image)
    image_batch = np.array(image).astype(np.float32)
    image_batch = np.expand_dims(image, axis=0)

    mask, labels_probabilities, boxes_centroids_offsets = model(image_batch, training=False)

    boxes_corners_decoded = decode_boxes_centroids_offsets(boxes_centroids_offsets)

    object_detection_output = non_maximum_suppression(boxes_corners_coordinates=boxes_corners_decoded, labels_probabilities=labels_probabilities)

    from matplotlib import patches
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
    ssdseglib.plot.move_figure(fig=fig, x=0, y=0)
         

    # labels conversions
    label_code_to_str = {
        0: 'background',
        1: 'monorail',
        2: 'person',
        3: 'forklift'
    }
    label_code_to_color = {
        0: 'white',
        1: 'red',
        2: 'green',
        3: 'blue'
    }

    ax1.set_aspect('equal')
    ax1.imshow(image, vmin=0, vmax=255)
    ax1.set_axis_off()

    for label, label_prob, xmin, ymin, xmax, ymax in object_detection_output:
        label = int(label)
        label_prob = float(label_prob)
        xmin = float(xmin)
        ymin = float(ymin)
        xmax = float(xmax)
        ymax = float(ymax)        
        rect = patches.Rectangle((xmin, ymin), xmax - xmin + 1, ymax - ymin + 1, linewidth=1, edgecolor=label_code_to_color[label], facecolor='none')
        ax1.add_patch(rect)
        ax1.text(xmin, ymin, label_code_to_str[label] + f' {int(label_prob*100)}%', fontsize=8, color=label_code_to_color[label], verticalalignment='top')        

        # mask = tf.math.argmax(tf.squeeze(mask, axis=0), axis=-1)
        # mask = tf.one_hot(mask, 4, dtype=tf.float32)
        mask_sample = tf.slice(tf.squeeze(mask, axis=0), begin=[0, 0, 1], size=[-1, -1, 3])

        # setup the subplot
        ax2.set_aspect('equal')
        ax2.imshow(mask_sample, vmin=0.0, vmax=1.0)
        ax2.set_axis_off()

    plt.show()

