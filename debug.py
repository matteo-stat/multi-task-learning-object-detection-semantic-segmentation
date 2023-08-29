import tensorflow as tf
tf.keras.saving.get_custom_objects().clear()
tf.random.set_seed(1993)

import json
import random
import csv
import numpy as np
from matplotlib import pyplot as plt, patches
from PIL import Image
import ssdseglib

# check environment
from os import environ
IS_KAGGLE_ENVIRONMENT = 'KAGGLE_KERNEL_RUN_TYPE' in environ

# models path
MODELS_PATH = '/kaggle/working/models/' if IS_KAGGLE_ENVIRONMENT else 'data/models/'

# data options
INPUT_IMAGE_SHAPE = (480, 640, 3)
NUMBER_OF_CLASSES = 4

# object detection options
STANDARD_DEVIATIONS_CENTROIDS_OFFSETS = (0.1, 0.1, 0.2, 0.2)

# tensorflow options
BATCH_SIZE = 32

# create default bounding boxes
boxes_default = ssdseglib.boxes.DefaultBoundingBoxes(
    feature_maps_shapes=((30, 40), (15, 20), (8, 10), (4, 5)),
    centers_padding_from_borders_percentage=0.025,
    boxes_scales=(0.15, 1.0),
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
    standard_deviations_centroids_offsets=STANDARD_DEVIATIONS_CENTROIDS_OFFSETS,
    augmentation_horizontal_flip=True
)

# test
with open('data/test.json', 'r') as f:
    path_files_images_test, path_files_masks_test, path_files_labels_boxes_test = map(list, zip(*json.load(f)))

path_files_images_test = path_files_images_test[:54]
path_files_masks_test = path_files_masks_test[:54]
path_files_labels_boxes_test = path_files_labels_boxes_test[:54]

# test
ds_test = (
    tf.data.Dataset.from_tensor_slices(path_files_images_test)
    .map(ssdseglib.datacoder.read_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size=BATCH_SIZE)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# model builder
model_builder = ssdseglib.models.MobileNetV2SsdSegBuilder(
    input_image_shape=INPUT_IMAGE_SHAPE,
    number_of_boxes_per_point=[
        len(aspect_ratios) + (1 if boxes_default.additional_square_box else 0)
        for aspect_ratios in boxes_default.feature_maps_aspect_ratios
    ],
    number_of_classes=NUMBER_OF_CLASSES,
    center_x_boxes_default=boxes_default.get_boxes_coordinates_center_x(coordinates_style='ssd'),
    center_y_boxes_default=boxes_default.get_boxes_coordinates_center_y(coordinates_style='ssd'),
    width_boxes_default=boxes_default.get_boxes_coordinates_width(coordinates_style='ssd'),
    height_boxes_default=boxes_default.get_boxes_coordinates_height(coordinates_style='ssd'),
    standard_deviations_centroids_offsets=STANDARD_DEVIATIONS_CENTROIDS_OFFSETS
)

# load trained model
model_trained = tf.keras.models.load_model(f'{MODELS_PATH}mobilenetv2-ssdseg-final-2608.keras', compile=False)

# transfer weights
model_inference = model_builder.get_model_for_inference(
    model_trained=model_trained,
    max_number_of_boxes_per_class=10,
    max_number_of_boxes_per_sample=20,
    boxes_iou_threshold=0.5,
    labels_probability_threshold=0.65,
    suppress_background_boxes=False
)

model_predictions = model_inference.predict(ds_test, use_multiprocessing=True)[1]

labels_pred_batch = model_predictions[:, :, 0].astype(int)
confidences_pred_batch = model_predictions[:, :, 1].astype(float)
boxes_pred_batch = model_predictions[:, :, 2:].astype(float)

ssdseglib.evaluators.average_precision_object_detection(
    labels_pred_batch=labels_pred_batch,
    confidences_pred_batch=confidences_pred_batch,
    boxes_pred_batch=boxes_pred_batch,
    iou_threshold=0.5,
    path_files_labels_boxes=path_files_labels_boxes_test,
    labels_codes=[0, 1, 2, 3],
    label_code_background=0
)