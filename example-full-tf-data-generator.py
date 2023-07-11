import random
import json
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import ssd
import plot
from PIL import Image

# global variables
INPUT_IMAGE_SHAPE = (480, 640)
SHUFFLE_BUFFER_SIZE = 512
BATCH_SIZE = 32
SEED = 1993

# load metadata
with open('data/train.json', 'r') as f:
    path_images_train, path_masks_train, path_labels_boxes_train = map(list, zip(*json.load(f)))

# create default bounding boxes
boxes_default = ssd.generate_default_bounding_boxes(
    feature_maps_shapes=((24, 32), (12, 16), (6, 8), (3, 4)),
    feature_maps_aspect_ratios=(1.0, 2.0, 3.0, 1/2, 1/3),
    centers_padding_from_borders=0.5,
    boxes_scales=(0.2, 0.9),
    additional_square_box=True,  
)

# default bounding boxes, reshaped as output from the network
boxes_default = np.concatenate([np.reshape(boxes_default, newshape=(-1, 4)) for boxes_default in boxes_default], axis=0)

# default bounding boxes, scaled at image shape
boxes_default[:, [0, 2]] = boxes_default[:, [0, 2]] * INPUT_IMAGE_SHAPE[1]
boxes_default[:, [1, 3]] = boxes_default[:, [1, 3]] * INPUT_IMAGE_SHAPE[0]

class DataReaderEncoder:
    def __init__(
            self,
            num_classes: int,
            boxes_default: np.ndarray,
            iou_threshold: float = 0.5,
            boxes_centroids_offsets_std: tuple[float] = (0.1, 0.1, 0.2, 0.2)
        ) -> None:

        # initialize args
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.boxes_centroids_offsets_std = tf.convert_to_tensor(boxes_centroids_offsets_std, dtype=tf.float32)


        # corners coordinates for each default bounding box
        self.xmin_boxes_default, self.ymin_boxes_default, self.xmax_boxes_default, self.ymax_boxes_default = tf.split(tf.convert_to_tensor(boxes_default, dtype=tf.float32), 4, axis=-1)

        # calculate area for each default bounding box
        self.area_boxes_default = (self.ymax_boxes_default - self.ymin_boxes_default + 1) * (self.xmax_boxes_default - self.xmin_boxes_default + 1)

    def _bounding_boxes_corners_to_centroids(self, xmin, ymin, xmax, ymax):
        center_x = (xmax + xmin) / 2
        center_y = (ymax + ymin) / 2
        width = xmax - xmin + 1
        height = ymax - ymin + 1

        return center_x, center_y, width, height

    def read_encode_data(
            self,
            path_image: str,
            path_mask: str,
            path_labels_boxes: str,
        ):

        # read csv file as text, split by lines and decode csv data to tensors
        labels_boxes = tf.strings.strip(tf.io.read_file(path_labels_boxes))
        labels_boxes = tf.strings.split(labels_boxes, sep='\r\n')
        labels_boxes = tf.io.decode_csv(labels_boxes, record_defaults=[int(), float(), float(), float(), float()])

        # create labels and boxes tensor
        labels_ground_truth, xmin_boxes_ground_truth, ymin_boxes_ground_truth, xmax_boxes_ground_truth, ymax_boxes_ground_truth = labels_boxes

        # calculate area of each bounding box
        area_boxes_ground_truth = (xmax_boxes_ground_truth - xmin_boxes_ground_truth + 1) * (ymax_boxes_ground_truth - ymin_boxes_ground_truth + 1)

        # coordinates of intersections between each default bounding box and all ground truth bounding boxes
        # it selects the maximum for xmin and ymin coordinates, the minimum for xmax and ymax coordinates
        xmin_boxes_intersection = tf.maximum(self.xmin_boxes_default, tf.transpose(xmin_boxes_ground_truth))
        ymin_boxes_intersection = tf.maximum(self.ymin_boxes_default, tf.transpose(ymin_boxes_ground_truth))
        xmax_boxes_intersection = tf.minimum(self.xmax_boxes_default, tf.transpose(xmax_boxes_ground_truth))
        ymax_boxes_intersection = tf.minimum(self.ymax_boxes_default, tf.transpose(ymax_boxes_ground_truth))

        # area of intersection between each default bounding box and all ground truth bounding boxes
        area_boxes_intersection = tf.maximum(0, xmax_boxes_intersection - xmin_boxes_intersection + 1) * tf.maximum(0, ymax_boxes_intersection - ymin_boxes_intersection + 1)

        # calculate intersection over union between each default bounding box and all ground truth bounding boxes
        # note that this is a matrix with shape (num default bounding boxes, num ground truth bounding boxes)
        iou = area_boxes_intersection / (self.area_boxes_default + tf.transpose(area_boxes_ground_truth) - area_boxes_intersection)

        # find the best match between each ground truth box and all default bounding boxes
        # note that the output shape will be (num ground truth boxes with iou > 0 with at least one default box, 2)
        # the last axis contains indexes for default boxes, ground truth boxes
        indexes_match_ground_truth = tf.stack([tf.math.argmax(iou, axis=0, output_type=tf.dtypes.int32), tf.range(len(xmin_boxes_ground_truth))], axis=1)
        indexes_match_ground_truth = tf.boolean_mask(tensor=indexes_match_ground_truth, mask=tf.math.greater(tf.reduce_max(iou, axis=0), 0), axis=0)

        # find the best match between each default box and all ground truth bounding boxes
        # note that the output shape will be (num default truth boxes with iou > threshold with at least one ground truth box, 2)
        # the last axis contains indexes for default boxes, ground truth boxes
        indexes_match_default = tf.stack([tf.range(len(self.xmin_boxes_default)), tf.math.argmax(iou, axis=1, output_type=tf.dtypes.int32)], axis=1)
        indexes_match_default = tf.boolean_mask(tensor=indexes_match_default, mask=tf.math.greater(tf.reduce_max(iou, axis=1), self.iou_threshold), axis=0)

        # put the best matches together, removing possible pair duplicates
        tf.math.logical_not(tf.reduce_any(tf.equal(indexes_match_default[:, tf.newaxis], indexes_match_ground_truth), axis=1))
        indexes_match, _ = tf.raw_ops.UniqueV2(x=tf.concat([indexes_match_ground_truth, indexes_match_default], axis=0), axis=[0])

        # get the class labels for each best match and one-hot encode them (0 is reserved for the background class)
        labels_match = tf.gather(labels_ground_truth, indexes_match[:, 1])
        labels_match = tf.one_hot(labels_match, self.num_classes)

        # convert box coordinates from corners to centroids
        centroids_default_center_x, centroids_default_center_y, centroids_default_width, centroids_default_height = self._bounding_boxes_corners_to_centroids(
            xmin=tf.gather(self.xmin_boxes_default, indexes_match[:, 0]),
            ymin=tf.gather(self.ymin_boxes_default, indexes_match[:, 0]),
            xmax=tf.gather(self.xmax_boxes_default, indexes_match[:, 0]),
            ymax=tf.gather(self.ymax_boxes_default, indexes_match[:, 0]),
        )
        centroids_ground_truth_center_x, centroids_ground_truth_center_y, centroids_ground_truth_width, centroids_ground_truth_height = self._bounding_boxes_corners_to_centroids(
            xmin=tf.gather(xmin_boxes_ground_truth, indexes_match[:, 1]),
            ymin=tf.gather(ymin_boxes_ground_truth, indexes_match[:, 1]),
            xmax=tf.gather(xmax_boxes_ground_truth, indexes_match[:, 1]),
            ymax=tf.gather(ymax_boxes_ground_truth, indexes_match[:, 1]),
        )


        centroids_default_center_x, centroids_default_center_y, centroids_default_width, centroids_default_height = ssd.bounding_boxes_corners_to_centroids(boxes=tf.gather(self.boxes_default, indexes_match[:, 0]))
        centroids_ground_truth = ssd.bounding_boxes_corners_to_centroids(boxes=tf.gather(boxes_ground_truth, indexes_match[:, 1]))

        # default bounding boxes encoded as required (one-hot encoding for classes, offsets for centroid coordinates)
        boxes_encoded = tf.zeros(shape=(tf.shape(self.boxes_default)[0], self.num_classes + 4))
        boxes_encoded = tf.tensor_scatter_nd_update(boxes_encoded, tf.expand_dims(indexes_match[:, 0], axis=1), labels_match)
        boxes_encoded = tf.tensor_scatter_nd_update(boxes_encoded, tf.expand_dims(indexes_match[:, 0], axis=1), (centroids_ground_truth[:, [0, 1]] - centroids_default[:, [0, 1]]) / centroids_default[:, [2, 3]] / self.boxes_centroids_offsets_std[:2])
        boxes_encoded = tf.tensor_scatter_nd_update(boxes_encoded, tf.expand_dims(indexes_match[:, 0], axis=1), tf.math.log(centroids_ground_truth[:, [2, 3]] / centroids_default[:, [2, 3]]) / self.boxes_centroids_offsets_std[-2:])

        # read the image, scale values between 0 and 1
        image = tf.image.decode_image(tf.io.read_file(path_image), dtype=tf.float32)
        image = tf.image.convert_image_dtype(image, tf.float32)

        # read the mask, one-hot encode the 4 classes, squeeze out unwanted dimension
        mask = tf.one_hot(tf.squeeze(tf.image.decode_image(tf.io.read_file(path_mask))), 4, dtype=tf.float32)


        a = 0

dataReaderEncoder = DataReaderEncoder(
    num_classes=4,
    boxes_default=boxes_default,
    iou_threshold=0.5,
    boxes_centroids_offsets_std=(0.1, 0.1, 0.2, 0.2),
)

dataReaderEncoder.read_encode_data('data/train/2335.png', 'data/train/2335_mask.png', 'data/train/2335_labels_boxes.csv')
