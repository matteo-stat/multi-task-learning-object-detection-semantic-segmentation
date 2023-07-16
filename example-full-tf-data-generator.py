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
BATCH_SIZE = 4
SEED = 1993

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

# corners coordinates for default bounding boxes, scaled at image shape
xmin_boxes_default, ymin_boxes_default, xmax_boxes_default, ymax_boxes_default = np.split(boxes_default, 4, axis=-1)
xmin_boxes_default = xmin_boxes_default.reshape(-1) * INPUT_IMAGE_SHAPE[1]
ymin_boxes_default = ymin_boxes_default.reshape(-1) * INPUT_IMAGE_SHAPE[0]
xmax_boxes_default = xmax_boxes_default.reshape(-1) * INPUT_IMAGE_SHAPE[1]
ymax_boxes_default = ymax_boxes_default.reshape(-1) * INPUT_IMAGE_SHAPE[0]

class DataReaderEncoder:
    def __init__(
            self,
            num_classes: int,
            xmin_boxes_default: np.ndarray,
            yxmin_boxes_default: np.ndarray,
            xmax_boxes_default: np.ndarray,
            ymax_boxes_default: np.ndarray,
            iou_threshold: float = 0.5,
            offsets_std: tuple[float] = (0.1, 0.1, 0.2, 0.2)
        ) -> None:
        # initialize args
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.offsets_center_x_std, self.offsets_center_y_std, self.offsets_width_std, self.offsets_height_std = offsets_std

        # corners coordinates for default bounding boxes
        self.xmin_boxes_default = tf.convert_to_tensor(xmin_boxes_default, dtype=tf.float32)
        self.ymin_boxes_default = tf.convert_to_tensor(ymin_boxes_default, dtype=tf.float32)
        self.xmax_boxes_default = tf.convert_to_tensor(xmax_boxes_default, dtype=tf.float32)
        self.ymax_boxes_default = tf.convert_to_tensor(ymax_boxes_default, dtype=tf.float32)

        # calculate area for default bounding boxes
        self.area_boxes_default = tf.expand_dims(
            input=(self.ymax_boxes_default - self.ymin_boxes_default + 1.) * (self.xmax_boxes_default - self.xmin_boxes_default + 1.),
            axis=1
        )

    def _bounding_boxes_corners_to_centroids(self, xmin, ymin, xmax, ymax):
        # calculate bounding boxes centroids coordinates
        center_x = (xmax + xmin) / 2.
        center_y = (ymax + ymin) / 2.
        width = xmax - xmin + 1.
        height = ymax - ymin + 1.

        return center_x, center_y, width, height

    def _bounding_boxes_centroids_to_corners(self, center_x, center_y, width, height):
        # calculate bounding boxes corners coordinates
        xmin = center_x - (width - 1.) / 2.
        ymin = center_y - (height - 1.) / 2.
        xmax = center_x + (width - 1.) / 2.
        ymax = center_y + (height - 1.) / 2.

        return xmin, ymin, xmax, ymax
    
    def _encode_ground_truth_bounding_boxes(self, path_labels_boxes: str):
        # read labels boxes csv file as text, split text by lines and then decode csv data to tensors
        labels_boxes = tf.strings.strip(tf.io.read_file(path_labels_boxes))
        labels_boxes = tf.strings.split(labels_boxes, sep='\r\n')
        labels_boxes = tf.io.decode_csv(labels_boxes, record_defaults=[int(), float(), float(), float(), float()])

        # create ground truth labels and boxes tensor
        labels_ground_truth, xmin_boxes_ground_truth, ymin_boxes_ground_truth, xmax_boxes_ground_truth, ymax_boxes_ground_truth = labels_boxes

        # calculate area for ground truth bounding boxes
        area_boxes_ground_truth = (xmax_boxes_ground_truth - xmin_boxes_ground_truth + 1.) * (ymax_boxes_ground_truth - ymin_boxes_ground_truth + 1.)

        # coordinates of intersections between each default bounding box and all ground truth bounding boxes
        # it selects the maximum for xmin and ymin coordinates, the minimum for xmax and ymax coordinates
        xmin_boxes_intersection = tf.maximum(tf.expand_dims(self.xmin_boxes_default, axis=1), tf.transpose(xmin_boxes_ground_truth))
        ymin_boxes_intersection = tf.maximum(tf.expand_dims(self.ymin_boxes_default, axis=1), tf.transpose(ymin_boxes_ground_truth))
        xmax_boxes_intersection = tf.minimum(tf.expand_dims(self.xmax_boxes_default, axis=1), tf.transpose(xmax_boxes_ground_truth))
        ymax_boxes_intersection = tf.minimum(tf.expand_dims(self.ymax_boxes_default, axis=1), tf.transpose(ymax_boxes_ground_truth))

        # area of intersection between each default bounding box and all ground truth bounding boxes
        area_boxes_intersection = tf.maximum(0., xmax_boxes_intersection - xmin_boxes_intersection + 1.) * tf.maximum(0., ymax_boxes_intersection - ymin_boxes_intersection + 1.)

        # calculate intersection over union between each default bounding box and all ground truth bounding boxes
        # note that this is a matrix with shape (num default bounding boxes, num ground truth bounding boxes)
        iou = area_boxes_intersection / (self.area_boxes_default + tf.transpose(area_boxes_ground_truth) - area_boxes_intersection)

        # now find the best match with 3 steps
        # first one find a match for each ground truth box
        # second one find a match for each default box
        # third one put the results together removing possible duplicates coming from the union of previous steps

        # step 1 - find the best match between each ground truth box and all default bounding boxes
        # note that the output shape will be (num ground truth boxes with iou > 0 with at least one default box, 2)
        # this matrix-like tensor contains indexes for default boxes and ground truth boxes
        indexes_match_ground_truth = tf.stack([tf.math.argmax(iou, axis=0, output_type=tf.dtypes.int32), tf.range(len(xmin_boxes_ground_truth))], axis=1)
        indexes_match_ground_truth = tf.boolean_mask(tensor=indexes_match_ground_truth, mask=tf.math.greater(tf.reduce_max(iou, axis=0), 0.), axis=0)

        # step 2 - find the best match between each default box and all ground truth bounding boxes
        # note that the output shape will be (num default truth boxes with iou > threshold with at least one ground truth box, 2)
        # this matrix-like tensor contains indexes for default boxes, ground truth boxes
        indexes_match_default = tf.stack([tf.range(len(self.xmin_boxes_default)), tf.math.argmax(iou, axis=1, output_type=tf.dtypes.int32)], axis=1)
        indexes_match_default = tf.boolean_mask(tensor=indexes_match_default, mask=tf.math.greater(tf.reduce_max(iou, axis=1), self.iou_threshold), axis=0)

        # step 3 - put all the best matches together, removing possible duplicates
        indexes_match, _ = tf.raw_ops.UniqueV2(x=tf.concat([indexes_match_ground_truth, indexes_match_default], axis=0), axis=[0])

        # get the class labels for each best match and one-hot encode them (0 is reserved for the background class)
        labels_match = tf.gather(labels_ground_truth, indexes_match[:, 1])
        labels_match = tf.one_hot(labels_match, self.num_classes)

        # convert all bounding boxes coordinates from corners to centroids
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

        # calculate centroids offsets between ground truth and default boxes and standardize them
        # note that for standardization we are assuming that the mean of the offsets it's zero
        # offsets standard deviation it's given as input
        offsets_center_x = (centroids_ground_truth_center_x - centroids_default_center_x) / centroids_default_width / self.offsets_center_x_std
        offsets_center_y = (centroids_ground_truth_center_y - centroids_default_center_y) / centroids_default_height / self.offsets_center_y_std
        offsets_width = tf.math.log(centroids_ground_truth_width / centroids_default_width) / self.offsets_width_std
        offsets_height = tf.math.log(centroids_ground_truth_height / centroids_default_height) / self.offsets_height_std
        
        # default bounding boxes encoded as required (one-hot encoding for classes, offsets for centroid coordinates)
        # if a default bounding box was matched with ground truth, then labels and offsets centroids coordinates are calculated
        # otherwise the default bounding box has background label and offsets centroid coordinates equal to zero        
        boxes_encoded = tf.zeros(shape=(len(self.xmin_boxes_default), self.num_classes + 4), dtype=tf.float32)
        boxes_encoded = tf.tensor_scatter_nd_update(
            tensor=boxes_encoded,
            indices=tf.expand_dims(indexes_match[:, 0], axis=1),
            updates=tf.concat(
                values=[
                    labels_match,
                    tf.expand_dims(offsets_center_x, axis=1),
                    tf.expand_dims(offsets_center_y, axis=1),
                    tf.expand_dims(offsets_height, axis=1),
                    tf.expand_dims(offsets_width, axis=1)
                ],
                axis=1)
        )
    
        return boxes_encoded

    def read_data(
            self,
            path_image: str,
            path_mask: str,
            path_labels_boxes: str,
        ):

        # encode ground truth bounding boxes
        encoded_boxes = self._encode_ground_truth_bounding_boxes(path_labels_boxes=path_labels_boxes)

        # read the image, resize and scale value between 0 and 1
        image = tf.io.read_file(path_image)
        image = tf.image.decode_png(image, channels=3)
        image = tf.cast(image, dtype=tf.float32) / 255

        # read the mask, ignoring transparency channel in the png, resize, one hot encode the 4 classes, squeeze out unwanted dimension
        mask = tf.io.read_file(path_mask)
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.one_hot(mask, depth=4, dtype=tf.float32)
        mask = tf.squeeze(mask, axis=2)

        return image, encoded_boxes, mask

dataReaderEncoder = DataReaderEncoder(
    num_classes=4,
    boxes_default=boxes_default,
    iou_threshold=0.5,
    offsets_std=(0.1, 0.1, 0.2, 0.2),
)

img, boxes, mask = dataReaderEncoder.read_data('data/train/2335.png', 'data/train/2335_mask.png', 'data/train/2335_labels_boxes.csv')

# load metadata
with open('data/train.json', 'r') as f:
    path_images_train, path_masks_train, path_labels_boxes_train = map(list, zip(*json.load(f)))

# tensorflow train dataset pipeline
ds_train = (
    tf.data.Dataset.from_tensor_slices((path_images_train, path_masks_train, path_labels_boxes_train))
    .shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
    .map(dataReaderEncoder.read_data, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size=BATCH_SIZE)
    #.map(dataAugmentation)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

s = 0

# check if images are loaded fine
for image_batch, mask_batch, boxes_encoded_batch in ds_train.take(1):
    k = 0

