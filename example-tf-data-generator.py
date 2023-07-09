import random
import json
import numpy as np
from matplotlib import pyplot as plt
import ssd
import plot
from PIL import Image

# global variables
SEED = 1993

# load metadata
with open('data/train.json', 'r') as f:
    json_train = json.load(f)

with open('data/eval.json', 'r') as f:
    json_eval = json.load(f)

with open('data/test.json', 'r') as f:
    json_test = json.load(f)

class DataReaderEncoder:
    def __init__(
            self,
            image_shape: tuple[int],
            num_classes: int,
            feature_maps_shapes: tuple[tuple[int]],
            feature_maps_aspect_ratios: tuple[float] | tuple[tuple[float]] = (1.0, 2.0, 3.0, 1/2, 1/3),
            centers_padding_from_borders: float = 0.5,
            boxes_scales: tuple[float] | tuple[tuple[float]] = (0.2, 0.9),
            additional_square_box: bool = True,
            iou_threshold: float = 0.5,
            boxes_centroids_offsets_std: tuple[float] = (0.1, 0.1, 0.2, 0.2)
        ) -> None:

        # initialize args
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.boxes_centroids_offsets_std = np.array(boxes_centroids_offsets_std)

        # generate default bounding boxes
        self.boxes_default = ssd.generate_default_bounding_boxes(
            feature_maps_shapes=feature_maps_shapes,
            feature_maps_aspect_ratios=feature_maps_aspect_ratios,
            centers_padding_from_borders=centers_padding_from_borders,
            boxes_scales=boxes_scales,
            additional_square_box=additional_square_box,
        )

        # default bounding boxes, reshaped as output from the network
        self.boxes_default = np.concatenate([np.reshape(boxes_default, newshape=(-1, 4)) for boxes_default in self.boxes_default], axis=0)

        # default bounding boxes, scaled at image shape
        self.boxes_default[:, [0, 2]] = self.boxes_default[:, [0, 2]] * image_shape[1]
        self.boxes_default[:, [1, 3]] = self.boxes_default[:, [1, 3]] * image_shape[0]

        # corners coordinates for each default bounding box
        self.xmin_boxes_default, self.ymin_boxes_default, self.xmax_boxes_default, self.ymax_boxes_default = np.split(self.boxes_default, 4, axis=-1)

        # calculate area for each default bounding box
        self.area_boxes_default = (self.ymax_boxes_default - self.ymin_boxes_default + 1) * (self.xmax_boxes_default - self.xmin_boxes_default + 1)

    def read_encode_data(
            self,
            path_image: str,
            path_mask: str,
            labels_ground_truth: list[int],
            boxes_ground_truth: list[float],
        ):
        
        # convert to numpy array
        labels_ground_truth = np.array(labels_ground_truth)
        boxes_ground_truth = np.array(boxes_ground_truth)

        # corners coordinates for ground truth bounding boxes
        xmin_boxes_ground_truth, ymin_boxes_ground_truth, xmax_boxes_ground_truth, ymax_boxes_ground_truth = np.split(boxes_ground_truth, 4, axis=-1)

        # calculate area of each bounding box
        area_boxes_ground_truth = (xmax_boxes_ground_truth - xmin_boxes_ground_truth + 1) * (ymax_boxes_ground_truth - ymin_boxes_ground_truth + 1)

        # coordinates of intersections between each default bounding box and all ground truth bounding boxes
        # it select the maximum for xmin and ymin coordinates, the minimum for xmax and ymax coordinates
        xmin_boxes_intersection = np.maximum(self.xmin_boxes_default, xmin_boxes_ground_truth.T)
        ymin_boxes_intersection = np.maximum(self.ymin_boxes_default, ymin_boxes_ground_truth.T)
        xmax_boxes_intersection = np.minimum(self.xmax_boxes_default, xmax_boxes_ground_truth.T)
        ymax_boxes_intersection = np.minimum(self.ymax_boxes_default, ymax_boxes_ground_truth.T)

        # area of intersection between each default bounding box and all ground truth bounding boxes
        area_boxes_intersection = np.maximum(0, xmax_boxes_intersection - xmin_boxes_intersection + 1) * np.maximum(0, ymax_boxes_intersection - ymin_boxes_intersection + 1)

        # calculate intersection over union between each default bounding box and all ground truth bounding boxes
        # note that this is a matrix with shape (num default bounding boxes, num ground truth bounding boxes)
        iou = area_boxes_intersection / (self.area_boxes_default + area_boxes_ground_truth.T - area_boxes_intersection)

        # find best match between each ground truth box and all default bounding boxes
        # note that output shape will be (num ground truth boxes with iou > 0 with at least one default box, 2)
        # the last axis contains indexes for default boxes, ground truth boxes
        indexes_match_ground_truth = np.column_stack([iou.argmax(axis=0), np.arange(len(boxes_ground_truth))])[iou.max(axis=0) > 0]

        # find best match between each default box and all ground truth bounding boxes
        # note that output shape will be (num default truth boxes with iou > threshold with at least one ground truth box, 2)
        # the last axis contains indexes for default boxes, ground truth boxes
        indexes_match_default = np.column_stack([np.arange(len(self.boxes_default)), iou.argmax(axis=1)])[iou.max(axis=1) > self.iou_threshold]

        # put best matches together, removing possible pair duplicates
        indexes_match = np.unique(np.vstack([indexes_match_ground_truth, indexes_match_default]), axis=0)

        # get the class labels for each best match and one hot encode them (0 it's reserved for background class)
        labels_match = labels_ground_truth[indexes_match[:, 1]]
        labels_match = np.eye(self.num_classes)[labels_match]

        # convert boxes coordinates from corners to centroids
        centroids_default = ssd.bounding_boxes_corners_to_centroids(boxes=self.boxes_default[indexes_match[:, 0]])
        centroids_ground_truth = ssd.bounding_boxes_corners_to_centroids(boxes=boxes_ground_truth[indexes_match[:, 1]])

        # default bounding boxes encoded as required (one hot encoding for classes, offsets for centroids coordinates)
        boxes_encoded = np.zeros(shape=(len(self.boxes_default), self.num_classes + 4))
        boxes_encoded[indexes_match[:, 0], :self.num_classes] = labels_match
        boxes_encoded[indexes_match[:, 0], -4:-2] = (centroids_ground_truth[:, [0, 1]] - centroids_default[:, [0, 1]]) / centroids_default[:, [2, 3]] / self.boxes_centroids_offsets_std[:2]
        boxes_encoded[indexes_match[:, 0], -2:] = np.log(centroids_ground_truth[:, [2, 3]] / centroids_default[:, [2, 3]]) / self.boxes_centroids_offsets_std[-2:]

        # read image, scale values between 0 and 1
        image = Image.open(path_image)
        image = np.array(image, dtype=np.float32)/ 255.0

        # read mask, one-hot encode the 4 classes, squeeze out unwanted dimension
        mask = Image.open(path_mask)
        mask = np.array(mask)
        mask = np.eye(4, dtype=np.float32)[mask]

        # return image, boxes encoded, mask
        return image, mask, boxes_encoded

dataReaderEncoder = DataReaderEncoder(
    image_shape=(480, 640),
    num_classes=4,
    feature_maps_shapes=((24, 32), (12, 16), (6, 8), (3, 4)),
    feature_maps_aspect_ratios=(1.0, 2.0, 3.0, 1/2, 1/3),
    centers_padding_from_borders=0.5,
    boxes_scales=(0.2, 0.9),
    additional_square_box=True,
    iou_threshold=0.5,
    boxes_centroids_offsets_std=(0.1, 0.1, 0.2, 0.2),
)

# sample training data
for path_image, path_mask, labels_ground_truth, boxes_ground_truth in random.sample(json_train, k=5):
    
    # read and encode data
    image, mask, boxes_encoded = dataReaderEncoder.read_encode_data(
        path_image=path_image,
        path_mask=path_mask,
        labels_ground_truth=labels_ground_truth,
        boxes_ground_truth=boxes_ground_truth
    )

    k = 0