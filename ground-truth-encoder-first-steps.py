import ssd
import random
import json
import numpy as np
from matplotlib import pyplot as plt, patches

# read training data
with open('data/train.json', 'r') as f:
    data = json.load(f)

# image shape
image_shape = (480, 640)

# feature maps shapes
feature_maps_shapes = (
    (24, 32),
    (12, 16),
    (6, 8),
    (3, 4)
)  

# default bounding boxes, organized by feature map
feature_maps_boxes = ssd.generate_default_bounding_boxes(
    feature_maps_shapes=feature_maps_shapes,
    boxes_scales=(0.1, 0.6),
    additional_square_box=True
)

# default bounding boxes, organized as output from the network
boxes_default = np.concatenate([np.reshape(feature_map_boxes, newshape=(-1, 4)) for feature_map_boxes in feature_maps_boxes], axis=0)

# default bounding boxes, scaled at image shape
boxes_default[:, [0, 2]] = boxes_default[:, [0, 2]] * image_shape[1]
boxes_default[:, [1, 3]] = boxes_default[:, [1, 3]] * image_shape[0]

# args
iou_threshold = 0.5
num_classes = 4

# sample training data
for path_image, path_mask, labels_ground_truth, boxes_ground_truth in random.sample(data, 5):

    # convert to np.array
    labels_ground_truth = np.array(labels_ground_truth)
    boxes_ground_truth = np.array(boxes_ground_truth)

    if len(labels_ground_truth) == 1:
        continue
    
    # corners coordinates for default an ground truth bounding boxes
    xmin_boxes_default, ymin_boxes_default, xmax_boxes_default, ymax_boxes_default = np.split(boxes_default, 4, axis=-1)
    xmin_boxes_ground_truth, ymin_boxes_ground_truth, xmax_boxes_ground_truth, ymax_boxes_ground_truth = np.split(boxes_ground_truth, 4, axis=-1)

    # calculate area of each bounding box
    area_boxes_ground_truth = (xmax_boxes_ground_truth - xmin_boxes_ground_truth + 1) * (ymax_boxes_ground_truth - ymin_boxes_ground_truth + 1)
    area_boxes_default = (ymax_boxes_default - ymin_boxes_default + 1) * (xmax_boxes_default - xmin_boxes_default + 1)

    # coordinates of intersections between each default bounding box and all ground truth bounding boxes
    # it select the maximum for xmin and ymin coordinates, the minimum for xmax and ymax coordinates
    xmin_boxes_intersection = np.maximum(xmin_boxes_default, xmin_boxes_ground_truth.T)
    ymin_boxes_intersection = np.maximum(ymin_boxes_default, ymin_boxes_ground_truth.T)
    xmax_boxes_intersection = np.minimum(xmax_boxes_default, xmax_boxes_ground_truth.T)
    ymax_boxes_intersection = np.minimum(ymax_boxes_default, ymax_boxes_ground_truth.T)

    # area of intersection between each default bounding box and all ground truth bounding boxes
    area_boxes_intersection = np.maximum(0, xmax_boxes_intersection - xmin_boxes_intersection + 1) * np.maximum(0, ymax_boxes_intersection - ymin_boxes_intersection + 1)

    # calculate intersection over union between each default bounding box and all ground truth bounding boxes
    # note that this it's a matrix with shape (num default bounding boxes, num ground truth bounding boxes)
    iou = area_boxes_intersection / (area_boxes_default + area_boxes_ground_truth.T - area_boxes_intersection)

    # find best match between each ground truth box and all default bounding boxes
    # note that output shape will be (num ground truth boxes with iou > 0 with at least one default box, 2)
    # the last axis contains indexes for default boxes, ground truth boxes
    indexes_match_ground_truth = np.column_stack([iou.argmax(axis=0), np.arange(len(boxes_ground_truth))])[iou.max(axis=0) > 0]

    # find best match between each default box and all ground truth bounding boxes
    # note that output shape will be (num default truth boxes with iou > threshold with at least one ground truth box, 2)
    # the last axis contains indexes for default boxes, ground truth boxes
    indexes_match_default = np.column_stack([np.arange(len(boxes_default)), iou.argmax(axis=1)])[iou.max(axis=1) > iou_threshold]

    # put best matches together, removing possible pair duplicates
    indexes_match = np.unique(np.vstack([indexes_match_ground_truth, indexes_match_default]), axis=0)

    # get the class labels for each best match and one hot encode them (0 it's reserved for background class)
    labels_match = labels_ground_truth[indexes_match[:, 1]]
    labels_match = np.eye(num_classes)[labels_match]

    # convert boxes coordinates from corners to centroids
    centroids_default = ssd.bounding_boxes_corners_to_centroids(boxes=boxes_default[indexes_match[:, 0]])
    centroids_ground_truth = ssd.bounding_boxes_corners_to_centroids(boxes=boxes_ground_truth[indexes_match[:, 1]])

    # encode data as required (one hot encoding for classes, offsets for centroids coordinates)
    data_encoded = np.zeros(shape=(len(boxes_default), num_classes + 4))
    data_encoded[indexes_match[:,0], :num_classes] = labels_match
    data_encoded[indexes_match[:,0], -4:-2] = (centroids_ground_truth[:, [0, 1]] - centroids_default[:, [0, 1]]) / centroids_default[:, [2, 3]]
    data_encoded[indexes_match[:,0], -2:] = np.log(centroids_ground_truth[:, [2, 3]] / centroids_default[:, [2, 3]])

    # keep only best matches
    boxes_to_plot = boxes_default[indexes_match[:, 0]].tolist()

    # create a figure and axis
    fig, ax = plt.subplots()

    # set the maximum x and y limits of the plot
    ax.set_xlim([0, image_shape[1]])
    ax.set_ylim([0, image_shape[0]])

    # plot the default boxes
    for box in boxes_to_plot:
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='b', facecolor='none')
        ax.add_patch(rect)

    # plot the ground truth boxes
    for box in boxes_ground_truth:
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    # show the plot
    plt.show()    