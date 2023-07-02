import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import ssd

labels_ground_truth = np.array([3, 2, 2])

boxes_ground_truth = np.array([
    [3, 4, 11, 13],
    [41, 5, 46, 10],
    [20, 13, 29, 28],
])

boxes_default = np.array([
    [3, 26, 11, 32],
    [5, 25, 9, 33],
    [6, 28, 8, 30],
    
    [4, 7, 12, 13],
    [6, 6, 10, 14],
    [7, 9, 9, 11],
    
    [16, 10, 24, 16],
    [18, 9, 22, 17],
    [19, 12, 21, 14],
    
    [21, 19, 29, 25],
    [23, 18, 27, 26],
    [24, 21, 26, 23],
    
    [28, 8, 36, 14],
    [30, 7, 34, 15],
    [31, 10, 33, 13],
])

# args
iou_threshold = 0.5
num_classes = 4

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
boxes_to_plot = boxes_default[indexes_match[:,0]].tolist()

# create a figure and axis
fig, ax = plt.subplots()

# set the maximum x and y limits of the plot
ax.set_xlim([0, 47])
ax.set_ylim([0, 37])

# plot the ground truth boxes
for box in boxes_ground_truth:
    xmin, ymin, xmax, ymax = box
    rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

# plot the default boxes
for box in boxes_to_plot:
    xmin, ymin, xmax, ymax = box
    rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='b', facecolor='none')
    ax.add_patch(rect)

# show the plot
plt.show()
