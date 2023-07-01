import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

gt_boxes = [
    [3, 4, 11, 13],
    [41, 5, 46, 10],
    [20, 13, 29, 28],
]

def_boxes = [
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
]

# default bounding boxes and ground truth bounding boxes
boxes_default = np.array(def_boxes)
boxes_ground_truth = np.array(gt_boxes)

# min max coordinates for default an ground truth bounding boxes
xmin_boxes_default, ymin_boxes_default, xmax_boxes_default, ymax_boxes_default = np.split(boxes_default, 4, axis=-1)
xmin_boxes_ground_truth, ymin_boxes_ground_truth, xmax_boxes_ground_truth, ymax_boxes_ground_truth = np.split(boxes_ground_truth, 4, axis=-1)

# calculate area of each bounding box
area_boxes_ground_truth = (xmax_boxes_ground_truth - xmin_boxes_ground_truth + 1) * (ymax_boxes_ground_truth - ymin_boxes_ground_truth + 1)
area_boxes_default = (ymax_boxes_default - ymin_boxes_default + 1) * (xmax_boxes_default - xmin_boxes_default + 1)

# coordinates of intersections between each default bounding box and all ground truth bounding boxes
# it select the maximum for xmin and ymin coordinates, the minimum for xmax and ymax coordinates
xmin_boxes_intersections = np.maximum(xmin_boxes_default, xmin_boxes_ground_truth.T)
ymin_boxes_intersections = np.maximum(ymin_boxes_default, ymin_boxes_ground_truth.T)
xmax_boxes_intersections = np.minimum(xmax_boxes_default, xmax_boxes_ground_truth.T)
ymax_boxes_intersections = np.minimum(ymax_boxes_default, ymax_boxes_ground_truth.T)

# area of intersection between each default bounding box and all ground truth bounding boxes
area_intersection = np.maximum(0, xmax_boxes_intersections - xmin_boxes_intersections + 1) * np.maximum(0, ymax_boxes_intersections - ymin_boxes_intersections + 1)

# area of union between each default bounding box and all ground truth bounding boxes
area_union = area_boxes_default + area_boxes_ground_truth.T - area_intersection

# calculate intersection over union betwee
iou = area_intersection / area_union

# create a figure and axis
fig, ax = plt.subplots()

# set the maximum x and y limits of the plot
ax.set_xlim([0, 47])
ax.set_ylim([0, 37])

# plot the ground truth boxes
for box in gt_boxes:
    xmin, ymin, xmax, ymax = box
    rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

# plot the default boxes
for box in def_boxes:
    xmin, ymin, xmax, ymax = box
    rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='b', facecolor='none')
    ax.add_patch(rect)

# show the plot
plt.show()
