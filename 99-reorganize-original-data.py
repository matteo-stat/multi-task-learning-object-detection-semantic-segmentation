from glob import iglob
from pathlib import Path
import os
import json
import csv
import numpy as np

# define folders to loop through
folders = ['train', 'eval', 'test']

# remove useless png files
for folder in folders:
    for file in iglob(f'data/labels/{folder}/*.png'):
        if not file.lower().endswith(('_id.png', '_mask.png')):
            os.remove(file)
        else:
            os.rename(file, file.replace('_id.png', '_mask.png'))

# convert labels string to codes (0 reserved for background)
label_str_to_code = {
    'monorail': 1,
    'person': 2,
    'forklift1': 3,
    'forklift2': 3
}


def merge_boxes_overlapping(boxes_xmin, boxes_ymin, boxes_xmax, boxes_ymax):
    # Initialize a list to store the boxes
    boxes = list(zip(boxes_xmin, boxes_ymin, boxes_xmax, boxes_ymax))
    
    def calculate_iou(box1, box2):
        # Calculate the intersection area
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Calculate the union area
        area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = area_box1 + area_box2 - intersection_area
        
        # Calculate the IoU
        iou = intersection_area / union_area
        
        return iou
    
    while len(boxes) > 1:
        max_iou = 0
        merge_indices = None
        
        # Find the two boxes with the highest IoU
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                iou = calculate_iou(boxes[i], boxes[j])
                if iou > max_iou:
                    max_iou = iou
                    merge_indices = (i, j)
        
        if max_iou == 0:
            break
        
        # Merge the two boxes with the highest IoU
        box1 = boxes[merge_indices[0]]
        box2 = boxes[merge_indices[1]]
        merged_box = [min(box1[0], box2[0]), min(box1[1], box2[1]),
                      max(box1[2], box2[2]), max(box1[3], box2[3])]
        
        # Remove the merged boxes and add the new merged_box
        boxes.pop(merge_indices[1])
        boxes.pop(merge_indices[0])
        boxes.append(merged_box)

    return np.split(np.array(boxes), 4, axis=-1)

def calculate_distance(box1, box2):
    # Calculate the distance between the boxes using the described method
    y_distance = min(abs(box1[1] - box2[3]), abs(box2[1] - box1[3]))
    x_distance = min(abs(box1[0] - box2[2]), abs(box2[0] - box1[2]))
    return y_distance, x_distance

def merge_boxes_too_close(boxes, threshold):
    while True:
        min_distance = float('inf')
        merge_indices = None

        # Calculate the distance between all pairs of boxes
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                y_distance, x_distance = calculate_distance(boxes[i], boxes[j])
                distance = min(y_distance, x_distance)
                if distance < min_distance:
                    min_distance = distance
                    merge_indices = (i, j)

        # Check if the minimum distance is below the threshold
        if min_distance <= threshold:
            # Merge the two closest boxes
            i, j = merge_indices
            merged_box = [
                min(boxes[i][0], boxes[j][0]),
                min(boxes[i][1], boxes[j][1]),
                max(boxes[i][2], boxes[j][2]),
                max(boxes[i][3], boxes[j][3])
            ]
            # Remove the original boxes and add the merged box
            boxes.pop(max(i, j))
            boxes.pop(min(i, j))
            boxes.append(merged_box)
        else:
            break

    return boxes


# calculate bounding boxes from segmentation polygons
for folder in folders:
    
    # new auxiliary dataset file
    data = []

    for file in iglob(f'data/images/{folder}/*.png'):
        file = Path(file).name

        # read polygons file
        with open(f'data/labels/{folder}/{file.replace(".png", ".json")}', 'r') as f:
            polygons = json.load(f)

        # labels list for the given image
        labels_boxes = []

        # convert segmentation polygons to bounding boxes coordinates
        for polygon in polygons['objects']:
            
            # skip background polygons
            if polygon['label'] == 'background':
                continue

            # get all x and y coordinates for the polygon
            x, y = zip(*polygon['polygon'])

            # add to labels list in the following order
            # label, x_min, y_min, x_max, y_max, x_center, y_center, width, height
            labels_boxes.append([label_str_to_code[polygon['label']], min(x), min(y), max(x), max(y)])

        # check rails data
        labels_boxes_not_rails = [content for content in labels_boxes if content[0] != 1]
        labels_boxes_rails = [content for content in labels_boxes if content[0] == 1]

        if len(labels_boxes_rails) > 0:
            labels_boxes_rails = np.array(labels_boxes_rails)
            xmin = labels_boxes_rails[:, 1]
            ymin = labels_boxes_rails[:, 2]
            xmax = labels_boxes_rails[:, 3]
            ymax = labels_boxes_rails[:, 4]

            xmin, ymin, xmax, ymax = merge_boxes_overlapping(xmin, ymin, xmax, ymax)

            boxes = merge_boxes_too_close(
                boxes=np.concatenate([xmin, ymin, xmax, ymax], axis=-1).tolist(),
                threshold=5
            )
            xmin, ymin, xmax, ymax = np.split(np.array(boxes), 4, -1)

            to_keep = np.where(((xmax - xmin + 1) * (ymax - ymin + 1)) / (480*640) > 1/1000)
            xmin = xmin[to_keep]
            ymin = ymin[to_keep]
            xmax = xmax[to_keep]
            ymax = ymax[to_keep]

            xmin, ymin, xmax, ymax = merge_boxes_overlapping(xmin, ymin, xmax, ymax)

            s = 0

            labels_boxes_rails = [
                [1, float(_xmin), float(_ymin), float(_xmax), float(_ymax)]
                for _xmin, _ymin, _xmax, _ymax
                in zip(xmin, ymin, xmax, ymax)
            ]

            labels_boxes = labels_boxes_not_rails + labels_boxes_rails

        if len(labels_boxes) == 0:
            print(f'error, no labels for {file}')

        # append to data list
        data.append([
            f'data/{folder}/{file}',
            f'data/{folder}/{file.replace(".png", "_mask.png")}',
            f'data/{folder}/{file.replace(".png", "_labels_boxes.csv")}'
        ])

        # write labels and boxes
        with open(f'data/{folder}/{file.replace(".png", "_labels_boxes.csv")}', 'w', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerows(labels_boxes)

        # move
        os.rename(f'data/images/{folder}/{file}', f'data/{folder}/{file}')
        os.rename(f'data/labels/{folder}/{file.replace(".png", "_mask.png")}', f'data/{folder}/{file.replace(".png", "_mask.png")}')
        os.rename(f'data/labels/{folder}/{file.replace(".png", ".json")}', f'data/{folder}/{file.replace(".png", "_polygons.json")}')

    # write auxiliary file
    with open(f'data/{folder}.json', 'w') as f:
        f.write(json.dumps(data, indent=4, separators=(',', ':')))
