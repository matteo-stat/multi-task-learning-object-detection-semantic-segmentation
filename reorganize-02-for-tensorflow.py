import json
from pathlib import Path

# reformat metadata
for folder in ['eval', 'test', 'train']:

    # open json data
    with open(f'data/{folder}.json', 'r') as f:
        data_json = json.load(f)

    # reformat path for images and masks
    data_path_images_masks = []

    # write a single file with labels and boxes for each image
    for path_image, path_mask, labels, boxes in data_json:

        data_path_images_masks.append([path_image, path_mask, f'data/{folder}/{Path(path_image).stem}_labels_boxes.json'])
        data_labels_boxes = {'labels': labels, 'boxes': boxes}
        
        with open(f'data/{folder}/{Path(path_image).stem}_labels_boxes.json', 'w') as f:
            f.write(json.dumps(data_labels_boxes, indent=4, separators=(',', ':')))
    
    # write a whole file for 
    with open(f'data/{folder}_path_images_masks.json', 'w') as f:
        f.write(json.dumps(data_path_images_masks, indent=4, separators=(',', ':')))








with open('data/eval.json', 'r') as f:
    json_eval = json.load(f)

with open('data/test.json', 'r') as f:
    json_test = json.load(f)