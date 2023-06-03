from glob import iglob
from pathlib import Path
import os
import json

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
        labels = []

        # convert segmentation polygons to bounding boxes coordinates
        for polygon in polygons['objects']:
            
            # skip background polygons
            if polygon['label'] == 'background':
                continue

            # get all x and y coordinates for the polygon
            x, y = zip(*polygon['polygon'])

            # add to labels list in the following order
            # label, x_min, y_min, x_max, y_max, x_center, y_center, width, height
            labels.append([
                label_str_to_code[polygon['label']],
                min(x),
                min(y),
                max(x),
                max(y),
                (min(x) + max(x)) / 2,
                (min(y) + max(y)) / 2,
                max(x) - min(x),
                max(y) - min(y),
            ])

        # append to data list
        data.append([
            f'data/{folder}/{file}',
            f'data/{folder}/{file.replace(".png", "_mask.png")}',
            labels
        ])

        # move
        os.rename(f'data/images/{folder}/{file}', f'data/{folder}/{file}')
        os.rename(f'data/labels/{folder}/{file.replace(".png", "_mask.png")}', f'data/{folder}/{file.replace(".png", "_mask.png")}')
        os.rename(f'data/labels/{folder}/{file.replace(".png", ".json")}', f'data/{folder}/{file.replace(".png", "_polygons.json")}')

    # write auxiliary file
    with open(f'data/{folder}.json', 'w') as f:
        f.write(json.dumps(data, indent=4, separators=(',', ':')))
