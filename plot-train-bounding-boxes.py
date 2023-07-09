import plot
import random
import json
import numpy as np
from matplotlib import pyplot as plt, patches, get_backend
from PIL import Image

# labels conversions
label_code_to_str = {
    1: 'monorail',
    2: 'person',
    3: 'forklift'
}
label_code_to_color = {
    1: 'red',
    2: 'green',
    3: 'blue'
}

# read training data
with open('data/train.json', 'r') as f:
    data = json.load(f)

# sample training data
for path_image, path_mask, labels, boxes in random.sample(data, k=5):
    
    # read image
    image = Image.open(path_image)
    image = np.array(image)
    image = image.astype(np.float32) / 255.0

    # create the plot
    fig = plt.figure(figsize=(8, 6))
    plot.move_figure(fig=fig, x=0, y=0)

    # display the image
    plt.imshow(image, vmin=0, vmax=1)
    plt.axis('off')

    # get the current plot object
    ax = plt.gca()

    # plot bounding boxes
    for label, (xmin, ymin, xmax, ymax) in zip(labels, boxes):
        rect = patches.Rectangle((xmin, ymin), xmax - xmin + 1, ymax - ymin + 1, linewidth=1, edgecolor=label_code_to_color[label], facecolor='none')
        ax.add_patch(rect)
        plt.text(xmin, ymin, label_code_to_str[label], fontsize=8, color=label_code_to_color[label], verticalalignment='top')

    # show the plot
    plt.show()
