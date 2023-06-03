import json
import tensorflow as tf
from matplotlib import pyplot as plt, patches
import random

# gpus = tf.config.list_physical_devices('GPU')
# if gpus: 
#     tf.config.set_logical_device_configuration(
#         gpus[0],
#         [tf.config.LogicalDeviceConfiguration(memory_limit=7800)]
#     )

# logical_gpus = tf.config.list_logical_devices('GPU')
# print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")

with open('data/train.json', 'r') as f:
    data = json.load(f)

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

for path_image, path_mask, labels in random.sample(data, 10):
    image = tf.io.read_file(path_image)
    image = tf.image.decode_png(image, channels=3)
    image = tf.cast(image, dtype=tf.float32) / 255.0

    # display the image
    plt.imshow(image, vmin=0, vmax=1)
    plt.axis('off')

    # get the current plot object
    ax = plt.gca()

    # plot bounding boxes
    for label, xmin, ymin, xmax, ymax, cx, cy, w, h in labels:
        rect = patches.Rectangle((xmin, ymin), w, h, linewidth=1, edgecolor=label_code_to_color[label], facecolor='none')
        ax.add_patch(rect)
        plt.text(xmin, ymin, label_code_to_str[label], fontsize=8, color=label_code_to_color[label], verticalalignment='top')

    plt.show()
