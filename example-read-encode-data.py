import ssdseglib
import json
import random
import csv
import numpy as np
from matplotlib import pyplot as plt, patches
from pathlib import Path
from PIL import Image
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf

# global variables
INPUT_IMAGE_SHAPE = (480, 640)
SHUFFLE_BUFFER_SIZE = 512
BATCH_SIZE = 8
SEED = 1993

# plot options
fig_size_width = 11

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

# create default bounding boxes
boxes_default = ssdseglib.boxes.DefaultBoundingBoxes(
    feature_maps_shapes=((30, 40), (15, 20), (8, 10), (4, 5)),
    centers_padding_from_borders_percentage=0.025,
    additional_square_box=False,  
)
boxes_default.rescale_boxes_coordinates(image_shape=INPUT_IMAGE_SHAPE)

# create a data reader encoder
data_reader_encoder = ssdseglib.datacoder.DataEncoderDecoder(
    num_classes=4,
    image_shape=INPUT_IMAGE_SHAPE,
    xmin_boxes_default=boxes_default.get_boxes_coordinates_xmin(coordinates_style='ssd'),
    ymin_boxes_default=boxes_default.get_boxes_coordinates_ymin(coordinates_style='ssd'),
    xmax_boxes_default=boxes_default.get_boxes_coordinates_xmax(coordinates_style='ssd'),
    ymax_boxes_default=boxes_default.get_boxes_coordinates_ymax(coordinates_style='ssd'),
    iou_threshold=0.5,
    std_offsets=(0.1, 0.1, 0.2, 0.2),
    augmentation_horizontal_flip=True
)

# load metadata
with open('data/train.json', 'r') as f:
    path_images_train, path_masks_train, path_labels_boxes_train = map(list, zip(*json.load(f)))

# just for debugging
# image, targets = data_reader_encoder.read_and_encode(path_images_train[0], path_masks_train[0], path_labels_boxes_train[0])

# sample some data randomly
random_sample = random.sample(range(len(path_images_train)), k=100)
path_images_train = np.array(path_images_train)[random_sample]
path_masks_train = np.array(path_masks_train)[random_sample]
path_labels_boxes_train = np.array(path_labels_boxes_train)[random_sample]

# tensorflow train dataset pipeline
ds_train = (
    tf.data.Dataset.from_tensor_slices((path_images_train, path_masks_train, path_labels_boxes_train))
    .map(data_reader_encoder.read_and_encode, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size=BATCH_SIZE)
    .map(ssdseglib.datacoder.augmentation_rgb_channels, num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# sample counter (used for plot ground truth bounding boxes)
i = 0

# check if images are loaded fine
for image_batch, targets_batch in ds_train.take(1):
    for image_sample, mask_sample, labels_sample, boxes_sample in zip(image_batch, targets_batch['output-mask'], targets_batch['output-labels'], targets_batch['output-boxes']):
        
        # read labels boxes
        with open(path_labels_boxes_train[i], 'r') as f:
            labels_boxes = list(csv.reader(f))

        # create subplots and set figure size
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3,  constrained_layout=True)        
        fig.set_size_inches(fig_size_width, int(fig_size_width / (INPUT_IMAGE_SHAPE[1] / INPUT_IMAGE_SHAPE[0])))
        ssdseglib.plot.move_figure(fig=fig, x=0, y=0)
        
        # ground truth subplot
        image = Image.open(path_images_train[i])
        image = np.array(image)
        image = image.astype(np.int32)
        ax1.set_aspect('equal')
        ax1.imshow(image, vmin=0, vmax=255)
        ax1.set_axis_off()
        ax1.set_title(f'ground truth ({Path(path_images_train[i]).name})')

        # plot ground truth bounding boxes
        for label, xmin, ymin, xmax, ymax in labels_boxes:
            label = int(label)
            xmin = float(xmin)
            ymin = float(ymin)
            xmax = float(xmax)
            ymax = float(ymax)        
            rect = patches.Rectangle((xmin, ymin), xmax - xmin + 1, ymax - ymin + 1, linewidth=1, edgecolor=label_code_to_color[label], facecolor='none')
            ax1.add_patch(rect)
            ax1.text(xmin, ymin, label_code_to_str[label], fontsize=8, color=label_code_to_color[label], verticalalignment='top')        

        # encoded mask subplot (drop the background channel)
        ax2.set_aspect('equal')
        mask_real = tf.slice(mask_sample, begin=[0, 0, 1], size=[-1, -1, 3])
        ax2.imshow(mask_real, vmin=0.0, vmax=1.0)
        ax2.set_axis_off()
        ax2.set_title('encoded mask')

        # boxes subplot
        # keep only valid default bounding boxes (boxes that were matched to ground truth data)
        # use the background class one hot information
        valid_samples = tf.math.equal(labels_sample[:, 0], 0.0)

        # keep valid default bounding boxes samples
        decoded_corners = data_reader_encoder.decode_to_corners(offsets_centroids=boxes_sample)
        valid_boxes_sample = tf.boolean_mask(
            tensor=decoded_corners,
            mask=valid_samples,
            axis=0
        )

        # keep valid default bounding boxes labels
        valid_labels_sample = tf.boolean_mask(
            tensor=labels_sample,
            mask=valid_samples,
            axis=0
        )

        ax3.set_aspect('equal')
        ax3.imshow(tf.cast(image_sample, tf.int32), vmin=0, vmax=255)
        ax3.set_axis_off()
        ax3.set_title(f'encoded boxes ({valid_boxes_sample.shape[0]} boxes)')

        # plot bounding boxes
        for labels_one_hot_encoded, (xmin, ymin, xmax, ymax) in zip(valid_labels_sample, valid_boxes_sample):
            label = int(tf.argmax(labels_one_hot_encoded))
            xmin = float(xmin)
            ymin = float(ymin)
            xmax = float(xmax)
            ymax = float(ymax)            
            rect = patches.Rectangle((xmin, ymin), xmax - xmin + 1, ymax - ymin + 1, linewidth=1, edgecolor=label_code_to_color[label], facecolor='none')
            ax3.add_patch(rect)
            ax3.text(xmin, ymin, label_code_to_str[label], fontsize=8, color=label_code_to_color[label], verticalalignment='top')

        # show the figure
        plt.show()

        # sample counter (used for plot ground truth bounding boxes)
        i += 1