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
fig_size_width = 8

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
    standard_deviations_centroids_offsets=(0.1, 0.1, 0.2, 0.2),
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
        
        # read labels boxes from csv file, this is for ground truth
        with open(path_labels_boxes_train[i], 'r') as f:
            labels_boxes = list(csv.reader(f))

        # create the needed subplots and set figure size
        fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(nrows=2, ncols=2,  constrained_layout=True)        
        fig.set_size_inches(fig_size_width, int(fig_size_width / (INPUT_IMAGE_SHAPE[1] / INPUT_IMAGE_SHAPE[0])))
        ssdseglib.plot.move_figure(fig=fig, x=0, y=0)
        
        # ------------------------------------------------------------------------------------------------------------------
        # ground truth - bounding boxes
        # ------------------------------------------------------------------------------------------------------------------
        # this is useful to see the original data, untouched by the tensorflow data pipeline

        # read ground truth image        
        image = Image.open(path_images_train[i])
        image = np.array(image)
        image = image.astype(np.int32)

        # setup the subplot
        ax1.set_aspect('equal')
        ax1.imshow(image, vmin=0, vmax=255)
        ax1.set_axis_off()
        ax1.set_title(f'ground truth data ({Path(path_images_train[i]).name})')

        # plot ground truth boxes
        for label, xmin, ymin, xmax, ymax in labels_boxes:
            label = int(label)
            xmin = float(xmin)
            ymin = float(ymin)
            xmax = float(xmax)
            ymax = float(ymax)        
            rect = patches.Rectangle((xmin, ymin), xmax - xmin + 1, ymax - ymin + 1, linewidth=1, edgecolor=label_code_to_color[label], facecolor='none')
            ax1.add_patch(rect)
            ax1.text(xmin, ymin, label_code_to_str[label], fontsize=8, color=label_code_to_color[label], verticalalignment='top')        

        # ------------------------------------------------------------------------------------------------------------------
        # encoded masks - semantic segmentation mask
        # ------------------------------------------------------------------------------------------------------------------
        # remove the background class and keep the other 3 classes on rgb channels
        mask_sample = tf.slice(mask_sample, begin=[0, 0, 1], size=[-1, -1, 3])

        # setup the subplot
        ax2.set_aspect('equal')
        ax2.imshow(mask_sample, vmin=0.0, vmax=1.0)
        ax2.set_axis_off()
        ax2.set_title('segmentation mask after encoding')

        # ------------------------------------------------------------------------------------------------------------------
        # decoded boxes - show the matched boxes after the encoding decoding process
        # ------------------------------------------------------------------------------------------------------------------
        # this subplot it's very important because it can easily show if the encoding decoding process was done properly or not
        # if some default bounding boxes were matched with ground truth boxes and properly encoded,
        # then through decoding process we should get back coordinates for the original ground truth boxes
        # basically the decoded boxes should match the ground truth ones

        # use the one-hot-encoded labels to create a boolean vector for select items not related to background class
        not_background = tf.math.equal(labels_sample[:, 0], 0.0)

        # keep only decoded boxes not related to background class
        decoded_boxes = data_reader_encoder.decode_to_corners(offsets_centroids=boxes_sample)
        decoded_boxes_sample_not_background = tf.boolean_mask(
            tensor=decoded_boxes,
            mask=not_background,
            axis=0
        )

        # keep only corresponding labels not related to background class
        valid_labels_sample = tf.boolean_mask(
            tensor=labels_sample,
            mask=not_background,
            axis=0
        )

        # setup subplot
        ax3.set_aspect('equal')
        ax3.imshow(tf.cast(image_sample, tf.int32), vmin=0, vmax=255)
        ax3.set_axis_off()
        ax3.set_title(f'decoded offsets after encoding ({decoded_boxes_sample_not_background.shape[0]} boxes)')

        # plot decoded boxes
        for labels_one_hot_encoded, (xmin, ymin, xmax, ymax) in zip(valid_labels_sample, decoded_boxes_sample_not_background):
            label = int(tf.argmax(labels_one_hot_encoded))
            xmin = float(xmin)
            ymin = float(ymin)
            xmax = float(xmax)
            ymax = float(ymax)            
            rect = patches.Rectangle((xmin, ymin), xmax - xmin + 1, ymax - ymin + 1, linewidth=1, edgecolor=label_code_to_color[label], facecolor='none')
            ax3.add_patch(rect)
            ax3.text(xmin, ymin, label_code_to_str[label], fontsize=8, color=label_code_to_color[label], verticalalignment='top')

        # ------------------------------------------------------------------------------------------------------------------
        # default bounding boxe boxes - show the matched default bounding boxes
        # ------------------------------------------------------------------------------------------------------------------
        # this is just for fun :)

        # keep default bounding boxes not related to background class 
        default_boxes_sample_not_background = tf.concat(
            values=[
                tf.expand_dims(input=tf.boolean_mask(tensor=data_reader_encoder.xmin_boxes_default, mask=not_background, axis=0), axis=1),
                tf.expand_dims(input=tf.boolean_mask(tensor=data_reader_encoder.ymin_boxes_default, mask=not_background, axis=0), axis=1),
                tf.expand_dims(input=tf.boolean_mask(tensor=data_reader_encoder.xmax_boxes_default, mask=not_background, axis=0), axis=1),
                tf.expand_dims(input=tf.boolean_mask(tensor=data_reader_encoder.ymax_boxes_default, mask=not_background, axis=0), axis=1),
            ],
            axis=1
        )

        # setup subplot
        ax4.set_aspect('equal')
        ax4.imshow(tf.cast(image_sample, tf.int32), vmin=0, vmax=255)
        ax4.set_axis_off()
        ax4.set_title(f'default boxes ({default_boxes_sample_not_background.shape[0]} boxes)')

        # plot default bounding boxes
        for labels_one_hot_encoded, (xmin, ymin, xmax, ymax) in zip(valid_labels_sample, default_boxes_sample_not_background):
            label = int(tf.argmax(labels_one_hot_encoded))
            xmin = float(xmin)
            ymin = float(ymin)
            xmax = float(xmax)
            ymax = float(ymax)            
            rect = patches.Rectangle((xmin, ymin), xmax - xmin + 1, ymax - ymin + 1, linewidth=1, edgecolor=label_code_to_color[label], facecolor='none')
            ax4.add_patch(rect)
            
            # text overlaps a lot between default bounding boxes, maybe it's better to hide it on this subplot
            # ax4.text(xmin, ymin, label_code_to_str[label], fontsize=8, color=label_code_to_color[label], verticalalignment='top')

        # show the figure
        plt.show()

        # sample counter (used for plot ground truth bounding boxes)
        i += 1
    