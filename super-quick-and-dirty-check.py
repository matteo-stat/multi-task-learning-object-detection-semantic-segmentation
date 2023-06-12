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

# check = 'bbox_labels'
# check = 'bbox_grid_default'
check = 'bbox_default'

for path_image, path_mask, labels in random.sample(data, 10):
    image = tf.io.read_file(path_image)
    image = tf.image.decode_png(image, channels=3)
    image = tf.cast(image, dtype=tf.float32) / 255.0

    if check == 'bbox_labels':
        # display the image
        plt.imshow(image, vmin=0, vmax=1)
        plt.axis('off')

        # get the current plot object
        ax = plt.gca()

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

        # plot bounding boxes
        for label, xmin, ymin, xmax, ymax, cx, cy, w, h in labels:
            rect = patches.Rectangle((xmin, ymin), w, h, linewidth=1, edgecolor=label_code_to_color[label], facecolor='none')
            ax.add_patch(rect)
            plt.text(xmin, ymin, label_code_to_str[label], fontsize=8, color=label_code_to_color[label], verticalalignment='top')
    
    elif check == 'bbox_grid_default':
        # create subplots and set figure size
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
        fig.set_size_inches(30, 10)

        # image subplot
        ax1.imshow(image, vmin=0, vmax=1)
        ax1.set_axis_off()
        ax1.set_title('grid feat map 1')
        
        # mask real subplot (drop background channel)        
        ax2.imshow(image, vmin=0, vmax=1)
        ax2.set_axis_off()
        ax2.set_title('grid feat map 2')

        # mask predicted subplot (drop background channel)
        ax3.imshow(image, vmin=0, vmax=1)
        ax3.set_axis_off()
        ax3.set_title('grid feat map 3')

        # example feature maps shapes
        feature_maps_shapes = (
            (48, 64),
            (24, 32),
            (12, 16)
        )

        import ssd

        # generate bounding boxes
        feature_maps_boxes = ssd.generate_default_bounding_boxes(feature_maps_shapes=feature_maps_shapes)

        # convert to image coordinates and plot
        for boxes, feature_map_shape, color, ax in zip(feature_maps_boxes, feature_maps_shapes, ('red', 'blue', 'green'), (ax1, ax2, ax3)):
            points_x = []
            points_y = []

            for cx, cy, w, h in boxes.reshape((-1, 4)):
                cx = cx / feature_map_shape[1] * image.shape[1]
                cy = cy / feature_map_shape[0] * image.shape[0]
                w = w / feature_map_shape[1] * image.shape[1]
                h = h / feature_map_shape[0] * image.shape[0]

                points_x.append(cx)
                points_y.append(cy)

            ax.scatter(x=points_x, y=points_y, color=color, marker='o', s=1)

            # break


        plt.show()

    elif check == 'bbox_default':
        # create subplots and set figure size
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
        fig.set_size_inches(20, 8)

        # image subplot
        ax1.imshow(image, vmin=0, vmax=1)
        ax1.set_axis_off()
        ax1.set_title('bbox feat map 1')
        
        # mask real subplot (drop background channel)        
        ax2.imshow(image, vmin=0, vmax=1)
        ax2.set_axis_off()
        ax2.set_title('bbox feat map 2')

        # mask predicted subplot (drop background channel)
        ax3.imshow(image, vmin=0, vmax=1)
        ax3.set_axis_off()
        ax3.set_title('bbox feat map 3')

        # example feature maps shapes
        feature_maps_shapes = (
            (48, 64),
            (24, 32),
            (12, 16)
        )

        import ssd

        # generate bounding boxes
        feature_maps_boxes = ssd.generate_default_bounding_boxes(
            feature_maps_shapes=feature_maps_shapes,
            feature_maps_aspect_ratios=((1.0, 2.0, 3.0, 1/2, 1/3), (1, 4.0), (1/2, 1/3, 1/4)),
            boxes_scales=(0.1, 0.7),
            additional_square_box=False
        )

        # convert to image coordinates and plot
        for boxes, feature_map_shape, color, ax in zip(feature_maps_boxes, feature_maps_shapes, ('red', 'blue', 'green'), (ax1, ax2, ax3)):
            points_x = []
            points_y = []

            boxes_sample = boxes[int(feature_map_shape[0] / 2), int(feature_map_shape[1] / 2)]
            boxes_sample = boxes_sample.reshape((-1, 4))
            
            for cx, cy, w, h in boxes_sample:
                cx = cx / feature_map_shape[1] * image.shape[1]
                cy = cy / feature_map_shape[0] * image.shape[0]
                w = w / feature_map_shape[1] * image.shape[1]
                h = h / feature_map_shape[0] * image.shape[0]

                ax.add_patch(patches.Rectangle((cx - w/2, cy - h/2), w, h, linewidth=1, edgecolor=color, facecolor='none'))
            # break


        plt.show()