from ssdseglib import boxes, plot
from matplotlib import pyplot as plt, colors as pltcolors

# input image shape
image_shape = (480, 640)

# plot type
fig_size_width = 11
fig_rows = 2
fig_cols = 2

# create default bounding boxes
default_bounding_boxes = boxes.DefaultBoundingBoxes(
    feature_maps_shapes=((24, 32), (12, 16), (6, 8), (3, 4)),
    feature_maps_aspect_ratios=((1.0, 2.0, 3.0, 1/2, 1/3), (1.0, 4.0), (1/2, 1/3, 1/4), (1.0, 2.0, 3.0, 1/2, 1/3)),
    centers_padding_from_borders=0.5,
    boxes_scales=(0.1, 0.5),
    additional_square_box=True
)
feature_maps_boxes = default_bounding_boxes.get_feature_maps_boxes()

# create subplots and set figure size
fig, axes = plt.subplots(nrows=fig_rows, ncols=fig_cols, constrained_layout=True)
fig.set_size_inches(fig_size_width, int(fig_size_width / (image_shape[1] / image_shape[0])))
plot.move_figure(fig=fig, x=0, y=0)

# set aspect ratio for each subplot
for ax in axes.flat:
    ax.set_aspect('equal')
axes = axes.flatten()
colors = list(pltcolors.BASE_COLORS.values())[:len(default_bounding_boxes.feature_maps_shapes)]

# scale and convert to centroids boxes for each feature map
i = 0
for boxes_default, color, feature_map_aspect_ratios in zip(feature_maps_boxes, colors, default_bounding_boxes.feature_maps_aspect_ratios):  
    
    # reshape to network output shape
    boxes_default = boxes_default.reshape((-1, boxes_default.shape[-1]))

    # scale to image shape
    boxes_default[:, [0, 2]] = boxes_default[:, [0, 2]] * image_shape[1]
    boxes_default[:, [1, 3]] = boxes_default[:, [1, 3]] * image_shape[0]

    # convert to centroids coordinates
    boxes_default = boxes.boxes_corners_to_centroids(boxes=boxes_default)

    # set plot axes limits
    axes[i].set_xlim(0, image_shape[1])
    axes[i].set_ylim(0, image_shape[0])

    # plot points
    axes[i].set_title(f'feature map boxes grid {i}')
    axes[i].scatter(x=boxes_default[:, 0], y=boxes_default[:, 1], color=color, marker='o', s=1)
    
    i+=1

plt.show()
