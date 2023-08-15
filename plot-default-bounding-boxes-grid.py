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
    feature_maps_shapes=((30, 40), (15, 20), (8, 10), (4, 5)),
    centers_padding_from_borders_percentage=0.025,
    additional_square_box=True
)

# scale default bounding boxes to image shape
default_bounding_boxes.rescale_boxes_coordinates(image_shape=image_shape)

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
for boxes_default, color in zip(default_bounding_boxes.get_boxes_coordinates_centroids('feature-maps'), colors):

    # reshape
    boxes_default = boxes_default.reshape((-1, boxes_default.shape[-1]))

    # set plot axes limits
    axes[i].set_xlim(0, image_shape[1])
    axes[i].set_ylim(0, image_shape[0])

    # plot points
    axes[i].set_title(f'feature map boxes grid {i}')
    axes[i].scatter(x=boxes_default[:, 0], y=boxes_default[:, 1], color=color, marker='o', s=1)
    
    i+=1

plt.show()
