from ssdseglib import boxes, plot
from matplotlib import pyplot as plt, colors as pltcolors

# input image shape
INPUT_IMAGE_SHAPE = (480, 640)

# plot options
fig_size_width = 11
fig_rows = 2
fig_cols = 2

# create default bounding boxes
default_bounding_boxes = boxes.DefaultBoundingBoxes(
    feature_maps_shapes=((30, 40), (15, 20), (8, 10), (4, 5)),
    feature_maps_aspect_ratios=(1.0,),
    centers_padding_from_borders_percentage=0.025,
    boxes_scales=(0.2, 0.9),
    additional_square_box=True,  
)

# scale default bounding boxes to image shape
default_bounding_boxes.rescale_boxes_coordinates(image_shape=INPUT_IMAGE_SHAPE)

# create subplots and set figure size
fig, axes = plt.subplots(nrows=fig_rows, ncols=fig_cols, constrained_layout=True)
fig.set_size_inches(fig_size_width, int(fig_size_width / (INPUT_IMAGE_SHAPE[1] / INPUT_IMAGE_SHAPE[0])))
plot.move_figure(fig=fig, x=0, y=0)

# set aspect ratio for each subplot
for ax in axes.flat:
    ax.set_aspect('equal')
axes = axes.flatten()
colors = list(pltcolors.BASE_COLORS.values())[:len(default_bounding_boxes.feature_maps_shapes)]

# get boxes centroids coordinates for each feature map and plot the boxes centers
i = 0
for boxes_default, color in zip(default_bounding_boxes.get_boxes_coordinates_centroids('feature-maps'), colors):

    # reshape
    boxes_default = boxes_default.reshape((-1, boxes_default.shape[-1]))

    # set plot axes limits
    axes[i].set_xlim(0, INPUT_IMAGE_SHAPE[1])
    axes[i].set_ylim(0, INPUT_IMAGE_SHAPE[0])

    # plot centers
    axes[i].set_title(f'feature map boxes grid {i}')
    axes[i].scatter(x=boxes_default[:, 0], y=boxes_default[:, 1], color=color, marker='o', s=1)
    
    i+=1

# show the plot
plt.show()
