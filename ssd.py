import numpy as np
import math
from typing import Literal

def get_scale_default_bounding_boxes(k: int, m: int, scale_min: float = 0.2, scale_max: float = 0.9) -> float:
    """
    calculate the scale of the default boxes for a given feature map, as suggested from the original single shot multibox detector paper

    Args:
        k (int): current feature maps index (starting from 1, not 0)
        m (int, optional): total number of feature maps . Defaults to 6.
        scale_min (float, optional): minimum scale for default bounding boxes. Defaults to 0.2.
        scale_max (float, optional): maximum scale for default bounding boxes. Defaults to 0.9.

    Returns:
        float: the scale for the current feature map
    """
    return scale_min + (scale_max - scale_min) * (k - 1) / (m - 1)


def generate_default_bounding_boxes(
        feature_maps_shapes: tuple[tuple[int]],
        feature_maps_aspect_ratios: tuple[float] | tuple[tuple[float]] = (1.0, 2.0, 3.0, 1/2, 1/3),
        centers_padding_from_borders: float = 0.5,
        boxes_scales: tuple[float] | tuple[tuple[float]] = (0.2, 0.9),
        additional_square_box: bool = True,
    ) -> list[np.ndarray]:
    """
    generate default bounding boxes for the given feature maps shapes and aspect ratios
    for each pixel of a feature map, a number of len(aspect_ratios) + 1 default bounding boxes are calculated

    Args:
        feature_maps_shapes (tuple[tuple[int]]): a list of shapes for the feature maps on which we want to generate default bounding boxes
        aspect_ratios (tuple[float] | tuple[tuple[float]], optional): a list of aspect ratios. Defaults to (1.0, 2.0, 3.0, 1/2, 1/3).
        centers_padding_from_borders (float): padding margin from borders for the centers grid Defaults to 0.5.
        boxes_scales (tuple[float], optional): minimum and maximum boxes scales. Defaults to (0.2, 0.9).
        additional_square_box (bool, optional): additional square box proposed by original ssd paper. Defaults to True.

    Returns:
        list.[nd.array]: a list of multi-dimensional numpy arrays, with shape (feature_map_shape[0], feature_map_shape[1], len(aspect_ratios) + 1, 4), where last dimension contains coordinates for the bounding boxes
    """    
    
    # if feature_maps_aspect_ratios it's simply a tuple[float] then convert it to a tuple[tuple[float]]
    if all(isinstance(item, float) for item in feature_maps_aspect_ratios):
        feature_maps_aspect_ratios = tuple(feature_maps_aspect_ratios for _ in range(len(feature_maps_shapes)))

    # list to store boxes for each feature map
    feature_maps_boxes = []

    # calculate boxes for each feature map
    for feature_map_index, (feature_map_shape, aspect_ratios) in enumerate(zip(feature_maps_shapes, feature_maps_aspect_ratios)):
        
        # get the smallest side of the feature map shape
        feature_map_size = min(feature_map_shape)

        # get scales for current feature map and the next one
        scale_current = get_scale_default_bounding_boxes(k=feature_map_index + 1, m=len(feature_maps_shapes), scale_min=boxes_scales[0], scale_max=boxes_scales[1])
        scale_next = get_scale_default_bounding_boxes(k=feature_map_index + 2, m=len(feature_maps_shapes), scale_min=boxes_scales[0], scale_max=boxes_scales[1])

        # calculate width and height for each aspect ratio
        # also calculate an additional square box with different scale, as proposed in original ssd paper
        # the output is a list of lists like [[width, height], ..., [width, height]], same length as given aspect ratios list
        boxes_width_height = [
            [feature_map_size * scale_current * math.sqrt(aspect_ratio), feature_map_size * scale_current / math.sqrt(aspect_ratio)]
            for aspect_ratio in aspect_ratios
        ]

        # optionally add an additional square box, with a different scale, as proposed by original ssd paper
        if additional_square_box:
            boxes_width_height.append([feature_map_size * math.sqrt(scale_current * scale_next), feature_map_size * math.sqrt(scale_current * scale_next)])

        # convert to numpy array
        boxes_width_height = np.array(boxes_width_height)

        # calculate centers coordinates for the boxes
        # there is a center for each pixel in the current feature map
        boxes_center_x = np.linspace(start=centers_padding_from_borders, stop=feature_map_shape[1] - 1 - centers_padding_from_borders, num=feature_map_shape[1])
        boxes_center_y = np.linspace(start=centers_padding_from_borders, stop=feature_map_shape[0] - 1 - centers_padding_from_borders, num=feature_map_shape[0])

        # manipulate the arrays of centers to get outputs of shape (feature_map_shape_y, feature_map_shape_x, 1)
        boxes_center_x, boxes_center_y = np.meshgrid(boxes_center_x, boxes_center_y)
        boxes_center_x = np.expand_dims(boxes_center_x, axis=-1)
        boxes_center_y = np.expand_dims(boxes_center_y, axis=-1)
        
        # prepare final output array with shape (feature_map_shape_y, feature_map_shape_x, number_of_boxes, 4)
        # we end up with an array containing box coordinates for each aspect ratio (+1) for each pixel in the feature map
        boxes = np.zeros((feature_map_shape[0], feature_map_shape[1], len(boxes_width_height), 4))

        # populate the output array
        # assign to the last dimension the 4 values x_min, y_min, x_max, y_max (normalized)
        boxes[:, :, :, 0] = (np.tile(boxes_center_x, (1, 1, len(boxes_width_height))) - (boxes_width_height[:, 0] - 1)/ 2) / feature_map_shape[1]
        boxes[:, :, :, 1] = (np.tile(boxes_center_y, (1, 1, len(boxes_width_height))) - (boxes_width_height[:, 1] - 1)/ 2) / feature_map_shape[0]
        boxes[:, :, :, 2] = (np.tile(boxes_center_x, (1, 1, len(boxes_width_height))) + (boxes_width_height[:, 0] - 1)/ 2) / feature_map_shape[1]
        boxes[:, :, :, 3] = (np.tile(boxes_center_y, (1, 1, len(boxes_width_height))) + (boxes_width_height[:, 1] - 1)/ 2) / feature_map_shape[0]

        # append boxes calculated for the current feature map
        feature_maps_boxes.append(boxes)

    return feature_maps_boxes
