import numpy as np
import math

class DefaultBoundingBoxes:
    def __init__(self,
            feature_maps_shapes: tuple[tuple[int]],
            feature_maps_aspect_ratios: tuple[float] = (1.0, 2.0, 3.0, 1/2, 1/3),
            # feature_maps_aspect_ratios: tuple[float] | tuple[tuple[float]] = (1.0, 2.0, 3.0, 1/2, 1/3),
            centers_padding_from_borders: float = 0.5,
            boxes_scales: tuple[float] = (0.2, 0.9),
            # boxes_scales: tuple[float] | tuple[tuple[float]] = (0.2, 0.9),
            additional_square_box: bool = True,                 
        ) -> None:
        """
        class for creating and managing default bounding boxes

        Args:
            feature_maps_shapes (tuple[tuple[int]]): a list of shapes for the feature maps on which we want to generate default bounding boxes
            aspect_ratios (tuple[float] | tuple[tuple[float]], optional): a list of aspect ratios. Defaults to (1.0, 2.0, 3.0, 1/2, 1/3).
            centers_padding_from_borders (float): padding margin from borders for the centers grid Defaults to 0.5.
            boxes_scales (tuple[float], optional): minimum and maximum boxes scales. Defaults to (0.2, 0.9).
            additional_square_box (bool, optional): additional square box proposed by original ssd paper. Defaults to True.

        Returns:
            list.[nd.array]: a list of multi-dimensional numpy arrays, with shape (feature_map_shape[0], feature_map_shape[1], len(aspect_ratios) + 1, 4), where last dimension contains corners coordinates for the default bounding boxes
        """

        # set attributes with arguments values
        self.feature_maps_shapes = feature_maps_shapes
        self.feature_maps_aspect_ratios = feature_maps_aspect_ratios
        self.centers_padding_from_borders = centers_padding_from_borders
        self.boxes_scales = boxes_scales
        self.additional_square_box = additional_square_box

        # initialize other attributes
        self.boxes_scaled_to_image = False
        self.feature_maps_boxes = None
        self.xmin = None
        self.ymin = None
        self.xmax = None
        self.ymax = None
        self.center_x = None
        self.center_y = None
        self.width = None
        self.height = None

        # generate default bounding boxes for each given feature map
        self._generate_feature_maps_boxes()

        # create corners coordinates from 


    
    def _generate_feature_maps_boxes(self,
            feature_maps_shapes: tuple[tuple[int]],
            feature_maps_aspect_ratios: tuple[float],
            # feature_maps_aspect_ratios: tuple[float] | tuple[tuple[float]],
            centers_padding_from_borders: float,
            boxes_scales: tuple[float],
            # boxes_scales: tuple[float] | tuple[tuple[float]],
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
            list.[nd.array]: a list of multi-dimensional numpy arrays, with shape (feature_map_shape[0], feature_map_shape[1], len(aspect_ratios) + 1, 4), where last dimension contains corners coordinates for the default bounding boxes
        """    
        
        # if feature_maps_aspect_ratios it's simply a tuple[float] then convert it to a tuple[tuple[float]]
        if all(isinstance(item, float) for item in feature_maps_aspect_ratios):
            feature_maps_aspect_ratios = tuple(feature_maps_aspect_ratios for _ in range(len(feature_maps_shapes)))

        # list to store boxes for each feature map
        feature_maps_boxes = []

        # list of linearly spaced scales for boxes of different feature maps
        scales = np.linspace(boxes_scales[0], boxes_scales[1], len(feature_maps_shapes) + 1)

        # calculate boxes for each feature map
        for feature_map_index, (feature_map_shape, aspect_ratios) in enumerate(zip(feature_maps_shapes, feature_maps_aspect_ratios)):
            
            # get the smallest side of the feature map shape
            feature_map_size = min(feature_map_shape)

            # get scales for current feature map and the next one
            scale_current = scales[feature_map_index]
            scale_next = scales[feature_map_index + 1]

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
            boxes[:, :, :, 0] = (np.tile(boxes_center_x, (1, 1, len(boxes_width_height))) - boxes_width_height[:, 0] / 2) / (feature_map_shape[1] - 1)
            boxes[:, :, :, 1] = (np.tile(boxes_center_y, (1, 1, len(boxes_width_height))) - boxes_width_height[:, 1] / 2) / (feature_map_shape[0] - 1)
            boxes[:, :, :, 2] = (np.tile(boxes_center_x, (1, 1, len(boxes_width_height))) + boxes_width_height[:, 0] / 2) / (feature_map_shape[1] - 1)
            boxes[:, :, :, 3] = (np.tile(boxes_center_y, (1, 1, len(boxes_width_height))) + boxes_width_height[:, 1] / 2) / (feature_map_shape[0] - 1)

            # append boxes calculated for the current feature map
            feature_maps_boxes.append(boxes)

        # set feature maps boxes attribute
        self.feature_maps_boxes = feature_maps_boxes

    def _feature_maps_boxes_to_corners_coordinates(self, feature_maps_boxes: list[np.ndarray]):
        # default bounding boxes, reshaped as output from the network
        feature_maps_boxes = np.concatenate([np.reshape(boxes, newshape=(-1, 4)) for boxes in feature_maps_boxes], axis=0)

        # corners coordinates for default bounding boxes, scaled at image shape
        xmin_boxes, ymin_boxes, xmax_boxes, ymax_boxes = np.split(feature_maps_boxes, 4, axis=-1)
        xmin_boxes = xmin_boxes.reshape(-1)
        ymin_boxes = ymin_boxes.reshape(-1)
        xmax_boxes = xmax_boxes.reshape(-1)
        ymax_boxes = ymax_boxes.reshape(-1)

        return xmin


    def _bounding_boxes_corners_to_centroids(self, xmin: np.ndarray[float], ymin: np.ndarray[float], xmax: np.ndarray[float], ymax: np.ndarray[float]):
        
        # calculate bounding boxes centroids coordinates
        center_x = (xmax + xmin) / 2.
        center_y = (ymax + ymin) / 2.
        width = xmax - xmin + 1.
        height = ymax - ymin + 1.

        return center_x, center_y, width, height

    def _bounding_boxes_centroids_to_corners(self, center_x: np.ndarray[float], center_y: np.ndarray[float], width: np.ndarray[float], height: np.ndarray[float]):
        # calculate bounding boxes corners coordinates
        xmin = center_x - (width - 1.) / 2.
        ymin = center_y - (height - 1.) / 2.
        xmax = center_x + (width - 1.) / 2.
        ymax = center_y + (height - 1.) / 2.

        return xmin, ymin, xmax, ymax

def bounding_boxes_corners_to_centroids(boxes: np.ndarray[np.ndarray]) -> np.ndarray[np.ndarray]:
    """
    convert bounding boxes coordinates from corners to centroids

    Args:
        boxes (np.ndarray[np.ndarray]): boxes with corners coordinates (x_min, y_min, x_max, y_max)

    Returns:
        np.ndarray[np.ndarray]: boxes with centroids coordinates (center_x, center_y, width, height), same shape as input
    """

    # prepare output object
    centroids = np.empty_like(boxes)

    # center x and y
    centroids[:, [0, 1]] = (boxes[:, [2, 3]] + boxes[:, [0, 1]]) / 2

    # width and height
    centroids[:, [2, 3]] = boxes[:, [2, 3]] - boxes[:, [0, 1]] + 1

    return centroids


def bounding_boxes_centroids_to_corners(boxes: np.ndarray[np.ndarray]) -> np.ndarray[np.ndarray]:
    """
    convert bounding boxes coordinates from centroids to corners

    Args:
        boxes (np.ndarray[np.ndarray]): boxes with centroids coordinates (center_x, center_y, width, height)

    Returns:
        np.ndarray[np.ndarray]: boxes with corners coordinates (x_min, y_min, x_max, y_max), same shape as input
    """

    # prepare output boxes object
    corners = np.empty_like(boxes)

    # center x and y
    corners[:, [0, 1]] = boxes[:, [0, 1]] - (boxes[:, [2, 3]] - 1) / 2

    # width and height
    corners[:, [2, 3]] = boxes[:, [0, 1]] + (boxes[:, [2, 3]] - 1) / 2
    
    return corners

