import numpy as np
import math

class DefaultBoundingBoxes:
    def __init__(self,
            feature_maps_shapes: tuple[tuple[int]],
            feature_maps_aspect_ratios: tuple[float] | tuple[tuple[float]] = (1.0, 2.0, 3.0, 1/2, 1/3),
            centers_padding_from_borders: float = 0.5,
            boxes_scales: tuple[float] | tuple[tuple[float]] = (0.2, 0.9),
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
        """

        # set attributes with arguments values
        self.feature_maps_shapes = feature_maps_shapes
        self.feature_maps_aspect_ratios = feature_maps_aspect_ratios
        self.centers_padding_from_borders = centers_padding_from_borders
        self.boxes_scales = boxes_scales
        self.additional_square_box = additional_square_box

        # generate default bounding boxes
        # the coordinates space needs to be scaled to the input image size
        self._feature_maps_boxes = self._generate_feature_maps_boxes(
            feature_maps_shapes=self.feature_maps_shapes,
            feature_maps_aspect_ratios=self.feature_maps_aspect_ratios,
            centers_padding_from_borders=self.centers_padding_from_borders,
            boxes_scales=self.boxes_scales,
            additional_square_box=self.additional_square_box
        )

        # initialize other attributes with default values
        self.boxes = False
        self.xmin = None
        self.ymin = None
        self.xmax = None
        self.ymax = None
        self.center_x = None
        self.center_y = None
        self.width = None
        self.height = None
    
    def _generate_feature_maps_boxes(self,
            feature_maps_shapes: tuple[tuple[int]],
            feature_maps_aspect_ratios: tuple[float] | tuple[tuple[float]],
            centers_padding_from_borders: float,
            boxes_scales: tuple[float] | tuple[tuple[float]],
            additional_square_box: bool = True,
        ) -> list[np.ndarray]:
        """
        generate default bounding boxes for the given feature maps shapes and aspect ratios
        for each pixel of each feature map, a number of len(aspect_ratios) + 1 default bounding boxes are calculated

        Args:
            feature_maps_shapes (tuple[tuple[int]]): a list of shapes for the feature maps on which we want to generate default bounding boxes
            aspect_ratios (tuple[float] | tuple[tuple[float]], optional): a list of aspect ratios. Defaults to (1.0, 2.0, 3.0, 1/2, 1/3).
            centers_padding_from_borders (float): padding margin from borders for the centers grid Defaults to 0.5.
            boxes_scales (tuple[float], optional): minimum and maximum boxes scales. Defaults to (0.2, 0.9).
            additional_square_box (bool, optional): additional square box proposed by original ssd paper. Defaults to True.

        Returns:
            list.[nd.array]: 
                a list of multi-dimensional numpy arrays, with shape (feature_map_shape[0], feature_map_shape[1], len(aspect_ratios) + 1, 4)
                the last dimension contains corners coordinates for the default bounding boxes: xmin, ymin, xmax, ymax
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
            # there is a center for each pixel of the feature map
            boxes_center_x = np.linspace(start=centers_padding_from_borders, stop=feature_map_shape[1] - 1 - centers_padding_from_borders, num=feature_map_shape[1])
            boxes_center_y = np.linspace(start=centers_padding_from_borders, stop=feature_map_shape[0] - 1 - centers_padding_from_borders, num=feature_map_shape[0])

            # manipulate the arrays of centers to get outputs of shape (feature_map_shape_y, feature_map_shape_x, 1)
            boxes_center_x, boxes_center_y = np.meshgrid(boxes_center_x, boxes_center_y)
            boxes_center_x = np.expand_dims(boxes_center_x, axis=-1)
            boxes_center_y = np.expand_dims(boxes_center_y, axis=-1)
            
            # prepare final output array with shape (feature_map_shape_y, feature_map_shape_x, number_of_boxes, 4)
            # we end up with an array containing box coordinates for each aspect ratio (+1 optionally), for each pixel of the feature map
            boxes = np.zeros((feature_map_shape[0], feature_map_shape[1], len(boxes_width_height), 4), dtype=np.float32)

            # populate the output array
            # assign to the last dimension the 4 values x_min, y_min, x_max, y_max (normalized)
            boxes[:, :, :, 0] = (np.tile(boxes_center_x, (1, 1, len(boxes_width_height))) - boxes_width_height[:, 0] / 2.0) / (feature_map_shape[1] - 1.0)
            boxes[:, :, :, 1] = (np.tile(boxes_center_y, (1, 1, len(boxes_width_height))) - boxes_width_height[:, 1] / 2.0) / (feature_map_shape[0] - 1.0)
            boxes[:, :, :, 2] = (np.tile(boxes_center_x, (1, 1, len(boxes_width_height))) + boxes_width_height[:, 0] / 2.0) / (feature_map_shape[1] - 1.0)
            boxes[:, :, :, 3] = (np.tile(boxes_center_y, (1, 1, len(boxes_width_height))) + boxes_width_height[:, 1] / 2.0) / (feature_map_shape[0] - 1.0)

            # append boxes calculated for the feature map
            feature_maps_boxes.append(boxes)

        return feature_maps_boxes

    def calculate_boxes_coordinates(self, image_shape: tuple[int]) -> None:
        """
        calculate and scale default bounding boxes coordinates

        Args:
            image_shape (tuple[int]): input image shape            
        """

        # default bounding boxes, reshaped as expected output from a ssd network
        self.boxes = np.concatenate([np.reshape(feature_map_boxes, newshape=(-1, 4)) for feature_map_boxes in self._feature_maps_boxes], axis=0)

        # scale default bounding boxes coordinates to input image size
        self.boxes[:, [0, 2]] = self.boxes[:, [0, 2]] * image_shape[1]
        self.boxes[:, [1, 3]] = self.boxes[:, [0, 2]] * image_shape[0]

        # set corners coordinates
        self.xmin, self.ymin, self.xmax, self.ymax = np.split(self.boxes, 4, axis=-1)
        self.xmin = self.xmin.reshape(-1)
        self.ymin = self.ymin.reshape(-1)
        self.xmax = self.xmax.reshape(-1)
        self.ymax = self.ymax.reshape(-1)

        # set centroids coordinates
        self.center_x = (self.xmax + self.xmin) / 2.0
        self.center_y = (self.ymax + self.ymin) / 2.0
        self.width = self.xmax - self.xmin + 1.0
        self.height = self.ymax - self.ymin + 1.0

    def get_feature_maps_boxes(self) -> list[np.ndarray]:
        """
        simply returns unscaled raw boxes for each feature map

        Returns:
            list[np.ndarray]:
                a list with boxes for each feature map
                boxes are numpy arrays with shape (feature_map_shape_y, feature_maps_shape_x, len(aspect_ratios) + 1, 4)
                for each position/pixel in the feature map there are n default bounding boxes
                boxes are described in corners coordinates (x_min, y_min, x_max, y_max)
        """
        return self._feature_maps_boxes

def boxes_corners_to_centroids(boxes: np.ndarray[np.ndarray]) -> np.ndarray[np.ndarray]:
    """
    convert bounding boxes coordinates from corners to centroids

    Args:
        boxes (np.ndarray[np.ndarray]): corners coordinates are expected on last axis in the order (x_min, y_min, x_max, y_max)

    Returns:
        np.ndarray[np.ndarray]: boxes with centroids coordinates (center_x, center_y, width, height), same shape as input
    """

    # prepare output object
    centroids = np.empty_like(boxes, dtype=np.float32)

    # center x and y
    centroids[:, [0, 1]] = (boxes[:, [2, 3]] + boxes[:, [0, 1]]) / 2.0

    # width and height
    centroids[:, [2, 3]] = boxes[:, [2, 3]] - boxes[:, [0, 1]] + 1.0

    return centroids

def boxes_centroids_to_corners(boxes: np.ndarray[np.ndarray]) -> np.ndarray[np.ndarray]:
    """
    convert bounding boxes coordinates from centroids to corners

    Args:
        boxes (np.ndarray[np.ndarray]): centroids coordinates are expected on last axis in the order (center_x, center_y, width, height)

    Returns:
        np.ndarray[np.ndarray]: boxes with corners coordinates (x_min, y_min, x_max, y_max), same shape as input
    """

    # prepare output boxes object
    corners = np.empty_like(boxes)

    # xmin, ymin
    corners[:, [0, 1]] = boxes[:, [0, 1]] - (boxes[:, [2, 3]] - 1) / 2

    # xmax, ymax
    corners[:, [2, 3]] = boxes[:, [0, 1]] + (boxes[:, [2, 3]] - 1) / 2
    
    return corners

def coordinates_corners_to_centroids(
        xmin: np.ndarray[float],
        ymin: np.ndarray[float],
        xmax: np.ndarray[float],
        ymax: np.ndarray[float]
    ) -> tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
    """
    convert corners coordinates to centroids

    Args:
        xmin (np.ndarray[float]): xmin coordinates
        ymin (np.ndarray[float]): ymin coordinates
        xmax (np.ndarray[float]): xmax coordinates
        ymax (np.ndarray[float]): ymax coordinates

    Returns:
        tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float]]: centroids coordinates (center_x, center_y, width, height)
    """

    # calculate centroids coordinates
    center_x = (xmax + xmin) / 2.0
    center_y = (ymax + ymin) / 2.0
    width = xmax - xmin + 1.0
    height = ymax - ymin + 1.0

    return center_x, center_y, width, height

def coordinates_corners_to_centroids(
        center_x: np.ndarray[float],
        center_y: np.ndarray[float],
        width: np.ndarray[float],
        height: np.ndarray[float]
    ) -> tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
    """_summary_

    Args:
        center_x (np.ndarray[float]): center_x coordinates
        center_y (np.ndarray[float]): center_y  coordinates
        width (np.ndarray[float]): width coordinates
        height (np.ndarray[float]): height coordinates

    Returns:
        tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float]]: corners coordinates (xmin, ymin, xmax, ymax)
    """

    # calculate corners coordinates
    xmin = center_x - (width - 1.0) / 2.0
    ymin = center_y - (height - 1.0) / 2.0
    xmax = center_x + (width - 1.0) / 2.0
    ymax = center_y + (height - 1.0) / 2.0

    return xmin, ymin, xmax, ymax