import numpy as np
import math
from typing import Literal

class DefaultBoundingBoxes:
    def __init__(self,
            feature_maps_shapes: tuple[tuple[int, int], ...],
            feature_maps_aspect_ratios: tuple[float | int, ...] | tuple[tuple[float | int, ...], ...] = (1.0, 2.0, 3.0, 1/2, 1/3),
            centers_padding_from_borders: float = 0.5,
            boxes_scales: tuple[float, float] | tuple[tuple[float, float], ...] = (0.2, 0.9),
            additional_square_box: bool = True,                 
        ) -> None:
        """
        class for creating and managing default bounding boxes\n
        the default bounding boxes are created for each pixel of the given feature maps, as proposed by the single-shot-detector (ssd) object detection framework\n
        the coordinates for the default bounding boxes are calculated during the class initialization and stored internally as coordinates scaled between 0 and 1\n
        if you want to get the coordinates for the default bounding boxes scaled/referred to a custom shape call the appropriate method of the class

        Args:
            feature_maps_shapes (tuple[tuple[int, int], ...]): a list of shapes for the feature maps on which we want to generate the default bounding boxes
            feature_maps_aspect_ratios (tuple[float | int, ...] | tuple[tuple[float | int, ...], ...], optional): a tuple of aspect ratios if all feature maps share the same aspect ratios,
            otherwise a tuple of tuples, where you specify different aspect ratios for each feature map. Defaults to (1.0, 2.0, 3.0, 1/2, 1/3).
            centers_padding_from_borders (float, optional): padding/margin to keep when calculating the centers of the default bounding boxes. Defaults to 0.5.
            boxes_scales (tuple[float, float] | tuple[tuple[float, float], ...], optional): minimum and maximum boxes scales. Defaults to (0.2, 0.9).
            additional_square_box (bool, optional): boolean flag for the additional square box proposed by the ssd paper. Defaults to True.
        """

        # set attributes - arguments values
        self.feature_maps_shapes = feature_maps_shapes
        self.feature_maps_aspect_ratios = feature_maps_aspect_ratios
        self.centers_padding_from_borders = centers_padding_from_borders
        self.boxes_scales = boxes_scales
        self.additional_square_box = additional_square_box

        # if arguments feature_maps_aspect_ratios it's a tuple of numbers then convert it to a tuple of tuples of numbers
        if all(isinstance(item, float) or isinstance(item, int) for item in feature_maps_aspect_ratios):
            self.feature_maps_aspect_ratios = tuple(feature_maps_aspect_ratios for _ in range(len(feature_maps_shapes)))        
        else:
            self.feature_maps_aspect_ratios = feature_maps_aspect_ratios

        # set attributes - default values
        self.feature_maps_boxes = None

        # set attributes - internal use
        # the coordinates are scaled between 0 and 1
        self._feature_maps_boxes = self._generate_feature_maps_boxes()
        
        # this is used to store indexes of the coordinates
        self._coordinates_indexes = {'xmin': 0, 'ymin': 1, 'xmax': 2, 'ymax': 3, 'center-x': 0, 'center-y': 1, 'width': 2, 'height': 2}
    
    def _generate_feature_maps_boxes(self) -> list[np.ndarray]:
        """
        generate default bounding boxes for each feature map and aspect ratio\n
        for each pixel of each feature map, a number of len(aspect_ratios) or len(aspect_ratios)+1 default bounding boxes are generated

        Returns:
            list.[nd.array]: 
                a list of multi-dimensional numpy arrays, each one with shape (feature map height, feature map width, number of default bounding boxes, 4 coordinates), 
                the coordinates are expressed between 0 and 1 for all the default bounding boxes of each feature map
        """    
    
        # list to store boxes for each feature map
        feature_maps_boxes = []

        # list of linearly spaced scales for boxes of different feature maps
        scales = np.linspace(self.boxes_scales[0], self.boxes_scales[1], len(self.feature_maps_shapes) + 1)

        # calculate boxes for each feature map
        for feature_map_index, (feature_map_shape, aspect_ratios) in enumerate(zip(self.feature_maps_shapes, self.feature_maps_aspect_ratios)):
            
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
            if self.additional_square_box:
                boxes_width_height.append([feature_map_size * math.sqrt(scale_current * scale_next), feature_map_size * math.sqrt(scale_current * scale_next)])

            # convert to numpy array
            boxes_width_height = np.array(boxes_width_height)

            # calculate centers coordinates for the boxes
            # there is a center for each pixel of the feature map
            boxes_center_x = np.linspace(start=self.centers_padding_from_borders, stop=feature_map_shape[1] - 1 - self.centers_padding_from_borders, num=feature_map_shape[1])
            boxes_center_y = np.linspace(start=self.centers_padding_from_borders, stop=feature_map_shape[0] - 1 - self.centers_padding_from_borders, num=feature_map_shape[0])

            # manipulate the arrays of centers to get outputs of shape (feature_map_shape_y, feature_map_shape_x, 1)
            boxes_center_x, boxes_center_y = np.meshgrid(boxes_center_x, boxes_center_y)
            boxes_center_x = np.expand_dims(boxes_center_x, axis=-1)
            boxes_center_y = np.expand_dims(boxes_center_y, axis=-1)
            
            # prepare final output array with shape (feature_map_shape_y, feature_map_shape_x, number_of_boxes, 4)
            # we end up with an array containing box coordinates for each aspect ratio (+1 optionally), for each pixel of the feature map
            boxes = np.zeros((feature_map_shape[0], feature_map_shape[1], len(boxes_width_height), 4), dtype=np.float32)

            # populate the output array
            # assign to the last dimension the 4 values x_min, y_min, x_max, y_max (normalized)
            # note: pixels coordinates should be threated as indexes, be careful with +-1 operations
            boxes[:, :, :, 0] = (np.tile(boxes_center_x, (1, 1, len(boxes_width_height))) - boxes_width_height[:, 0] / 2.0) / (feature_map_shape[1] - 1.0)
            boxes[:, :, :, 1] = (np.tile(boxes_center_y, (1, 1, len(boxes_width_height))) - boxes_width_height[:, 1] / 2.0) / (feature_map_shape[0] - 1.0)
            boxes[:, :, :, 2] = (np.tile(boxes_center_x, (1, 1, len(boxes_width_height))) + boxes_width_height[:, 0] / 2.0) / (feature_map_shape[1] - 1.0)
            boxes[:, :, :, 3] = (np.tile(boxes_center_y, (1, 1, len(boxes_width_height))) + boxes_width_height[:, 1] / 2.0) / (feature_map_shape[0] - 1.0)

            # append boxes calculated for the feature map
            feature_maps_boxes.append(boxes)

        return feature_maps_boxes

    def rescale_boxes_coordinates(self, image_shape: tuple[int, int]) -> None:
        """
        rescale the default bounding boxes coordinates to a given image shape

        Args:
            image_shape (tuple[int]): image shape where you want to visualize/show the default bounding boxes
        """

        # reset the default bounding boxes coordinates
        self.feature_maps_boxes = []

        # rescale the coordinates of the default bounding boxes for each feature map
        for feature_map_boxes in self._feature_maps_boxes:
            # scale width
            feature_map_boxes[:, [0, 2]] = feature_map_boxes[:, [0, 2]] * image_shape[1]

            # scale height
            feature_map_boxes[:, [1, 3]] = feature_map_boxes[:, [1, 3]] * image_shape[0]

            # append to the default bounding boxes attribute
            self.feature_maps_boxes.append(feature_map_boxes)

    def _get_coordinates(coordinate: Literal['xmin', 'ymin', 'xmax', 'ymax', 'center-x', 'center-y', 'width', 'height']):
        pass        

        

        # # set corners coordinates
        # self.xmin, self.ymin, self.xmax, self.ymax = np.split(self.boxes, 4, axis=-1)
        # self.xmin = self.xmin.reshape(-1)
        # self.ymin = self.ymin.reshape(-1)
        # self.xmax = self.xmax.reshape(-1)
        # self.ymax = self.ymax.reshape(-1)

        # # set centroids coordinates
        # self.center_x = (self.xmax + self.xmin) / 2.0
        # self.center_y = (self.ymax + self.ymin) / 2.0
        # self.width = self.xmax - self.xmin + 1.0
        # self.height = self.ymax - self.ymin + 1.0

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
    # note: pixels coordinates should be threated as indexes, be careful with +-1 operations
    center_x = (xmax + xmin) / 2.0
    center_y = (ymax + ymin) / 2.0
    width = xmax - xmin + 1.0
    height = ymax - ymin + 1.0

    return center_x, center_y, width, height

def coordinates_centroids_to_corners(
        center_x: np.ndarray[float],
        center_y: np.ndarray[float],
        width: np.ndarray[float],
        height: np.ndarray[float]
    ) -> tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
    """
    convert centroids coordinates to corners

    Args:
        center_x (np.ndarray[float]): center_x coordinates
        center_y (np.ndarray[float]): center_y coordinates
        width (np.ndarray[float]): width coordinates
        height (np.ndarray[float]): height coordinates

    Returns:
        tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float]]: corners coordinates (xmin, ymin, xmax, ymax)
    """

    # calculate corners coordinates
    # note: pixels coordinates should be threated as indexes, be careful with +-1 operations
    xmin = center_x - (width - 1.0) / 2.0
    ymin = center_y - (height - 1.0) / 2.0
    xmax = center_x + (width - 1.0) / 2.0
    ymax = center_y + (height - 1.0) / 2.0

    return xmin, ymin, xmax, ymax
