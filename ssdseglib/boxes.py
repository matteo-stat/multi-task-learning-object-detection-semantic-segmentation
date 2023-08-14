import numpy as np
import math
from typing import Literal

class DefaultBoundingBoxes:
    def __init__(self,
            feature_maps_shapes: tuple[tuple[int, int], ...],
            feature_maps_aspect_ratios: tuple[float | int, ...] | tuple[tuple[float | int, ...], ...] = (1, 2, 3, 1/2, 1/3),
            boxes_scales: tuple[float, float] = (0.2, 0.9),
            centers_padding_from_borders: float = 0.5,
            additional_square_box: bool = True,                 
        ) -> None:
        """
        class for creating and managing default bounding boxes\n
        the default bounding boxes are created for each pixel in the given feature maps, as proposed by the single-shot-detector (ssd) object detection framework\n
        the coordinates for the default bounding boxes are calculated during the class initialization, normalized/scaled in the range [0, 1] and stored internally\n
        if you want to get the coordinates for the default bounding boxes scaled to a custom image shape, call the appropriate class method

        Args:
            feature_maps_shapes (tuple[tuple[int, int], ...]): a list of shapes for the feature maps on which you want to generate the default bounding boxes
            feature_maps_aspect_ratios (tuple[float | int, ...] | tuple[tuple[float | int, ...], ...], optional): a tuple of aspect ratios if all feature maps share the same aspect ratios,
            otherwise a tuple of tuples, where you specify different aspect ratios for each feature map. Defaults to (1, 2, 3, 1/2, 1/3).
            boxes_scales (tuple[float, float], optional): a tuple of minimum and maximum boxes scales,
            first feature map will have minimum boxes scale, last one maximum boxes scale, the middle ones have boxes scales linearly spaced. Defaults to (0.2, 0.9).
            centers_padding_from_borders (float, optional): padding/margin to keep from image borders when calculating the default bounding boxes centers. Defaults to 0.5.
            additional_square_box (bool, optional): boolean flag for the additional square box proposed by the ssd paper. Defaults to True.
        """

        # set attributes with arguments
        self.feature_maps_shapes = feature_maps_shapes
        self.centers_padding_from_borders = centers_padding_from_borders
        self.additional_square_box = additional_square_box

        # an additional scale it's calculated because it could be needed when calculating the additional square box in last feature map
        self.boxes_scales = np.linspace(boxes_scales[0], boxes_scales[1], len(self.feature_maps_shapes) + 1)

        # if arguments feature_maps_aspect_ratios it's a tuple of numbers then convert it to a tuple of tuples of numbers
        # this not an exhaustive validation.. but should be enough to avoid obvious mistakes
        if all(isinstance(item, float) or isinstance(item, int) for item in feature_maps_aspect_ratios):
            self.feature_maps_aspect_ratios = tuple(feature_maps_aspect_ratios for _ in range(len(feature_maps_shapes)))

        elif len(feature_maps_aspect_ratios) < len(feature_maps_shapes):
            raise ValueError('if you are passing a tuple of tuples of aspect ratios, then it should have same length as the tuple of feature maps shapes')
        
        else:
            self.feature_maps_aspect_ratios = feature_maps_aspect_ratios

        # store indexes for the various types of coordinates
        self._coordinates_indexes = {'xmin': 0, 'ymin': 1, 'xmax': 2, 'ymax': 3, 'center-x': 0, 'center-y': 1, 'width': 2, 'height': 3}

        # store default bounding boxes for each feature map, scaled between [0, 1]
        # it's a list with length equal to the number of feature maps
        # each element of the list it's a numpy.ndarray, with shape (feature map height, feature map width, number of default bounding boxes, 4 coordinates)
        # the coordinates are in the common corners format, so (xmin, ymin, xmax, ymax)
        self._feature_maps_boxes = self._generate_feature_maps_boxes()

        # store default bounding boxes for each feature map, scaled to a custom image shape
        # this attribute is set when the appropriate method it's called
        # same structure as _feature_maps_boxes attribute
        self.feature_maps_boxes = None
        
    def _generate_feature_maps_boxes(self) -> list[np.ndarray]:
        """
        generate default bounding boxes for each feature map\n
        for each pixel of each feature map, a number of len(aspect_ratios) or len(aspect_ratios)+1 default bounding boxes are generated

        Returns:
            list.[nd.array]:
                a list with length equal to the number of feature maps,
                where each element it's a numpy.ndarray, with shape (feature map height, feature map width, number of default bounding boxes, 4 coordinates)\n
                the coordinates are in the common corners format (xmin, ymin, xmax, ymax) and scaled between [0, 1]
        """
    
        # list to store boxes for each feature map
        feature_maps_boxes = []

        # calculate boxes for each feature map
        for feature_map_index, (feature_map_shape, aspect_ratios) in enumerate(zip(self.feature_maps_shapes, self.feature_maps_aspect_ratios)):
            
            # get the smallest side of the feature map shape
            feature_map_size = min(feature_map_shape)

            # get scales for current feature map and the next one
            boxes_scale_current = self.boxes_scales[feature_map_index]
            boxes_scale_next = self.boxes_scales[feature_map_index + 1]

            # calculate width and height for each aspect ratio
            # the output is a list of lists like [[width, height], ..., [width, height]], same length as given aspect ratios list
            boxes_width_height = [
                [feature_map_size * boxes_scale_current * math.sqrt(aspect_ratio), feature_map_size * boxes_scale_current / math.sqrt(aspect_ratio)]
                for aspect_ratio in aspect_ratios
            ]

            # optionally add an additional square box, with a different scale, as proposed by original ssd paper
            if self.additional_square_box:
                boxes_width_height.append([feature_map_size * math.sqrt(boxes_scale_current * boxes_scale_next), feature_map_size * math.sqrt(boxes_scale_current * boxes_scale_next)])

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
            # assign to the last dimension the 4 values xmin, ymin, xmax, ymax (normalized between 0 and 1)
            # note: pixels coordinates should be threated as indexes, be careful with +-1 operations
            boxes[:, :, :, self._coordinates_indexes['xmin']] = (np.tile(boxes_center_x, (1, 1, len(boxes_width_height))) - boxes_width_height[:, 0] / 2.0) / (feature_map_shape[1] - 1.0)
            boxes[:, :, :, self._coordinates_indexes['ymin']] = (np.tile(boxes_center_y, (1, 1, len(boxes_width_height))) - boxes_width_height[:, 1] / 2.0) / (feature_map_shape[0] - 1.0)
            boxes[:, :, :, self._coordinates_indexes['xmax']] = (np.tile(boxes_center_x, (1, 1, len(boxes_width_height))) + boxes_width_height[:, 0] / 2.0) / (feature_map_shape[1] - 1.0)
            boxes[:, :, :, self._coordinates_indexes['ymax']] = (np.tile(boxes_center_y, (1, 1, len(boxes_width_height))) + boxes_width_height[:, 1] / 2.0) / (feature_map_shape[0] - 1.0)

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
            feature_map_boxes[:, [self._coordinates_indexes['xmin'], self._coordinates_indexes['xmax']]] = feature_map_boxes[:, [self._coordinates_indexes['xmin'], self._coordinates_indexes['xmax']]] * image_shape[1]

            # scale height
            feature_map_boxes[:, [self._coordinates_indexes['ymin'], self._coordinates_indexes['ymax']]] = feature_map_boxes[:, [self._coordinates_indexes['ymin'], self._coordinates_indexes['ymax']]] * image_shape[0]

            # append to the default bounding boxes attribute
            self.feature_maps_boxes.append(feature_map_boxes)

    def _get_boxes_coordinates(self, coordinates_style: Literal['ssd', 'feature-maps'], coordinates_type: Literal['corners', 'xmin', 'ymin', 'xmax', 'ymax', 'centroids', 'center-x', 'center-y', 'width', 'height',]):
        
        if coordinates_type == 'corners' and coordinates_style == 'feature-maps':
            coordinates = self.feature_maps_boxes

        elif coordinates_type == 'corners' and coordinates_style == 'ssd':
            coordinates = np.concatenate([np.reshape(feature_map_boxes, newshape=(-1, 4)) for feature_map_boxes in self.feature_maps_boxes], axis=0)

        elif coordinates_type == 'centroids' and coordinates_style == 'feature-maps':
            coordinates = []
            for feature_map_boxes in self.feature_maps_boxes:
                # prepare output object
                centroids = np.empty_like(feature_map_boxes, dtype=np.float32)

                # center x and y
                centroids[:, [0, 1]] = (feature_map_boxes[:, [2, 3]] + feature_map_boxes[:, [0, 1]]) / 2.0

                # width and height
                centroids[:, [2, 3]] = feature_map_boxes[:, [2, 3]] - feature_map_boxes[:, [0, 1]] + 1.0
                
                coordinates.append(centroids)

        elif coordinates_type == 'centroids' and coordinates_style == 'ssd':
            corners = np.concatenate([np.reshape(feature_map_boxes, newshape=(-1, 4)) for feature_map_boxes in self.feature_maps_boxes], axis=0)
            # prepare output object
            centroids = np.empty_like(corners, dtype=np.float32)

            # center x and y
            centroids[:, [0, 1]] = (corners[:, [2, 3]] + corners[:, [0, 1]]) / 2.0

            # width and height
            centroids[:, [2, 3]] = corners[:, [2, 3]] - corners[:, [0, 1]] + 1.0          
            
        elif coordinates_type == 'corners':
            pass

        elif coordinates_type == 'centroids':
            pass

        return coordinates
 

        

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


