import numpy as np
from typing import Literal, Tuple, List

class DefaultBoundingBoxes:
    def __init__(self,
            feature_maps_shapes: Tuple[Tuple[int, int], ...],
            feature_maps_aspect_ratios: Tuple[float | int, ...] | Tuple[Tuple[float | int, ...], ...] = (1, 2, 3, 1/2, 1/3),
            boxes_scales: Tuple[float, float] = (0.2, 0.9),
            centers_padding_from_borders: float = 0.5,
            additional_square_box: bool = True,                 
        ) -> None:
        """
        class for creating and managing default bounding boxes\n
        the default bounding boxes are created for each pixel in the given feature maps, as proposed by the single-shot-detector (ssd) object detection framework\n
        the coordinates for the default bounding boxes are calculated during the class initialization, normalized/scaled in the range [0, 1] and stored internally\n
        if you want to get the coordinates for the default bounding boxes scaled to a custom image shape, call the appropriate class method

        Args:
            feature_maps_shapes (Tuple[Tuple[int, int], ...]): a list of shapes for the feature maps on which you want to generate the default bounding boxes
            feature_maps_aspect_ratios (Tuple[float | int, ...] | Tuple[Tuple[float | int, ...], ...], optional): a tuple of aspect ratios if all feature maps share the same aspect ratios,
            otherwise a Tuple of tuples, where you specify different aspect ratios for each feature map (aspect ratios are intended as width:height). Defaults to (1, 2, 3, 1/2, 1/3).
            boxes_scales (Tuple[float, float], optional): a tuple of minimum and maximum boxes scales,
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
        self._coordinates_indexes = {'xmin': 0, 'ymin': 1, 'xmax': 2, 'ymax': 3, 'center-x': 0, 'center-y': 1, 'width': 2, 'height': 3, 'corners': [0, 1, 2, 3], 'centroids': [0, 1, 2, 3]}

        # store default bounding boxes for each feature map, scaled between [0, 1]
        # it's a list with length equal to the number of feature maps
        # each element of the list it's a numpy.ndarray, with shape (feature map height, feature map width, number of default bounding boxes, 4 coordinates)
        # the coordinates are in the common corners format, so (xmin, ymin, xmax, ymax)
        self._feature_maps_boxes = self._generate_feature_maps_boxes()

        # store default bounding boxes for each feature map, scaled to a custom image shape
        # this attribute is set when the appropriate method it's called
        # same structure as _feature_maps_boxes attribute
        self.feature_maps_boxes = None
        
    def _generate_feature_maps_boxes(self) -> List[np.ndarray]:
        """
        generate default bounding boxes for each feature map\n
        for each pixel of each feature map, a number of len(aspect_ratios) or len(aspect_ratios)+1 default bounding boxes are generated

        Returns:
            List[np.ndarray]:
                a list with length equal to the number of feature maps,
                where each element it's a numpy.ndarray, with shape (feature map height, feature map width, number of default bounding boxes, 4 coordinates)\n
                the coordinates are in the common corners format (xmin, ymin, xmax, ymax) and scaled between [0, 1]
        """
        # list to store boxes for each feature map
        feature_maps_boxes = []

        # calculate boxes for each feature map
        for feature_map_index, (feature_map_shape, feature_map_aspect_ratios) in enumerate(zip(self.feature_maps_shapes, self.feature_maps_aspect_ratios)):            

            # get scales for current feature map and the next one
            boxes_scale_current = self.boxes_scales[feature_map_index]
            boxes_scale_next = self.boxes_scales[feature_map_index + 1]

            # calculate boxes shapes for each aspect ratio
            # the output is a list of lists like [[height, width], ..., [height, width]]        
            boxes_shapes = [
                [feature_map_shape[1] * boxes_scale_current / aspect_ratio, feature_map_shape[1] * boxes_scale_current]
                for aspect_ratio in feature_map_aspect_ratios
            ]                

            # optionally add an additional square box, with a different scale, as proposed by the original ssd paper
            if self.additional_square_box:
                boxes_shapes.append([feature_map_shape[1] * boxes_scale_next, feature_map_shape[1] * boxes_scale_next])

            # convert to numpy array
            boxes_shapes = np.array(boxes_shapes)

            # for each pixel of the feature map calculate centers coordinates for the boxes
            # note: pixels coordinates should be threated as image indexes, be careful with +-1 operations
            boxes_center_x = np.linspace(start=self.centers_padding_from_borders, stop=feature_map_shape[1] - 1.0 - self.centers_padding_from_borders, num=feature_map_shape[1])
            boxes_center_y = np.linspace(start=self.centers_padding_from_borders, stop=feature_map_shape[0] - 1.0 - self.centers_padding_from_borders, num=feature_map_shape[0])

            # manipulate the arrays of centers to get outputs of shape (feature_map_shape_y, feature_map_shape_x, 1)
            boxes_center_x, boxes_center_y = np.meshgrid(boxes_center_x, boxes_center_y)
            boxes_center_x = np.expand_dims(boxes_center_x, axis=-1)
            boxes_center_y = np.expand_dims(boxes_center_y, axis=-1)
            
            # prepare final output array with shape (feature_map_shape_y, feature_map_shape_x, number_of_boxes, 4)
            # for each pixel in the feature map we will get corners coordinates (xmin, ymin, xmax, ymax) for all the default bounding boxes
            boxes = np.zeros((feature_map_shape[0], feature_map_shape[1], len(boxes_shapes), 4), dtype=np.float32)

            # populate output array
            # assign to the last dimension the 4 values xmin, ymin, xmax, ymax (normalized between 0 and 1)
            boxes[:, :, :, self._coordinates_indexes['xmin']] = (np.tile(boxes_center_x, (1, 1, len(boxes_shapes))) - (boxes_shapes[:, 1] - 1.0) / 2.0) / feature_map_shape[1]
            boxes[:, :, :, self._coordinates_indexes['ymin']] = (np.tile(boxes_center_y, (1, 1, len(boxes_shapes))) - (boxes_shapes[:, 0] - 1.0) / 2.0) / feature_map_shape[0]
            boxes[:, :, :, self._coordinates_indexes['xmax']] = (np.tile(boxes_center_x, (1, 1, len(boxes_shapes))) + (boxes_shapes[:, 1] - 1.0) / 2.0) / feature_map_shape[1]
            boxes[:, :, :, self._coordinates_indexes['ymax']] = (np.tile(boxes_center_y, (1, 1, len(boxes_shapes))) + (boxes_shapes[:, 0] - 1.0) / 2.0) / feature_map_shape[0]

            # append boxes calculated for the feature map
            feature_maps_boxes.append(boxes)

        return feature_maps_boxes

    def rescale_boxes_coordinates(self, image_shape: Tuple[int, int]) -> None:
        """
        rescale the default bounding boxes coordinates to a given image shape

        Args:
            image_shape (tuple[int]): image shape where you want to visualize/show the default bounding boxes
        """
        # reset the default bounding boxes coordinates
        self.feature_maps_boxes = []
        feature_maps_boxes = self._feature_maps_boxes     

        # rescale the coordinates of the default bounding boxes for each feature map
        for boxes in feature_maps_boxes:
            # scale width
            boxes[:, [self._coordinates_indexes['xmin'], self._coordinates_indexes['xmax']]] = boxes[:, [self._coordinates_indexes['xmin'], self._coordinates_indexes['xmax']]] * image_shape[1]

            # scale height
            boxes[:, [self._coordinates_indexes['ymin'], self._coordinates_indexes['ymax']]] = boxes[:, [self._coordinates_indexes['ymin'], self._coordinates_indexes['ymax']]] * image_shape[0]

            # append to the default bounding boxes attribute
            self.feature_maps_boxes.append(boxes)

    def _get_boxes_coordinates_corners_all_or_single(
            self,
            coordinates_type: Literal['xmin', 'ymin', 'xmax', 'ymax', 'corners'],
            coordinates_style: Literal['ssd', 'feature-maps']
        ) -> Tuple[np.ndarray] | np.ndarray:

        # retrieve the coordinates using the proper indexes
        coordinates_indexes = (self._coordinates_indexes[coordinates_type], ) if isinstance(self._coordinates_indexes[coordinates_type], int) else self._coordinates_indexes[coordinates_type]
        coordinates = tuple(boxes[:, :, :, coordinates_indexes] for boxes in self.feature_maps_boxes)

        # if the requested coordinates style it's single-shot-detector (ssd) then concatenate all the coordinates
        # this return the coordinates with a shape like the detection output from a network using ssd framework for object detection
        if coordinates_style == 'ssd':
            coordinates_shape = (-1, 4) if coordinates_type == 'corners' else (-1, )
            coordinates = [np.reshape(item, newshape=coordinates_shape) for item in coordinates]
            coordinates = np.concatenate(coordinates, axis=0)

        return coordinates

    def get_boxes_coordinates_corners(self, coordinates_style: Literal['ssd', 'feature-maps']) -> Tuple[np.ndarray]:        
        return self._get_boxes_coordinates_corners_all_or_single(coordinates_type='corners', coordinates_style=coordinates_style)
    
    def get_boxes_coordinates_xmin(self, coordinates_style: Literal['ssd', 'feature-maps']) -> np.ndarray:
        return self._get_boxes_coordinates_corners_all_or_single(coordinates_type='xmin', coordinates_style=coordinates_style)

    def get_boxes_coordinates_ymin(self, coordinates_style: Literal['ssd', 'feature-maps']) -> np.ndarray:
        return self._get_boxes_coordinates_corners_all_or_single(coordinates_type='ymin', coordinates_style=coordinates_style)
    
    def get_boxes_coordinates_xmax(self, coordinates_style: Literal['ssd', 'feature-maps']) -> np.ndarray:
        return self._get_boxes_coordinates_corners_all_or_single(coordinates_type='xmax', coordinates_style=coordinates_style)

    def get_boxes_coordinates_ymax(self, coordinates_style: Literal['ssd', 'feature-maps']) -> np.ndarray:
        return self._get_boxes_coordinates_corners_all_or_single(coordinates_type='ymax', coordinates_style=coordinates_style)
    

    def _get_boxes_coordinates_centroids_all_or_single(
            self,
            coordinates_type: Literal['xmin', 'ymin', 'xmax', 'ymax', 'corners'],
            coordinates_style: Literal['ssd', 'feature-maps']
        ):

        corners = self._get_boxes_coordinates_corners_all_or_single(coordinates_type='corners', coordinates_style=coordinates_style)


        return None

    def get_boxes_coordinates_centroids(self, coordinates_style: Literal['ssd', 'feature-maps']) -> Tuple[np.ndarray]:
        return None
            


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


