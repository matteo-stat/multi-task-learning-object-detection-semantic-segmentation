import numpy as np
import math
from typing import Literal, Tuple, List

class DefaultBoundingBoxes:
    def __init__(self,
            feature_maps_shapes: Tuple[Tuple[int, int], ...],
            feature_maps_aspect_ratios: Tuple[float | int, ...] | Tuple[Tuple[float | int, ...], ...] = (1, 2, 3, 1/2, 1/3),
            boxes_scales: Tuple[float, float] = (0.2, 0.9),
            centers_padding_from_borders_percentage: float = 0.05,
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
            centers_padding_from_borders_percentage (float, optional): padding/margin in percentage to keep from image borders when calculating the default bounding boxes centers. Defaults to 0.05.
            additional_square_box (bool, optional): boolean flag for the additional square box proposed by the ssd paper. Defaults to True.
        """

        # # due to this class implementation you can't 
        # if any(shape < 2 for shapes in feature_maps_shapes for shape in shapes):
        #     raise ValueError('due to this class implementation can generate default bounding boxes only on feature maps with shapes greater than 1')
        

        # set attributes with arguments        
        self.feature_maps_shapes = feature_maps_shapes
        if 0 <= centers_padding_from_borders_percentage < 0.5:
            self.centers_padding_from_borders_percentage = centers_padding_from_borders_percentage
        else:
            raise ValueError('the percentage padding from borders when calculating default bounding boxes centers must be between 0 and 0.5')
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
            feature_map_size = min(feature_map_shape)
            boxes_shapes = [
                [feature_map_size * boxes_scale_current / math.sqrt(aspect_ratio), feature_map_size * boxes_scale_current * math.sqrt(aspect_ratio)]
                for aspect_ratio in feature_map_aspect_ratios
            ]

            # optionally add an additional square box, with a different scale, as proposed by original ssd paper
            if self.additional_square_box:
                boxes_shapes.append([feature_map_size * math.sqrt(boxes_scale_current * boxes_scale_next), feature_map_size * math.sqrt(boxes_scale_current * boxes_scale_next)])

            # convert to numpy array
            boxes_shapes = np.array(boxes_shapes)

            # for each pixel of the feature map calculate centers coordinates for the boxes
            # note: pixels coordinates should be threated as image indexes, be careful with +-1 operations
            if feature_map_shape[0] == 1:
                boxes_center_y = np.array([0.5])
            else:
                centers_padding_from_borders_pixels = self.centers_padding_from_borders_percentage * (feature_map_shape[0] - 1.0)
                boxes_center_y = np.linspace(
                    start=centers_padding_from_borders_pixels,
                    stop=feature_map_shape[0] - 1.0 - centers_padding_from_borders_pixels,
                    num=feature_map_shape[0]
                )

            if feature_map_shape[1] == 1:
                boxes_center_x = np.array([0.5])
            else:
                centers_padding_from_borders_pixels = self.centers_padding_from_borders_percentage * (feature_map_shape[1] - 1.0)
                boxes_center_x = np.linspace(
                    start=centers_padding_from_borders_pixels,
                    stop=feature_map_shape[1] - 1.0 - centers_padding_from_borders_pixels,
                    num=feature_map_shape[1]
                )             

            # manipulate the arrays of centers to get outputs of shape (feature_map_shape_y, feature_map_shape_x, 1)
            boxes_center_x, boxes_center_y = np.meshgrid(boxes_center_x, boxes_center_y)
            boxes_center_x = np.expand_dims(boxes_center_x, axis=-1)
            boxes_center_y = np.expand_dims(boxes_center_y, axis=-1)
            
            # prepare final output array with shape (feature_map_shape_y, feature_map_shape_x, number_of_boxes, 4)
            # for each pixel in the feature map we will get corners coordinates (xmin, ymin, xmax, ymax) for all the default bounding boxes
            boxes = np.zeros((feature_map_shape[0], feature_map_shape[1], len(boxes_shapes), 4), dtype=np.float32)

            # populate output array (convert the centroids coordinates to corners)
            # assign to the last dimension the 4 values xmin, ymin, xmax, ymax (normalized between 0 and 1)
            normalization_factor_x = feature_map_shape[1] - 1.0 if feature_map_shape[1] > 1.0 else 1.0
            normalization_factor_y = feature_map_shape[0] - 1.0 if feature_map_shape[0] > 1.0 else 1.0
            boxes[:, :, :, self._coordinates_indexes['xmin']] = (np.tile(boxes_center_x, (1, 1, len(boxes_shapes))) - (boxes_shapes[:, 1] - 1.0) / 2.0) / normalization_factor_x
            boxes[:, :, :, self._coordinates_indexes['ymin']] = (np.tile(boxes_center_y, (1, 1, len(boxes_shapes))) - (boxes_shapes[:, 0] - 1.0) / 2.0) / normalization_factor_y
            boxes[:, :, :, self._coordinates_indexes['xmax']] = (np.tile(boxes_center_x, (1, 1, len(boxes_shapes))) + (boxes_shapes[:, 1] - 1.0) / 2.0) / normalization_factor_x
            boxes[:, :, :, self._coordinates_indexes['ymax']] = (np.tile(boxes_center_y, (1, 1, len(boxes_shapes))) + (boxes_shapes[:, 0] - 1.0) / 2.0) / normalization_factor_y

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
            boxes[:, :, :, [self._coordinates_indexes['xmin'], self._coordinates_indexes['xmax']]] = boxes[:, :, :, [self._coordinates_indexes['xmin'], self._coordinates_indexes['xmax']]] * image_shape[1]

            # scale height
            boxes[:, :, :, [self._coordinates_indexes['ymin'], self._coordinates_indexes['ymax']]] = boxes[:, :, :, [self._coordinates_indexes['ymin'], self._coordinates_indexes['ymax']]] * image_shape[0]

            # append to the default bounding boxes attribute
            self.feature_maps_boxes.append(boxes)

    def _get_boxes_coordinates_corners_all_or_single(
            self,
            coordinates_type: Literal['xmin', 'ymin', 'xmax', 'ymax', 'corners'],
            coordinates_style: Literal['ssd', 'feature-maps']
        ) -> Tuple[np.ndarray] | np.ndarray:
        """
        internal method that return the requested corners coordinates        

        Args:
            coordinates_type (Literal['xmin', 'ymin', 'xmax', 'ymax', 'corners']): pass 'corners' to get all the corners coordinates,
            otherwise pass a valid arguments to get the specific corners coordinates type
            coordinates_style (Literal['ssd', 'feature-maps'): if 'ssd' then return the default bounding boxes with a shape similar to the object detection output from a network using the single-shot-detector framework,
            if 'feature-map' then return a list containing default bounding boxes for each feature map

        Returns:
            Tuple[np.ndarray] | np.ndarray: default bounding boxes as corners coordinates
        """
        # retrieve the requested coordinates using the proper indexes
        coordinates_indexes = (self._coordinates_indexes[coordinates_type], ) if isinstance(self._coordinates_indexes[coordinates_type], int) else self._coordinates_indexes[coordinates_type]
        coordinates = tuple(boxes[:, :, :, coordinates_indexes] for boxes in self.feature_maps_boxes)

        # if the requested coordinates style it's single-shot-detector (ssd) then concatenate all the coordinates
        # this return the coordinates with a shape like the detection output from a network using ssd framework for object detection
        if coordinates_style == 'ssd':
            coordinates_shape = (-1, 4) if coordinates_type == 'corners' else (-1, )
            coordinates = [np.reshape(item, newshape=coordinates_shape) for item in coordinates]
            coordinates = np.concatenate(coordinates, axis=0)

        return coordinates

    def get_boxes_coordinates_corners(self, coordinates_style: Literal['ssd', 'feature-maps']) -> Tuple[np.ndarray] | np.ndarray:
        """
        return all the default bounding boxes as corners coordinates, with the specified style        

        Args:
            coordinates_style (Literal['ssd', 'feature-maps'): if 'ssd' then return the default bounding boxes with a shape similar to the object detection output from a network using the single-shot-detector framework,
            if 'feature-map' then return a list containing default bounding boxes for each feature map

        Returns:
            Tuple[np.ndarray] | np.ndarray: default bounding boxes as corners coordinates
        """                
        return self._get_boxes_coordinates_corners_all_or_single(coordinates_type='corners', coordinates_style=coordinates_style)
    
    def get_boxes_coordinates_xmin(self, coordinates_style: Literal['ssd', 'feature-maps']) -> Tuple[np.ndarray] | np.ndarray:
        """
        return all the default bounding boxes xmin corners coordinates, with the specified style        

        Args:
            coordinates_style (Literal['ssd', 'feature-maps'): if 'ssd' then return the default bounding boxes with a shape similar to the object detection output from a network using the single-shot-detector framework,
            if 'feature-map' then return a list containing default bounding boxes for each feature map

        Returns:
            Tuple[np.ndarray] | np.ndarray: default bounding boxes xmin corners coordinates
        """          
        return self._get_boxes_coordinates_corners_all_or_single(coordinates_type='xmin', coordinates_style=coordinates_style)

    def get_boxes_coordinates_ymin(self, coordinates_style: Literal['ssd', 'feature-maps']) -> Tuple[np.ndarray] | np.ndarray:
        """
        return all the default bounding boxes ymin corners coordinates, with the specified style        

        Args:
            coordinates_style (Literal['ssd', 'feature-maps'): if 'ssd' then return the default bounding boxes with a shape similar to the object detection output from a network using the single-shot-detector framework,
            if 'feature-map' then return a list containing default bounding boxes for each feature map

        Returns:
            Tuple[np.ndarray] | np.ndarray: default bounding boxes ymin corners coordinates
        """ 
        return self._get_boxes_coordinates_corners_all_or_single(coordinates_type='ymin', coordinates_style=coordinates_style)
    
    def get_boxes_coordinates_xmax(self, coordinates_style: Literal['ssd', 'feature-maps']) -> Tuple[np.ndarray] | np.ndarray:
        """
        return all the default bounding boxes xmax corners coordinates, with the specified style        

        Args:
            coordinates_style (Literal['ssd', 'feature-maps'): if 'ssd' then return the default bounding boxes with a shape similar to the object detection output from a network using the single-shot-detector framework,
            if 'feature-map' then return a list containing default bounding boxes for each feature map

        Returns:
            Tuple[np.ndarray] | np.ndarray: default bounding boxes xmax corners coordinates
        """         
        return self._get_boxes_coordinates_corners_all_or_single(coordinates_type='xmax', coordinates_style=coordinates_style)

    def get_boxes_coordinates_ymax(self, coordinates_style: Literal['ssd', 'feature-maps']) -> Tuple[np.ndarray] | np.ndarray:
        """
        return all the default bounding boxes ymax corners coordinates, with the specified style        

        Args:
            coordinates_style (Literal['ssd', 'feature-maps'): if 'ssd' then return the default bounding boxes with a shape similar to the object detection output from a network using the single-shot-detector framework,
            if 'feature-map' then return a list containing default bounding boxes for each feature map

        Returns:
            Tuple[np.ndarray] | np.ndarray: default bounding boxes ymax corners coordinates
        """         
        return self._get_boxes_coordinates_corners_all_or_single(coordinates_type='ymax', coordinates_style=coordinates_style)
    
    def _get_boxes_coordinates_centroids_all_or_single(
            self,
            coordinates_type: Literal['center-x', 'center-y', 'width', 'height', 'centroids'],
            coordinates_style: Literal['ssd', 'feature-maps']
        ) -> Tuple[np.ndarray] | np.ndarray:
        """
        internal method that return the requested centroids coordinates        

        Args:
            coordinates_type (Literal['center-x', 'center-y', 'width', 'height', 'centroids']): pass 'centroids' to get all the centroids coordinates,
            otherwise pass a valid arguments to get the specific centroids coordinates type
            coordinates_style (Literal['ssd', 'feature-maps'): if 'ssd' then return the default bounding boxes as centroids coordinates a shape similar to the object detection output from a network using the single-shot-detector framework,
            if 'feature-map' then return a list containing boxes centroids coordinates for each feature map

        Returns:
            Tuple[np.ndarray] | np.ndarray: default bounding boxes as centroids coordinates
        """        

        # get coordinates with feature-maps style and corners type
        boxes_corners = self._get_boxes_coordinates_corners_all_or_single(coordinates_type='corners', coordinates_style='feature-maps')
        
        # initialize the list for store the centroids coordinates
        boxes_centroids = []

        # for each feature map convert the corners coordinates to centroids coordinates
        for i in range(len(boxes_corners)):
            # split the coordinates for easier computation
            xmin, ymin, xmax, ymax = np.split(boxes_corners[i], 4, axis=3)

            # calculate centroids coordinates
            center_x = (xmax + xmin) / 2.0
            center_y = (ymax + ymin) / 2.0
            width = xmax - xmin + 1.0
            height = ymax - ymin + 1.0
            
            # concatenate the centroids coordinates and add them to the list
            boxes_centroids.append(np.concatenate((center_x, center_y, width, height), axis=3))

        # retrieve the requested coordinates using the proper indexes
        coordinates_indexes = (self._coordinates_indexes[coordinates_type], ) if isinstance(self._coordinates_indexes[coordinates_type], int) else self._coordinates_indexes[coordinates_type]
        coordinates = tuple(boxes[:, :, :, coordinates_indexes] for boxes in boxes_centroids)

        # if the requested coordinates style it's single-shot-detector (ssd) then concatenate all the coordinates
        # this return the coordinates with a shape like the detection output from a network using ssd framework for object detection
        if coordinates_style == 'ssd':
            coordinates_shape = (-1, 4) if coordinates_type == 'centroids' else (-1, )
            coordinates = [np.reshape(item, newshape=coordinates_shape) for item in coordinates]
            coordinates = np.concatenate(coordinates, axis=0)

        return coordinates

    def get_boxes_coordinates_centroids(self, coordinates_style: Literal['ssd', 'feature-maps']) -> Tuple[np.ndarray]:        
        """
        return all the default bounding boxes as centroids coordinates, with the specified style        

        Args:
            coordinates_style (Literal['ssd', 'feature-maps'): if 'ssd' then return the default bounding boxes as centroids coordinates a shape similar to the object detection output from a network using the single-shot-detector framework,
            if 'feature-map' then return a list containing boxes centroids coordinates for each feature map

        Returns:
            Tuple[np.ndarray] | np.ndarray: default bounding boxes as centroids coordinates
        """         
        return self._get_boxes_coordinates_centroids_all_or_single(coordinates_type='centroids', coordinates_style=coordinates_style)
    
    def get_boxes_coordinates_center_x(self, coordinates_style: Literal['ssd', 'feature-maps']) -> Tuple[np.ndarray] | np.ndarray:
        """
        return all the default bounding boxes center-x centroids coordinates, with the specified style        

        Args:
            coordinates_style (Literal['ssd', 'feature-maps'): if 'ssd' then return the default bounding boxes as centroids coordinates a shape similar to the object detection output from a network using the single-shot-detector framework,
            if 'feature-map' then return a list containing boxes centroids coordinates for each feature map

        Returns:
            Tuple[np.ndarray] | np.ndarray: default bounding boxes as center-x centroids coordinates
        """         
        return self._get_boxes_coordinates_centroids_all_or_single(coordinates_type='center-x', coordinates_style=coordinates_style)

    def get_boxes_coordinates_center_y(self, coordinates_style: Literal['ssd', 'feature-maps']) -> Tuple[np.ndarray] | np.ndarray:
        """
        return all the default bounding boxes center-y centroids coordinates, with the specified style        

        Args:
            coordinates_style (Literal['ssd', 'feature-maps'): if 'ssd' then return the default bounding boxes as centroids coordinates a shape similar to the object detection output from a network using the single-shot-detector framework,
            if 'feature-map' then return a list containing boxes centroids coordinates for each feature map

        Returns:
            Tuple[np.ndarray] | np.ndarray: default bounding boxes as center-y centroids coordinates
        """      
        return self._get_boxes_coordinates_centroids_all_or_single(coordinates_type='center-y', coordinates_style=coordinates_style)
    
    def get_boxes_coordinates_width(self, coordinates_style: Literal['ssd', 'feature-maps']) -> Tuple[np.ndarray] | np.ndarray:
        """
        return all the default bounding boxes width centroids coordinates, with the specified style        

        Args:
            coordinates_style (Literal['ssd', 'feature-maps'): if 'ssd' then return the default bounding boxes as centroids coordinates a shape similar to the object detection output from a network using the single-shot-detector framework,
            if 'feature-map' then return a list containing boxes centroids coordinates for each feature map

        Returns:
            Tuple[np.ndarray] | np.ndarray: default bounding boxes as width centroids coordinates
        """      
        return self._get_boxes_coordinates_centroids_all_or_single(coordinates_type='width', coordinates_style=coordinates_style)

    def get_boxes_coordinates_height(self, coordinates_style: Literal['ssd', 'feature-maps']) -> Tuple[np.ndarray] | np.ndarray:
        """
        return all the default bounding boxes height centroids coordinates, with the specified style        

        Args:
            coordinates_style (Literal['ssd', 'feature-maps'): if 'ssd' then return the default bounding boxes as centroids coordinates a shape similar to the object detection output from a network using the single-shot-detector framework,
            if 'feature-map' then return a list containing boxes centroids coordinates for each feature map

        Returns:
            Tuple[np.ndarray] | np.ndarray: default bounding boxes as height centroids coordinates
        """      
        return self._get_boxes_coordinates_centroids_all_or_single(coordinates_type='height', coordinates_style=coordinates_style)
            

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


