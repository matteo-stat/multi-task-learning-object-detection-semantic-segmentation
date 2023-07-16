from numpy import ndarray
import tensorflow as tf

class DataReaderEncoder:
    def __init__(
            self,
            num_classes: int,
            xmin_boxes_default: ndarray[float],
            ymin_boxes_default: ndarray[float],
            xmax_boxes_default: ndarray[float],
            ymax_boxes_default: ndarray[float],
            iou_threshold: float = 0.5,
            offsets_std: tuple[float] = (0.1, 0.1, 0.2, 0.2)
        ) -> None:
        """
        class for read and encode data, designed to work with tensorflow data/input pipelines

        Args:
            num_classes (int): number of classes for object detection and segmentation problem, including background
            xmin_boxes_default (ndarray[float]): array of coordinates for xmin (corners coordinates)
            ymin_boxes_default (ndarray[float]): array of coordinates for ymin (corners coordinates)
            xmax_boxes_default (ndarray[float]): array of coordinates for xmax (corners coordinates)
            ymax_boxes_default (ndarray[float]): array of coordinates for ymax (corners coordinates)
            iou_threshold (float, optional): minimum intersection over union threshold with ground truth boxes to consider a default bounding box not background. Defaults to 0.5.
            offsets_std (tuple[float], optional): offsets standard deviation between ground truth and default bounding boxes, expected as (offsets_center_x_std, offsets_center_y_std, offsets_width_std, offsets_height_std). Defaults to (0.1, 0.1, 0.2, 0.2).
        """

        # set attributes
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.offsets_center_x_std, self.offsets_center_y_std, self.offsets_width_std, self.offsets_height_std = offsets_std
        self.xmin_boxes_default = tf.convert_to_tensor(xmin_boxes_default, dtype=tf.float32)
        self.ymin_boxes_default = tf.convert_to_tensor(ymin_boxes_default, dtype=tf.float32)
        self.xmax_boxes_default = tf.convert_to_tensor(xmax_boxes_default, dtype=tf.float32)
        self.ymax_boxes_default = tf.convert_to_tensor(ymax_boxes_default, dtype=tf.float32)

        # calculate area for default bounding boxes
        self.area_boxes_default = tf.expand_dims(
            input=(self.ymax_boxes_default - self.ymin_boxes_default + 1.0) * (self.xmax_boxes_default - self.xmin_boxes_default + 1.0),
            axis=1
        )


    def _coordinates_corners_to_centroids(
            self,
            xmin: tf.Tensor,
            ymin: tf.Tensor,
            xmax: tf.Tensor,
            ymax: tf.Tensor
        ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        convert corners coordinates to centroids

        Args:
            xmin (tf.Tensor): xmin coordinates
            ymin (tf.Tensor): ymin coordinates
            xmax (tf.Tensor): xmax coordinates
            ymax (tf.Tensor): ymax coordinates

        Returns:
            tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]: centroids coordinates (center_x, center_y, width, height)
        """

        # calculate centroids coordinates
        center_x = (xmax + xmin) / 2.0
        center_y = (ymax + ymin) / 2.0
        width = xmax - xmin + 1.0
        height = ymax - ymin + 1.0

        return center_x, center_y, width, height


    def coordinates_centroids_to_corners(
            self,
            center_x: tf.Tensor,
            center_y: tf.Tensor,
            width: tf.Tensor,
            height: tf.Tensor
        ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        convert centroids coordinates to corners        

        Args:
            center_x (tf.Tensor): center_x coordinates
            center_y (tf.Tensor): center_y  coordinates
            width (tf.Tensor): width coordinates
            height (tf.Tensor): height coordinates

        Returns:
            tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]: corners coordinates (xmin, ymin, xmax, ymax)
        """

        # calculate corners coordinates
        xmin = center_x - (width - 1.0) / 2.0
        ymin = center_y - (height - 1.0) / 2.0
        xmax = center_x + (width - 1.0) / 2.0
        ymax = center_y + (height - 1.0) / 2.0

        return xmin, ymin, xmax, ymax

    
    def _encode_ground_truth_labels_boxes(self, path_labels_boxes: str) -> tf.Tensor:
        """
        encode ground truth data as required by a ssd network        

        Args:
            path_labels_boxes (str): path and filename for ground truth labels and boxes

        Returns:
            tf.Tensor:
                encoded data, a tensor with shape (total default boxes, num classes + 4)
                last axis contains labels one hot encoded and offsets between ground truth and default bounding boxes
                note that offsets are calculated from centroids coordinates (offset_center_x, offset_center_y, offset_width, offset_height)
        """
        
        # read labels boxes csv file as text, split text by lines and then decode csv data to tensors
        labels_boxes = tf.strings.strip(tf.io.read_file(path_labels_boxes))
        labels_boxes = tf.strings.split(labels_boxes, sep='\r\n')
        labels_boxes = tf.io.decode_csv(labels_boxes, record_defaults=[int(), float(), float(), float(), float()])

        # create ground truth labels and boxes tensors
        labels_ground_truth, xmin_boxes_ground_truth, ymin_boxes_ground_truth, xmax_boxes_ground_truth, ymax_boxes_ground_truth = labels_boxes

        # calculate area for ground truth bounding boxes
        area_boxes_ground_truth = (xmax_boxes_ground_truth - xmin_boxes_ground_truth + 1.0) * (ymax_boxes_ground_truth - ymin_boxes_ground_truth + 1.0)

        # coordinates of intersections between each default bounding box and all ground truth bounding boxes
        # it selects the maximum for xmin and ymin coordinates, the minimum for xmax and ymax coordinates
        xmin_boxes_intersection = tf.maximum(tf.expand_dims(self.xmin_boxes_default, axis=1), tf.transpose(xmin_boxes_ground_truth))
        ymin_boxes_intersection = tf.maximum(tf.expand_dims(self.ymin_boxes_default, axis=1), tf.transpose(ymin_boxes_ground_truth))
        xmax_boxes_intersection = tf.minimum(tf.expand_dims(self.xmax_boxes_default, axis=1), tf.transpose(xmax_boxes_ground_truth))
        ymax_boxes_intersection = tf.minimum(tf.expand_dims(self.ymax_boxes_default, axis=1), tf.transpose(ymax_boxes_ground_truth))

        # area of intersection between each default bounding box and all ground truth bounding boxes
        area_boxes_intersection = tf.maximum(0.0, xmax_boxes_intersection - xmin_boxes_intersection + 1.0) * tf.maximum(0.0, ymax_boxes_intersection - ymin_boxes_intersection + 1.0)

        # calculate intersection over union between each default bounding box and all ground truth bounding boxes
        # note that this is a matrix with shape (num default bounding boxes, num ground truth bounding boxes)
        iou = area_boxes_intersection / (self.area_boxes_default + tf.transpose(area_boxes_ground_truth) - area_boxes_intersection)

        # find best matches between ground truth and default bounding boxes with 3 steps, following original ssd paper suggestion
        # first one find a match for each ground truth box
        # second one find a match for each default box
        # third one put together results from previous steps, removing possible duplicates coming from the union operation

        # step 1 - find the best match between each ground truth box and all default bounding boxes
        # note that the output shape will be (num ground truth boxes with iou > 0 with at least one default box, 2)
        # this matrix-like tensor contains indexes for default boxes and ground truth boxes
        indexes_match_ground_truth = tf.stack([tf.math.argmax(iou, axis=0, output_type=tf.dtypes.int32), tf.range(len(xmin_boxes_ground_truth))], axis=1)
        indexes_match_ground_truth = tf.boolean_mask(tensor=indexes_match_ground_truth, mask=tf.math.greater(tf.reduce_max(iou, axis=0), 0.0), axis=0)

        # step 2 - find the best match between each default box and all ground truth bounding boxes
        # note that the output shape will be (num default truth boxes with iou > threshold with at least one ground truth box, 2)
        # this matrix-like tensor contains indexes for default boxes, ground truth boxes
        indexes_match_default = tf.stack([tf.range(len(self.xmin_boxes_default)), tf.math.argmax(iou, axis=1, output_type=tf.dtypes.int32)], axis=1)
        indexes_match_default = tf.boolean_mask(
            tensor=indexes_match_default,
            mask=tf.math.greater(tf.reduce_max(iou, axis=1), self.iou_threshold),
            axis=0
        )

        # step 3 - put all best matches together, removing possible duplicates
        indexes_match, _ = tf.raw_ops.UniqueV2(x=tf.concat([indexes_match_ground_truth, indexes_match_default], axis=0), axis=[0])

        # get the class labels for each best match and one-hot encode them (0 is reserved for the background class)
        labels_match = tf.gather(labels_ground_truth, indexes_match[:, 1])
        labels_match = tf.one_hot(labels_match, self.num_classes)

        # convert all bounding boxes coordinates from corners to centroids
        centroids_default_center_x, centroids_default_center_y, centroids_default_width, centroids_default_height = self._coordinates_corners_to_centroids(
            xmin=tf.gather(self.xmin_boxes_default, indexes_match[:, 0]),
            ymin=tf.gather(self.ymin_boxes_default, indexes_match[:, 0]),
            xmax=tf.gather(self.xmax_boxes_default, indexes_match[:, 0]),
            ymax=tf.gather(self.ymax_boxes_default, indexes_match[:, 0]),
        )
        centroids_ground_truth_center_x, centroids_ground_truth_center_y, centroids_ground_truth_width, centroids_ground_truth_height = self._coordinates_corners_to_centroids(
            xmin=tf.gather(xmin_boxes_ground_truth, indexes_match[:, 1]),
            ymin=tf.gather(ymin_boxes_ground_truth, indexes_match[:, 1]),
            xmax=tf.gather(xmax_boxes_ground_truth, indexes_match[:, 1]),
            ymax=tf.gather(ymax_boxes_ground_truth, indexes_match[:, 1]),
        )

        # calculate centroids offsets between ground truth and default boxes and standardize them
        # for standardization we are assuming that the mean zero and standard deviation given as input
        offsets_center_x = (centroids_ground_truth_center_x - centroids_default_center_x) / centroids_default_width / self.offsets_center_x_std
        offsets_center_y = (centroids_ground_truth_center_y - centroids_default_center_y) / centroids_default_height / self.offsets_center_y_std
        offsets_width = tf.math.log(centroids_ground_truth_width / centroids_default_width) / self.offsets_width_std
        offsets_height = tf.math.log(centroids_ground_truth_height / centroids_default_height) / self.offsets_height_std
        
        # default bounding boxes encoded as required (one-hot encoding for classes, offsets for centroids coordinates)
        # if a default bounding box was matched with ground truth, then labels and offsets centroids coordinates are calculated
        # otherwise to the default bounding box, background label and zero offsets centroid coordinates are assigned      
        boxes_encoded = tf.zeros(shape=(len(self.xmin_boxes_default), self.num_classes + 4), dtype=tf.float32)
        boxes_encoded = tf.tensor_scatter_nd_update(
            tensor=boxes_encoded,
            indices=tf.expand_dims(indexes_match[:, 0], axis=1),
            updates=tf.concat(
                values=[
                    labels_match,
                    tf.expand_dims(offsets_center_x, axis=1),
                    tf.expand_dims(offsets_center_y, axis=1),
                    tf.expand_dims(offsets_height, axis=1),
                    tf.expand_dims(offsets_width, axis=1)
                ],
                axis=1)
        )
    
        return boxes_encoded


    def read_encode(
            self,
            path_image: str,
            path_mask: str,
            path_labels_boxes: str,
        ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        read and encode ground truth data

        Args:
            path_image (str): path and filename for input image
            path_mask (str): path and filename for ground truth segmentation mask
            path_labels_boxes (str): path and filename for ground truth labels and boxes

        Returns:
            tuple[tf.Tensor, tf.Tensor, tf.Tensor]: a tuple containing image, segmentation mask and encoded labels boxes
        """

        # encode ground truth labels and bounding boxes
        labels_boxes_encoded = self._encode_ground_truth_labels_boxes(path_labels_boxes=path_labels_boxes)

        # read the image, resize and scale value between 0 and 1
        image = tf.io.read_file(path_image)
        image = tf.image.decode_png(image, channels=3)
        image = tf.cast(image, dtype=tf.float32) / 255.0

        # read the segmentation mask, ignoring transparency channel in the png, one hot encode the classes, squeeze out unwanted dimension
        mask = tf.io.read_file(path_mask)
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.one_hot(mask, depth=self.num_classes, dtype=tf.float32)
        mask = tf.squeeze(mask, axis=2)

        return image, mask, labels_boxes_encoded
