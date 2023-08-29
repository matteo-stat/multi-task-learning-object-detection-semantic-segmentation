import csv
import numpy as np
import tensorflow as tf
from typing import Dict, List, Union

def _iou_boxes_pred_vs_true(
        labels_pred: np.ndarray[int],
        boxes_pred: np.ndarray[float],
        labels_true: np.ndarray[int],
        boxes_true: np.ndarray[float]
    ) -> np.ndarray:
    """
    calculate iou between each predicted box and all ground truth boxes\n
    if a predicted label don't match the ground truth label then iou will be zero

    Args:
        labels_pred (np.ndarray[int]): predicted labels, expected shape it's (number of predicted boxes, 1)
        boxes_pred (np.ndarray[float]): predicted boxes, in corners coordinates (xmin, ymin, xmax, ymax), expected shapte it's (number of predicted boxes, 4)
        labels_true (np.ndarray[int]): ground truth labels, expected shape it's (number of ground truth boxes, 1)
        boxes_true (np.ndarray[float]): ground truth boxes, in corners coordinates (xmin, ymin, xmax, ymax), expected shapte it's (number of ground truth boxes, 4)

    Returns:
        np.ndarray: a matrix containing iou values between predicted boxes and ground truth boxes, output shape it's (number of predicted boxes, number of ground truth boxes)
    """

    # if there are no ground truth boxes than iou is zero for all predicted boxes
    if len(labels_true) == 0:
        iou = np.zeros(shape=(boxes_pred.shape[0], 1), dtype=np.float32)

    else:
        # compare predictions and ground truth using broadcasting
        # a value of 1 indicates a right classification, otherwise 0
        # note that output shape it's (number of boxes predicted, number of boxes ground truth)
        labels_pred = np.expand_dims(labels_pred, axis=1)
        labels_true = np.expand_dims(labels_true, axis=0)
        true_positives = labels_pred == labels_true
        true_positives = true_positives.astype(np.float32)

        # split to corners coordinates for easier areas calculations
        xmin_pred, ymin_pred, xmax_pred, ymax_pred = np.split(boxes_pred, 4, axis=-1)    
        xmin_true, ymin_true, xmax_true, ymax_true = np.split(boxes_true, 4, axis=-1)

        # intersection coordinates between all predictions and ground truth boxes, using broadcasting
        # it selects the maximum for xmin and ymin coordinates, the minimum for xmax and ymax coordinates
        # note that output shape it's (number of boxes predicted, number of boxes ground truth)
        xmin_intersection = np.maximum(xmin_pred, np.transpose(xmin_true))
        ymin_intersection = np.maximum(ymin_pred, np.transpose(ymin_true))
        xmax_intersection = np.minimum(xmax_pred, np.transpose(xmax_true))
        ymax_intersection = np.minimum(ymax_pred, np.transpose(ymax_true))

        # areas
        area_pred = (xmax_pred - xmin_pred + 1.0) * (ymax_pred - ymin_pred + 1.0)
        area_true = (xmax_true - xmin_true + 1.0) * (ymax_true - ymin_true + 1.0)
        area_intersection = np.maximum(0.0, xmax_intersection - xmin_intersection + 1.0) * np.maximum(0.0, ymax_intersection - ymin_intersection + 1.0)

        # iou between all predictions and ground truth boxes (small epsilon used to avoid possible division by zero problems)
        epsilon = 1e-7
        iou = area_intersection / (area_pred + np.transpose(area_true) - area_intersection + epsilon)

        # keep only iou where the predicted class match the ground truth one, otherwise set iou to zero
        iou = iou * true_positives

    return iou

def average_precision_object_detection(
        labels_pred_batch: np.ndarray[int],
        confidences_pred_batch: np.ndarray[float],
        boxes_pred_batch: np.ndarray[float],
        iou_threshold: float,
        path_files_labels_boxes: List[str],
        labels_codes: List[int],
        label_code_background: int
    ) -> Dict[str, float]:
    """
    calculate the average precision metric for each class for an object detection problem, given a iou threshold\n
    the iou threshold it's used to calculate predictions that are true positives\n
    a true positive it's defined as a predicted box that overlaps with a ground truth box having iou >= than iou threshold\n
    the first dimension of predictions it's expected to be the batch dimension (this should be equal to the number of samples to evaluate)    

    Args:
        labels_pred_batch (np.ndarray[int]): the batch of predicted labels, expected shape it's (number of samples to evaluate, number of boxes after nms, 1)
        confidences_pred_batch (np.ndarray[float]): the batch of predicted labels confidences, expected shape it's (number of samples to evaluate, number of boxes after nms, 1)
        boxes_pred_batch (np.ndarray[float]): the batch of predicted boxes corners coordinates (xmin, ymin, xmax, ymax), expected shape it's (number of samples to evaluate, number of boxes after nms, 4)
        iou_threshold (float): minimum iou required to consider a predicted box that overlaps with a ground truth one as a true positive
        path_files_labels_boxes (List[str]): list containing path and file name for ground truth boxes, expected length it's (number of samples to evaluate),
        each csv file should contain the following columns (label, xmin, ymin, xmax, ymax)
        labels_codes (List[int]): a list with the the labels codes (you can omit background code if you want)
        label_code_background (int): the label code for background class (this should exist only for predictions)

    Returns:
        Dict[str, float]: a dictionary containing average precision (value) for each class (key)
    """

    # for each class label store true positives and predictions confidences
    # true positives are represented by a dummy value (0 or 1)
    # dummy value equal to 1 means that the prediction it's a true positive
    # dummy value equal to 0 means that the prediction it's a false positive
    # this will be used for calculate precision and recall
    true_positives_confidences_per_label = {label: [] for label in labels_codes if label != label_code_background}

    # for each class label store the count of ground truth boxes
    # this will be used for calculate recall
    ground_truth_boxes_counter = {label: 0 for label in labels_codes if label != label_code_background}

    # for each sample evaluates the object detection predictions against ground truth data
    for path_file_labels_boxes, labels_pred, confidences_pred, boxes_pred in zip(path_files_labels_boxes, labels_pred_batch, confidences_pred_batch, boxes_pred_batch):

        # read ground truth labels and boxes
        labels_true = []
        boxes_true = []
        with open(path_file_labels_boxes, 'r') as f:
            for label, xmin, ymin, xmax, ymax in csv.reader(f):
                # format the label code
                label = int(label)

                # append to the ground truth data lists
                labels_true.append(label)
                boxes_true.append([float(xmin), float(ymin), float(xmax), float(ymax)])
                
                # increment ground truth boxes counter for the corresponding class
                ground_truth_boxes_counter[label] += 1

        # convert ground truth data to numpy array
        labels_true = np.array(labels_true, dtype=np.int32)
        boxes_true = np.array(boxes_true, dtype=np.float32)

        # predicted boxes related to background class should be ignored
        # there are no ground truth boxes for background class
        not_background = labels_pred != label_code_background
        labels_pred = labels_pred[not_background]
        confidences_pred = confidences_pred[not_background]
        boxes_pred = boxes_pred[not_background]

        # if all predicted boxes are background then skip to next sample predictions
        # there is the extreme case where there are no predicted boxes and no ground truth boxes
        # but we're not tracking metrics for the background class, so we can skip
        if len(labels_pred) == 0:
            continue

        # calculate iou between all predictions and ground truth boxes
        iou = _iou_boxes_pred_vs_true(
            labels_pred=labels_pred,
            boxes_pred=boxes_pred,
            labels_true=labels_true,
            boxes_true=boxes_true
        )

        # for each prediction keep best ground truth iou
        iou = np.max(iou, axis=1)

        # apply iou threshold to determin if a prediction should be considered a true positive
        true_positives = iou >= iou_threshold
        true_positives = true_positives.astype(np.int32)

        # for each prediction add true positive and confidence values to the right list
        for label_pred, confidence_pred, true_positive in zip(labels_pred, confidences_pred, true_positives):
            true_positives_confidences_per_label[label_pred].append((true_positive, confidence_pred))

    # initialize the average precision container
    # it will contain average precision metric (value) for each class (key)
    average_precision_per_label = {}

    # for each class calculate average precision
    for label, true_positives_confidences in true_positives_confidences_per_label.items():

        # if no predictions were made or no ground truth boxes exists the average precision it's zero
        if ground_truth_boxes_counter[label] == 0 or len(true_positives_confidences) == 0:
            average_precision_per_label[label] = 0.0
            
        else:
            # convert the true positives and confidences list to numpy array
            true_positives_confidences = np.array(true_positives_confidences, dtype=np.float32)

            # sort by confidences in descending order
            descending_order_by_confidences = np.argsort(true_positives_confidences[:, 1])[::-1]

            # true positives sorted by confidences in descending order
            true_positives = true_positives_confidences[descending_order_by_confidences, 0]

            # calculate precision and recall points
            precision_values = np.cumsum(true_positives) / np.arange(1, len(true_positives) + 1)
            recall_values = np.cumsum(true_positives) / ground_truth_boxes_counter[label]

            # calculate area under the precision recall curve, which is the average precision by definition
            average_precision_per_label[label] = np.trapz(y=precision_values, x=recall_values)
            
    return average_precision_per_label

def jaccard_iou_semantic_segmentation(
        masks_pred_batch: np.ndarray,
        path_files_masks: List[str],
        labels_codes: List[int],
        label_code_background: int
    ) -> Dict[str, float]:
    """
    calculate the jaccard iou metric for each class for a semantic segmnetation problem\n
    the first dimension of predictions it's expected to be the batch dimension (this should be equal to the number of samples to evaluate)

    Args:
        masks_pred_batch (np.ndarray): the batch of predicted masks, expected shape it's (number of samples to evaluate, height, width, number of classes)
        path_files_masks (List[str]): list containing path and file name for ground truth masks, expected length it's (number of samples to evaluate),
        each ground truth mask should be a single channel png containing pixel values equal to the class label
        labels_codes (List[int]): a list with the the labels codes (you can omit background code if you want)
        label_code_background (int): the label code for background class (this should exist only for predictions)

    Returns:
        Dict[str, float]: _description_
    """
    # number of classes
    num_classes = len(labels_codes)

    # read all ground truth masks
    masks_true_batch = []
    for path_file_mask in path_files_masks:
        # read the segmentation mask, ignoring transparency channel in the png, one hot encode the classes, squeeze out unwanted dimension
        mask = tf.io.read_file(path_file_mask)
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.one_hot(mask, depth=num_classes, dtype=tf.float32)
        mask = tf.squeeze(mask, axis=2)
        masks_true_batch.append(mask)

    # convert ground truth masks to numpy array
    masks_true_batch = np.array(masks_true_batch, dtype=np.float32)

    # intersection area, along height and width dimensions, output shape it's (number of samples to evaluate, number of classes)
    intersection = np.sum(masks_true_batch * masks_pred_batch, axis=(1, 2))

    # total area, along height and width dimensions, output shape it's (number of samples to evaluate, number of classes)
    total = np.sum(masks_true_batch + masks_pred_batch, axis=(1, 2))

    # jaccard iou metric (a small epsilon value it's used to avoid division by zero)
    # union can be calculated easily as difference between total and intersection
    epsilon = 1e-7
    iou = intersection / (total - intersection + epsilon)

    # average along batch dimension, output shape it's (number of labels,)
    iou = np.mean(iou, axis=0)

    # create output dictionary
    iou = {
        label: iou_label
        for label, iou_label in zip(labels_codes, iou)
        if label != label_code_background
    }

    return iou
