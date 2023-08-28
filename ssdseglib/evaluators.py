import csv
import numpy as np
import tensorflow as tf
from typing import Dict, List, Union

def TO_DELETE_iou_boxes_pred_vs_true(
        labels_pred: np.ndarray[int],
        boxes_pred: np.ndarray[float],
        labels_true: np.ndarray[int],
        boxes_true: np.ndarray[float],
        label_code_background: int
    ):

    # compare predictions and ground truth using broadcasting
    # a value of 1 indicate a right classification, otherwise 0
    # note that output shape it's (number of boxes predicted, number of boxes ground truth)
    labels_pred = np.expand_dims(labels_pred, axis=1)
    labels_true = np.expand_dims(labels_true, axis=0)
    true_positives = labels_pred == labels_true   
    true_positives = true_positives.astype(np.float32)

    # dummy array for identify predictions different from background
    # it's used to keep calculation results related to predictions different from background
    not_background = labels_pred != label_code_background
    not_background = not_background.astype(np.float32)

    # split to corners coordinates for easier areas calculations
    xmin_pred, ymin_pred, xmax_pred, ymax_pred = np.split(boxes_pred, 4, axis=-1)    
    xmin_true, ymin_true, xmax_true, ymax_true = np.split(boxes_true, 4, axis=-1)

    # intersection coordinates between all predictions and ground truth boxes, using broadcasting
    # it selects the maximum for xmin and ymin coordinates, the minimum for xmax and ymax coordinates
    # note that output shape it's (number of boxes predicted, number of boxes ground truth)
    xmin_intersection = np.maximum(xmin_pred, np.transpose(xmin_true)) * not_background
    ymin_intersection = np.maximum(ymin_pred, np.transpose(ymin_true)) * not_background
    xmax_intersection = np.minimum(xmax_pred, np.transpose(xmax_true)) * not_background
    ymax_intersection = np.minimum(ymax_pred, np.transpose(ymax_true)) * not_background

    # areas
    area_pred = (xmax_pred - xmin_pred + 1.0) * (ymax_pred - ymin_pred + 1.0) * not_background
    area_true = (xmax_true - xmin_true + 1.0) * (ymax_true - ymin_true + 1.0)
    area_intersection = np.maximum(0.0, xmax_intersection - xmin_intersection + 1.0) * np.maximum(0.0, ymax_intersection - ymin_intersection + 1.0) * not_background

    # iou between all predictions and ground truth boxes
    epsilon = 1e-7
    iou = area_intersection / (area_pred + np.transpose(area_true) - area_intersection + epsilon)

    # keep only iou where the predicted class match the ground truth one
    iou = iou * true_positives

    return iou

def _iou_boxes_pred_vs_true(
        labels_pred: np.ndarray[int],
        boxes_pred: np.ndarray[float],
        labels_true: np.ndarray[int],
        boxes_true: np.ndarray[float]
    ):

    # compare predictions and ground truth using broadcasting
    # a value of 1 indicate a right classification, otherwise 0
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

    # iou between all predictions and ground truth boxes
    epsilon = 1e-7
    iou = area_intersection / (area_pred + np.transpose(area_true) - area_intersection + epsilon)

    # keep only iou where the predicted class match the ground truth one
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


    # for each class label store true positive and probability
    # dummy value equal to 1 means that the prediction it's a true positive
    # dummy value equal to 0 means that the prediction it's a false positive
    true_positives_confidences_per_label = {label: [] for label in labels_codes if label != label_code_background}

    # for each class label count the total number of ground truth boxes
    ground_truth_boxes_counter = {label: 0 for label in labels_codes if label != label_code_background}

    # evaluate each sample prediction against the corresponding ground truth
    for path_file_labels_boxes, labels_pred, confidences_pred, boxes_pred in zip(path_files_labels_boxes, labels_pred_batch, confidences_pred_batch, boxes_pred_batch):

        # read labels and boxes (in ground truth data background labels don't exists)
        labels_true = []
        boxes_true = []
        with open(path_file_labels_boxes, 'r') as f:
            for label, xmin, ymin, xmax, ymax in csv.reader(f):
                # format the label code
                label = int(label)

                # append to the ground truth data lists
                labels_true.append(label)
                boxes_true.append([float(xmin), float(ymin), float(xmax), float(ymax)])
                
                # increment ground truth boxes counter
                ground_truth_boxes_counter[label] += 1

        # convert to numpy array
        labels_true = np.array(labels_true, dtype=np.int32)
        boxes_true = np.array(boxes_true, dtype=np.float32)

        # ignore background class predictions
        not_background = labels_pred != label_code_background
        labels_pred = labels_pred[not_background]
        confidences_pred = confidences_pred[not_background]
        boxes_pred = boxes_pred[not_background]

        # calculate iou between all predictions and ground truth boxes
        iou = _iou_boxes_pred_vs_true(
            labels_pred=labels_pred,
            boxes_pred=boxes_pred,
            labels_true=labels_true,
            boxes_true=boxes_true
        )

        # for each prediction keep best ground truth iou
        iou = np.max(iou, axis=1)

        # apply iou threshold to determine if the prediction it's a true positive
        true_positives = iou > iou_threshold
        true_positives = true_positives.astype(np.int32)

        # add to the list
        for label_pred, confidence_pred, true_positive in zip(labels_pred, confidences_pred, true_positives):
            true_positives_confidences_per_label[label_pred].append((true_positive, confidence_pred))

    # calculate average precision for each class
    average_precision_per_label = {}
    for label, true_positives_confidences in true_positives_confidences_per_label.items():

        if len(true_positives_confidences) > 0:
            true_positives_confidences = np.array(true_positives_confidences, dtype=np.float32)
            descending_order_by_confidences = np.argsort(true_positives_confidences[:, 1])[::-1]
            true_positives = true_positives_confidences[descending_order_by_confidences, 0]

            precision = np.cumsum(true_positives) / np.arange(1, len(true_positives) + 1)
            recall = np.cumsum(true_positives) / ground_truth_boxes_counter[label]

            average_precision_per_label[label] = np.trapz(y=precision, x=recall)

        else:
            average_precision_per_label[label] = 0.

    # teo ricordati di testare
    # - se non ci sono box di ground truth cosa succede (basta mettere un continue mi sa)
    # - se tutte le box previste sono background (basta mettere un continue anche in questo caso probabilmente)
    # però attento all'ordine, se il modello non prevede niente cmq le box di ground truth le dobbiamo contare!

    s = 0

