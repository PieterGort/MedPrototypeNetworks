import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize
import wandb
import wandb.plot

def get_sample_weights(targets):
    class_sample_counts = np.bincount(targets)
    class_weights = 1. / class_sample_counts
    sample_weights = np.array([class_weights[t] for t in targets])
    return sample_weights

def binary_metrics(targets, predictions, probabilities, pos_label=1, sample_weights=None):
    """
    Calculate performance metrics for binary classification.
    
    Args:
    - targets (array-like): True labels.
    - predictions (array-like): Predicted labels.
    - probabilities (array-like): Probabilities for the positive class.

    Returns:
    - Dictionary of metrics (precision, recall, f1-score, balanced accuracy, AUC-ROC).
    """
    precision = precision_score(targets, predictions, pos_label=pos_label, average='binary', sample_weight=sample_weights)
    recall = recall_score(targets, predictions, pos_label=pos_label, average='binary', sample_weight=sample_weights)
    f1 = f1_score(targets, predictions, pos_label=pos_label, average='binary', sample_weight=sample_weights)
    bacc = balanced_accuracy_score(targets, predictions, sample_weight=sample_weights)
    auc_roc = roc_auc_score(targets, probabilities[:, 1], sample_weight=sample_weights)  # probabilities is the probability of the positive class

    return {
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Balanced Accuracy': bacc,
        'AUC-ROC': auc_roc
    }

def multiclass_metrics(targets, predictions, probabilities, n_classes, sample_weights=None):
    """
    Calculate performance metrics for multi-class classification.
    
    Args:
    - targets (array-like): True labels.
    - predictions (array-like): Predicted labels.
    - probabilities (array-like): Probabilities for each class.
    - n_classes (int): Number of classes.

    Returns:
    - Dictionary of metrics (precision, recall, f1-score, balanced accuracy, AUC-ROC).
    """

    label_vector = [i for i in range(n_classes)]

    targets_binarized = label_binarize(targets, classes=np.arange(n_classes))
    auc_roc = roc_auc_score(targets_binarized, probabilities, average='weighted', labels=label_vector, multi_class='ovr', sample_weight=sample_weights)
    precision = precision_score(targets, predictions, average='weighted', labels=label_vector, sample_weight=sample_weights)
    recall = recall_score(targets, predictions, average='weighted', labels=label_vector, sample_weight=sample_weights)
    f1 = f1_score(targets, predictions, average='weighted', labels=label_vector, sample_weight=sample_weights)
    balanced_acc = balanced_accuracy_score(targets, predictions, sample_weight=sample_weights)
    # confusion = confusion_matrix(targets, predictions, labels=label_vector, sample_weight=sample_weights)
    confusion_plot = wandb.plot.confusion_matrix(probs=None, y_true=targets, preds=predictions, class_names=label_vector)

    return {
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Balanced Accuracy': balanced_acc,
        'AUC-ROC': auc_roc,
        'Confusion Matrix': confusion_plot
    }


def bb_intersection_over_union(boxA, boxB):
	# copy the function custom_bb_metric in here using the boxA and boxB as input
    assert boxA[0] < boxA[2]
    assert boxA[1] < boxA[3]
    assert boxB[0] < boxB[2]
    assert boxB[1] < boxB[3]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    if xA > xB or yA > yB:
        return 0
    
    interArea = (xA - xB) * (yA - yB)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    assert iou >= 0
    assert iou <= 1

    return iou

def overlap_percentage(ground_truth_box, predicted_box):
    """
    Calculate the percentage of the ground truth box that is overlapped by the predicted box.
    
    :param ground_truth_box: [x1, y1, x2, y2] of the ground truth (physician's) box
    :param predicted_box: [x1, y1, x2, y2] of the predicted (prototype's) box
    :return: Overlap percentage (0.0 to 1.0)
    """

    # Extract coordinates
    gt_x1, gt_y1, gt_x2, gt_y2 = ground_truth_box
    pred_x1, pred_y1, pred_x2, pred_y2 = predicted_box
    
    # Calculate intersection
    x_left = max(gt_x1, pred_x1)
    y_top = max(gt_y1, pred_y1)
    x_right = min(gt_x2, pred_x2)
    y_bottom = min(gt_y2, pred_y2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0  # No overlap
    
    # Calculate areas
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    pb_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    
    # Calculate overlap percentage
    overlap_percent = intersection_area / pb_area
    
    return overlap_percent
