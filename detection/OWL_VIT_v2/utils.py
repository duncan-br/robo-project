import os
import numpy as np
import skimage
from skimage import io as skimage_io
import matplotlib; matplotlib.use("tkagg")
from matplotlib import pyplot as plt
from scipy.special import expit as sigmoid
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, accuracy_score

def plot_confusion_matrix(gt_boxes, est_boxes, class_names, iou_threshold=0.5):
    """
    Plots a classification confusion matrix based on matched bounding boxes with IOU > iou_threshold.

    Args:
        gt_boxes: List of ground truth boxes [(x1, y1, x2, y2, class_id), ...].
        est_boxes: List of estimated boxes [(x1, y1, x2, y2, class_id), ...].
        class_names: List of class names corresponding to class IDs.
        iou_threshold: IOU threshold to consider a match.
    """
    gt_labels = []
    est_labels = []

    matched_gt = set()  # Track matched ground truth indices
    matched_est = set()  # Track matched estimated indices

    # Match estimated boxes to ground truth
    for est_idx, est_box in enumerate(est_boxes):
        est_x1, est_y1, est_x2, est_y2, est_class_id = est_box
        best_iou = 0
        best_gt_idx = 0
        
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            
            gt_x1, gt_y1, gt_x2, gt_y2, gt_class_id = gt_box
            iou = calculate_iou((est_x1, est_y1, est_x2, est_y2), (gt_x1, gt_y1, gt_x2, gt_y2))
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou > iou_threshold:
            matched_gt.add(best_gt_idx)
            matched_est.add(est_idx)
            gt_labels.append(gt_boxes[best_gt_idx][4])  # Append the matched GT class
            est_labels.append(est_class_id)  # Append the estimated class
    
    # Generate the confusion matrix
    cm = confusion_matrix(gt_labels, est_labels, labels=list(range(len(class_names))))

    # Plot the absolute confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("Classification Confusion Matrix (Matched BBoxes)")
    plt.show()

    # Calculate the percentage-based confusion matrix
    cm_percentage = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100
    cm_percentage = np.nan_to_num(cm_percentage)  # Replace NaNs (from division by zero) with 0

    # Plot the percentage confusion matrix
    disp_percentage = ConfusionMatrixDisplay(confusion_matrix=cm_percentage, display_labels=class_names)
    disp_percentage.plot(cmap=plt.cm.Greens, xticks_rotation=45, values_format=".2f")
    plt.title("Percentage Classification Confusion Matrix (Matched BBoxes)")
    plt.show()

def calculate_iou(bbox1, bbox2):

    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

    union_area = bbox1_area + bbox2_area - inter_area

    if union_area == 0:
        return 0 
    iou = inter_area / union_area

    return iou

def calculate_iou_multi(bboxes1, bboxes2):
    num_boxes1 = len(bboxes1)
    num_boxes2 = len(bboxes2)
    
    iou_matrix = np.zeros((num_boxes1, num_boxes2), dtype=np.float32)
    
    for i in range(num_boxes1):
        for j in range(num_boxes2):
            iou_matrix[i, j] = calculate_iou(bboxes1[i], bboxes2[j])
    
    return iou_matrix

def evaluate_predictions(outputs, targets, iou_threshold=0.5, confidence_threshold=0.4):
    all_preds, all_targets = [], []
    
    for output, target in zip(outputs, targets):
        # print(type(output), output, outputs)
        pred_boxes = output['boxes']
        pred_labels = output['labels']
        pred_scores = output['scores']
        
        gt_boxes = target['boxes']
        gt_labels = target['labels']
        
        # Apply confidence threshold
        keep = pred_scores > confidence_threshold
        pred_boxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]
        
        if len(pred_boxes) == 0:
            # If no predictions are left after applying threshold, mark all as false negatives
            all_targets.extend(gt_labels)
            all_preds.extend([0] * len(gt_labels))
            continue
        
        # Compute IoU between predictions and ground truth
        ious = calculate_iou_multi(pred_boxes, gt_boxes)
        
        # Find matches based on IoU threshold
        for i, gt_label in enumerate(gt_labels):
            iou_vals = ious[:, i] if ious.size > 0 else []
            if len(iou_vals) > 0 and iou_vals.max() > iou_threshold:
                max_idx = iou_vals.argmax().item()
                pred_label = pred_labels[max_idx]
                all_preds.append(pred_label)
            else:
                all_preds.append(0)
            all_targets.append(gt_label)

        ## The following is based on a wrong assumption and leads to 
        ## error. Because we assume that when there is a prediction
        ## and no matching gt, there should be something there but
        ## not detected.
        # for i, pred_label in enumerate(pred_labels):
        #     iou_vals = ious[i, :] if ious.size > 0 else []
        #     if len(iou_vals) > 0 and iou_vals.max() > iou_threshold:
        #         pass
        #     else:
        #         all_targets.append(0)
        #         all_preds.append(pred_label)
    
    return all_preds, all_targets

def compute_ap(recalls, precisions):

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return ap

def evaluate_ap(outputs, targets, iou_threshold=0.5, confidence_threshold=0.4):

    predictions = []
    gts = []
    
    for img_id, (output, target) in enumerate(zip(outputs, targets)):
        gt_boxes = target['boxes']
        gt_labels = target['labels']
        for box, label in zip(gt_boxes, gt_labels):
            label_val = label.item() if hasattr(label, 'item') else label
            gts.append({
                'label': label_val,
                'img_id': img_id,
                'box': box, 
                'detected': False
            })
        
        pred_boxes = output['boxes']
        pred_labels = output['labels']
        pred_scores = output['scores']
        
        keep = pred_scores > confidence_threshold
        pred_boxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]
        pred_scores = pred_scores[keep]
        
        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            label_val = label.item() if hasattr(label, 'item') else label
            predictions.append({
                'label': label_val,
                'img_id': img_id,
                'box': box,
                'score': score.item() if hasattr(score, 'item') else score
            })
    
    predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)
    
    tp = np.zeros(len(predictions))
    fp = np.zeros(len(predictions))
    num_gt = len(gts)
    
    for i, pred in enumerate(predictions):
        img_id = pred['img_id']
        pred_box = pred['box']
        pred_cls = pred['label']
        gt_for_img = [gt for gt in gts if (gt['img_id'] == img_id and pred_cls == gt['label'])]
        
        best_iou = 0.0
        best_gt = None
        for gt in gt_for_img:
            iou = calculate_iou(pred_box, gt['box'])
            if iou > best_iou:
                best_iou = iou
                best_gt = gt
        
        if best_iou >= iou_threshold and best_gt is not None and not best_gt['detected']:
            tp[i] = 1 
            best_gt['detected'] = True  
        else:
            fp[i] = 1  
    
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    recalls = tp_cumsum / (num_gt + 1e-6)
    
    ap = compute_ap(recalls, precisions)

    return ap 

def evaluate_map50(outputs, targets, iou_threshold=0.5, confidence_threshold=0.4):

    predictions_by_class = {}
    gt_by_class = {}
    
    for img_id, (output, target) in enumerate(zip(outputs, targets)):
        gt_boxes = target['boxes']
        gt_labels = target['labels']
        for box, label in zip(gt_boxes, gt_labels):
            label_val = label.item() if hasattr(label, 'item') else label
            if label_val not in gt_by_class:
                gt_by_class[label_val] = []
            gt_by_class[label_val].append({
                'img_id': img_id,
                'box': box, 
                'detected': False
            })
        
        pred_boxes = output['boxes']
        pred_labels = output['labels']
        pred_scores = output['scores']
        
        keep = pred_scores > confidence_threshold
        pred_boxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]
        pred_scores = pred_scores[keep]
        
        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            label_val = label.item() if hasattr(label, 'item') else label
            if label_val not in predictions_by_class:
                predictions_by_class[label_val] = []
            predictions_by_class[label_val].append({
                'img_id': img_id,
                'box': box,
                'score': score.item() if hasattr(score, 'item') else score
            })
    
    aps = []
    for cls, gt_items in gt_by_class.items():
        preds = predictions_by_class.get(cls, [])
        if len(preds) == 0:
            aps.append(0)
            continue
        
        preds = sorted(preds, key=lambda x: x['score'], reverse=True)
        
        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        num_gt = len(gt_items)
        
        for i, pred in enumerate(preds):
            img_id = pred['img_id']
            pred_box = pred['box']
            gt_for_img = [gt for gt in gt_items if gt['img_id'] == img_id]
            
            best_iou = 0.0
            best_gt = None
            for gt in gt_for_img:
                iou = calculate_iou(pred_box, gt['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt
            
            if best_iou >= iou_threshold and best_gt is not None and not best_gt['detected']:
                tp[i] = 1 
                best_gt['detected'] = True  
            else:
                fp[i] = 1  
        
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        recalls = tp_cumsum / (num_gt + 1e-6)
        
        ap = compute_ap(recalls, precisions)
        aps.append(ap)
    
    return np.mean(aps) if len(aps) > 0 else 0.0

def save_confusion_matrix(y_true, y_pred, class_names, save_path_numbers, save_path_percentages):
    y_true = np.array([y for y in y_true if y is not None])
    y_pred = np.array([y if y is not None else 0 for y in y_pred])  
    
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    
    plt.figure(figsize=(40, 40))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, values_format="d") 
    plt.xticks(rotation=45, ha="right", fontsize=12)  
    plt.yticks(fontsize=12)
    plt.tight_layout()  
    plt.savefig(save_path_numbers)
    plt.close()
    
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  
    plt.figure(figsize=(40, 40))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_percentage, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, values_format=".2f")  
    plt.xticks(rotation=45, ha="right", fontsize=12)  
    plt.yticks(fontsize=12)
    plt.tight_layout()  
    plt.savefig(save_path_percentages)
    plt.close()

def metrics(y_true, y_pred, class_names):
    y_true = np.array([y for y in y_true if y is not None])
    y_pred = np.array([y if y is not None else 0 for y in y_pred]) 
    
    accuracy = accuracy_score(y_true, y_pred)
    precision_class = precision_score(y_true, y_pred, labels=range(len(class_names)), average=None, zero_division=0)
    recall_class = recall_score(y_true, y_pred, labels=range(len(class_names)), average=None, zero_division=0)
    precision_macro = precision_score(y_true, y_pred, labels=range(len(class_names)), average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, labels=range(len(class_names)), average='macro', zero_division=0)
    precision_micro = precision_score(y_true, y_pred, labels=range(len(class_names)), average='micro', zero_division=0)
    recall_micro = recall_score(y_true, y_pred, labels=range(len(class_names)), average='micro', zero_division=0)

    # Print the metrics
    print(f"Accuracy: {accuracy:.4f}")
    print("Class-wise Precision:")
    for i, prec in enumerate(precision_class):
        print(f"  {class_names[i]}: {prec:.4f}")
    print("Class-wise Recall:")
    for i, rec in enumerate(recall_class):
        print(f"  {class_names[i]}: {rec:.4f}")
    print(f"Overall Precision (Macro): {precision_macro:.4f}")
    print(f"Overall Recall (Macro): {recall_macro:.4f}")
    print(f"Overall Precision (Micro): {precision_micro:.4f}")
    print(f"Overall Recall (Micro): {recall_micro:.4f}")
    
    # Return metrics for further use if needed
    return accuracy

def read_yolo_label_file(label_file_path):

    annotations = []

    try:
        with open(label_file_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) != 5:
                    raise ValueError(f"Invalid line in label file: {line}")

                # Parse the line into class ID and bounding box attributes
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                # Convert YOLO format to corner coordinates
                x1 = x_center - (width / 2)
                y1 = y_center - (height / 2)
                x2 = x_center + (width / 2)
                y2 = y_center + (height / 2)

                annotation = [x1, y1, x2, y2, class_id]
                annotations.append(annotation)
    except FileNotFoundError:
        print(f"Error: Label file '{label_file_path}' not found.")
    except ValueError as e:
        print(f"Error: {e}")

    return annotations


def read_prep_image(dir, input_size):
    # Load example image:
    filename = os.path.join(skimage.data_dir, dir)
    image_uint8 = skimage_io.imread(filename)
    image = image_uint8.astype(np.float32) / 255.0

    # Pad to square with gray pixels on bottom and right:
    h, w, _ = image.shape
    size = max(h, w)
    image_padded = np.pad(
        image, ((0, size - h), (0, size - w), (0, 0)), constant_values=0.5)

    # Resize to model input size:
    input_image = skimage.transform.resize(
        image_padded,
        (input_size, input_size),
        anti_aliasing=True)
    
    return input_image

def draw_detection_results(input_image, scores, boxes, labels, text_queries, output_image_dir=None, score_threshold=0.2):    

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(input_image, extent=(0, 1, 1, 0))
    ax.set_axis_off()

    for score, box, label in zip(scores, boxes, labels):
        cx, cy, w, h = box
        ax.plot([cx - w / 2, cx + w / 2, cx + w / 2, cx - w / 2, cx - w / 2],
                [cy - h / 2, cy - h / 2, cy + h / 2, cy + h / 2, cy - h / 2], 'r')
        ax.text(
            cx - w / 2,
            cy + h / 2 + 0.015,
            f'{text_queries[label]}: {score:1.2f}',
            ha='left',
            va='top',
            color='red',
            bbox={
                'facecolor': 'white',
                'edgecolor': 'red',
                'boxstyle': 'square,pad=.3'
            })
    
    if output_image_dir:
        plt.savefig(output_image_dir)
    else:
        plt.show()