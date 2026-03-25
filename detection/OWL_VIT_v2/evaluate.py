import os
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from matplotlib.patches import Rectangle
import re
# from text_conditioned import class_query_map

from detection.OWL_VIT_v2.utils import evaluate_predictions, evaluate_map50, save_confusion_matrix, metrics


results_folder_results = '/home/hojat/Documents/Saxion/e_waste_dataset/valid/results/'
ground_truth_folder = '/home/hojat/Documents/Saxion/e_waste_dataset/valid/labels'
images_folder = '/home/hojat/Documents/Saxion/e_waste_dataset/valid/images'


gt_loaded = False
k10s = {}
graphs = {}
ground_truths = []

for mode in os.listdir(results_folder_results):

    mode_folder_parent = os.path.join(results_folder_results, mode)
    methods = os.listdir(mode_folder_parent)

    for method in methods:

        graphs[method] = {}

        method_folder_parent = os.path.join(mode_folder_parent, method)

        for item in os.listdir(method_folder_parent):
            results_folder = os.path.join(method_folder_parent, item)
            # Check if the item is a directory
            if os.path.isdir(results_folder):
                print(f"Found folder: {results_folder}")
            else:
                continue 

            class_names_path = os.path.join(os.path.dirname(ground_truth_folder), "classes.txt")
            class_names = ['background']
            with open(class_names_path, mode='r') as file:
                for line in file:
                    class_names.append(line.strip())
            no_match_label = 0

            predictions = []
            
            for filename in os.listdir(results_folder):
                if '.txt' not in filename:
                    continue

                prediction = {"boxes": [], "labels": [], "scores": []}
                ground_truth = {"boxes": [], "labels": []}

                result_file = os.path.join(results_folder, filename)
                gt_file = os.path.join(ground_truth_folder, filename)

                try:
                    src_img_file_name = os.path.join(images_folder, filename.replace('.txt', '.jpg'))
                    src_image = plt.imread(src_img_file_name)
                except:
                    src_img_file_name = os.path.join(images_folder, filename.replace('.txt', '.png'))
                    src_image = plt.imread(src_img_file_name)
                src_aspect_ratio = src_image.shape[0] / src_image.shape[1] 

                # Load predictions and ground truths
                with open(result_file, 'r') as f:
                    for line in f:
                        vals_list = line.strip().split()
                        vals_list = [float(val) for val in vals_list]
                        if len(vals_list) == 6:
                            label, x_center, y_center, width, height, score = vals_list
                            prediction["scores"].append(float(score))
                        else:
                            label, x_center, y_center, width, height = vals_list
                            prediction["scores"].append(1.0)
                        xc, yc, w, h = x_center, y_center / src_aspect_ratio, width, height / src_aspect_ratio
                        x1, y1, x2, y2 = xc - w/2, yc - h/2, xc + w/2, yc + h/2
                        
                        prediction["boxes"].append(np.array([x1, y1, x2, y2]))
                        prediction["labels"].append(int(label)+1)
                        

                prediction["boxes"] = np.array(prediction["boxes"])
                prediction["labels"] = np.array(prediction["labels"])
                prediction["scores"] = np.array(prediction["scores"])
                predictions.append(prediction)

                if not gt_loaded:
                    with open(gt_file, 'r') as f:
                        for line in f:
                            
                            try:
                                label, x_center, y_center, width, height = map(float, line.strip().split())

                                ground_truth["boxes"].append(np.array([x_center - width/2, y_center - height/2, x_center + width/2, y_center + height/2]))
                                ground_truth["labels"].append(int(label)+1)
                            except:
                                print(f"Error: Could not read ground-truth data in file {gt_file}")
                                pass

                    ground_truth["boxes"] = np.array(ground_truth["boxes"])
                    ground_truth["labels"] = np.array(ground_truth["labels"])
                    ground_truths.append(ground_truth)
            
            gt_loaded = True
            
            all_preds, all_targets = evaluate_predictions(predictions, ground_truths)
            map = evaluate_map50(predictions, ground_truths)
            # save_confusion_matrix(all_targets, all_preds, class_names, results_folder + "_numbers.jpg", results_folder + "_percentages.jpg")
            acc = metrics(all_targets, all_preds, class_names)
            print(f"mAP50: {map:.4f}")
            print("\n\n\n")

            if mode != "image_conditioned":
                continue

            match = re.search(r'image_conditioned_k(\d+)', results_folder)
            if match:
                number = int(match.group(1))
            else:
                raise ValueError(f"The k number in the results folder {results_folder}")
            if number not in graphs[method].keys():
                graphs[method][number] = []

            graphs[method][number].append(acc)

            if "k10" in results_folder:
                k10s[method] = f"acc: {acc:.4f}, map: {map:.3f}"

            

    print("k-10:")
    for mthd in k10s.keys():
        print(f"\t{mthd}\t{k10s[mthd]}")

    plt.figure(figsize=(6, 4))
    num_methods = len(graphs)
    colors = cm.get_cmap('tab10', num_methods)

    for idx, (mthd, accs) in enumerate(graphs.items()):
        sorted_keys = sorted(accs.keys())
        sorted_accs = [accs[k] for k in sorted_keys]

        method_name = mthd
        if mthd == "knn_median":
            method_name = "fine-grained clustered"

        xs = list(range(len(sorted_keys)))
        color = colors(idx)
        plt.plot(xs, sorted_accs, label=method_name, color=color, linewidth=2, marker='o')
        
        # for i, acc in enumerate(accs):
        #     plt.annotate(f"{mthd}_{i}", (xs[i], acc), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)

    plt.xticks(range(len(sorted_keys)), sorted_keys)

    plt.title('Accuracy Change Over Time')
    plt.xlabel('Number of Sample Queries (k)')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.show()