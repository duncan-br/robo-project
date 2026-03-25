import sys
from pathlib import Path

# Add project root so "detection" and "ui" resolve when run by full path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap import Style
from ttkbootstrap.constants import *
from tkinter import filedialog, Label, Toplevel, Button, Entry, Label, StringVar, filedialog, messagebox
from PIL import Image, ImageTk, ImageFont, ImageDraw, ImageEnhance
import threading
import numpy as np
import json
import os
import cv2
import time 
import yaml
import shutil


from detection.OWL_VIT_v2.utils import read_yolo_label_file, calculate_iou, plot_confusion_matrix

_UI_DIR = Path(__file__).resolve().parent
_ROS_TMP_DIR = _UI_DIR / "tmp"
_ROS_TMP_DIR.mkdir(parents=True, exist_ok=True)
_ROS_TMP_IMAGE = _ROS_TMP_DIR / "image.png"

class ImageAnnotatorApp:
    def __init__(self, root):
        self.style = Style()  # Initialize ttkbootstrap with a theme
        self.root = root
        self.root.title("Open Robo Vision")

        # Top Bar Frame
        self.top_bar = ttk.Frame(self.root, padding=5, style="secondary.TFrame")
        self.top_bar.pack(side=tk.TOP, fill=tk.X)

        # Entry Label and Field in Top Bar
        self.input_label = ttk.Label(self.top_bar, text="Objectness Threshold:", style="info.TLabel")
        self.input_label.pack(side=tk.LEFT, padx=5)

        validate_command = self.root.register(self.validate_float)
        # Entry field with validation
        self.text_entry = ttk.Entry(
            self.top_bar,
            width=5,
            validate="key",
            validatecommand=(validate_command, "%P")  # Pass the new value to the validation function
        )
        self.text_entry.pack(side=tk.LEFT, padx=5)
        self.text_entry.insert(0, "0.2") 

        self.input_label1 = ttk.Label(self.top_bar, text="Number of queries per class:", style="info.TLabel")
        self.input_label1.pack(side=tk.LEFT, padx=5)

        validate_command1 = self.root.register(self.validate_int)
        self.text_entry1 = ttk.Entry(
            self.top_bar,
            width=5,
            validate="key",
            validatecommand=(validate_command1, "%P")  # Pass the new value to the validation function
        )
        self.text_entry1.pack(side=tk.LEFT, padx=5)
        self.text_entry1.insert(0, "1") 

        self.input_label2 = ttk.Label(self.top_bar, text="Querry merging:", style="info.TLabel")
        self.input_label2.pack(side=tk.LEFT, padx=5)

        options_last_op = ["average", "median", "fine-grained", "knn_median"]
        self.selected_last_op = tk.StringVar(value=options_last_op[0]) 
        for option in options_last_op:
            radiobutton = tk.Radiobutton(self.top_bar, text=option, value=option, variable=self.selected_last_op)
            radiobutton.pack(anchor="w") 

        self.input_label3 = ttk.Label(self.top_bar, text="Mode:", style="info.TLabel")
        self.input_label3.pack(side=tk.LEFT, padx=5)

        options_mode = ["image_conditioned", "text_conditioned"]
        self.selected_mode = tk.StringVar(value=options_mode[0]) 
        for option in options_mode:
            radiobutton = tk.Radiobutton(self.top_bar, text=option, value=option, variable=self.selected_mode)
            radiobutton.pack(anchor="w") 

        # self.read_button = ttk.Button(
        #     self.top_bar, text="Read Value", style="success.TButton", command=self.read_entry_value
        # )
        # self.read_button.pack(side=tk.LEFT, padx=5)

        # Separator below the top bar
        ttk.Separator(self.root, orient=tk.HORIZONTAL).pack(fill=tk.X)

        # Create a PanedWindow for adjustable panels
        self.paned_window = ttk.Panedwindow(self.root, orient=tk.VERTICAL)
        self.paned_window.pack(fill=tk.BOTH, expand=1)

        # Create top and bottom horizontal PanedWindows for the 2x2 layout
        self.top_pane = ttk.Panedwindow(self.paned_window, orient=tk.HORIZONTAL)
        self.bottom_pane = ttk.Panedwindow(self.paned_window, orient=tk.HORIZONTAL)

        self.paned_window.add(self.top_pane)
        self.paned_window.add(self.bottom_pane)

        # Top Left Panel - Image Loader
        self.top_left_panel = ttk.Frame(self.top_pane, padding=10, style="secondary.TFrame")
        self.top_pane.add(self.top_left_panel)

        # Load Button and Listbox for images in Top Left Panel
        self.load_button = ttk.Button(
            self.top_left_panel, text="Load Example Images", style="success.TButton", command=self.load_images
        )
        self.load_button.pack(anchor="nw", pady=5)

        self.load_dataset_button = ttk.Button(
            self.top_left_panel, text="Load YOLO Dataset", style="primary.TButton", command=self.load_yolo_dataset
        )
        self.load_dataset_button.pack(anchor="nw", pady=5)

        self.sim_acl = ttk.Button(
            self.top_left_panel, text="Simulate ACL", style="primary.TButton", command=self.sim_acl
        )
        self.sim_acl.pack(anchor="nw", pady=5)
        

        self.image_listbox = ttk.Treeview(self.top_left_panel, columns=("Image"), show="headings")
        self.image_listbox.heading("Image", text="Loaded Images")
        self.image_listbox.pack(fill=tk.BOTH, expand=1)
        self.image_listbox.bind("<<TreeviewSelect>>", self.display_image)

        self.bbox_mode_label = tk.Label(self.top_left_panel, text="Bbox mode:")
        self.bbox_mode_label.pack(pady=10)

        options = ["From GT", "From OWL"]
        self.selected_var = tk.StringVar(value=options[0]) 
        for option in options:
            radiobutton = tk.Radiobutton(self.top_left_panel, text=option, value=option, variable=self.selected_var)
            radiobutton.pack(anchor="w") 

        # Top Right Panel - Split Horizontally into Image Display and Class Info
        self.top_right_panel = ttk.Frame(self.top_pane, style="secondary.TFrame")
        self.top_pane.add(self.top_right_panel)

        self.top_right_pane = ttk.Panedwindow(self.top_right_panel, orient=tk.HORIZONTAL)
        self.top_right_pane.pack(fill=tk.BOTH, expand=1)

        # Image Canvas in Top Right Panel (Left Section)
        self.image_canvas = tk.Canvas(
            self.top_right_pane,
            bg="white",  
            width=400, 
            height=400  
        )
        self.image_canvas.pack_propagate(False)  
        self.image_canvas.bind("<Button-1>", self.on_canvas_click)
        self.top_right_pane.add(self.image_canvas)

        # Right Section in Top Right Panel for Class Info
        self.class_info_panel = ttk.Frame(self.top_right_pane, padding=10, style="secondary.TFrame")
        self.top_right_pane.add(self.class_info_panel)

        self.class_display_label = ttk.Label(self.class_info_panel, text="Classes", style="info.TLabel")
        self.class_display_label.pack(anchor="nw", pady=5)

        self.class_display = ttk.Label(self.class_info_panel, text="", style="info.TLabel", justify="left")
        self.class_display.pack(anchor="nw")

        # Save and Load Buttons
        self.buttons_frame = ttk.Frame(self.class_info_panel, style="secondary.TFrame")
        self.buttons_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

        # Save Button
        self.save_button = ttk.Button(
            self.buttons_frame, text="Save Classes", style="success.TButton", command=self.save_class_embeddings
        )
        self.save_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Load Button
        self.load_button = ttk.Button(
            self.buttons_frame, text="Load Classes", style="warning.TButton", command=self.load_class_embeddings
        )
        self.load_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Bottom Left Panel - Image Loader
        self.bottom_left_panel = ttk.Frame(self.bottom_pane, padding=10, style="secondary.TFrame")
        self.bottom_pane.add(self.bottom_left_panel)

        self.load_button_bottom = ttk.Button(
            self.bottom_left_panel, text="Load Images", style="success.TButton", command=self.load_images_bottom
        )
        self.load_button_bottom.pack(anchor="nw", pady=5)

        self.load_dataset_button1 = ttk.Button(
            self.bottom_left_panel, text="Load YOLO Dataset", style="primary.TButton", command=self.load_yolo_dataset1
        )
        self.load_dataset_button1.pack(anchor="nw", pady=5)

        self.connect_to_ros_btn = ttk.Button(
            self.bottom_left_panel, text="Connect to ROS", style="primary.TButton", command=self.connect_to_ros
        )
        self.connect_to_ros_btn.pack(anchor="nw", pady=5)

        self.image_listbox_bottom = ttk.Treeview(self.bottom_left_panel, columns=("Image"), show="headings")
        self.image_listbox_bottom.heading("Image", text="Loaded Images")
        self.image_listbox_bottom.pack(fill=tk.BOTH, expand=1)
        self.image_listbox_bottom.bind("<<TreeviewSelect>>", self.display_image_bottom)

        # Bottom Right Panel - Output Image Display
        self.bottom_right_panel = ttk.Frame(self.bottom_pane, style="secondary.TFrame")
        self.bottom_pane.add(self.bottom_right_panel)

        # Bottom Right Panel - Output Image Display
        self.image_canvas_bottom = tk.Canvas(
            self.bottom_right_panel,
            bg="white",  # Background color for the canvas
            width=400,   # Set the width
            height=400   # Set the height
        )
        self.image_canvas_bottom.pack(fill=tk.BOTH, expand=1)

        # Status label
        self.status_label = ttk.Label(self.root, text="Loading object detector...", style="info.TLabel", anchor="w")
        self.status_label.configure(foreground="red")
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        # Initialize variables
        self.icod = None
        threading.Thread(target=self.load_icod).start()

        self.busy_processing_image = False
        self.busy_processing_dataset = False
        self.idx_obj_map = {}
        self.class_embeddings = {}

        self.images = []
        self.labels = []
        self.images_bottom = []
        self.labels_bottom = []
        self.class_names = []
        self.class_names1 = []

        self.ros_thread = None
        self.keep_node = True

    def load_images(self):
        self.images = []
        self.labels = []
        for item in self.image_listbox.get_children():
            self.image_listbox.delete(item)
        
        filepaths = filedialog.askopenfilenames(title="Select Images", filetypes=[("Image Files", "*.png *.jpg *.jpeg")], initialdir="~/Documents")
        for filepath in filepaths:
            filename = filepath.split('/')[-1]  
            self.image_listbox.insert("", "end", values=(filename,))  
            self.images.append(filepath)  

    def connect_to_ros(self):
        import subprocess
        import tkinter as tk
        from tkinter import ttk
        from tkinter import messagebox

        try:
            from ui.ros_node import main_ros_loop
        except ImportError as e:
            messagebox.showerror(
                "ROS not available",
                "ROS 2 Python packages (rclpy, cv_bridge, etc.) are not installed.\n"
                "Use a host with ROS or extend the Docker image.\n\n"
                f"{e}",
            )
            return

        # Execute the terminal command to get the list of ROS2 topics.
        try:
            result = subprocess.run(
                ["ros2", "topic", "list"],
                capture_output=True, text=True, check=True
            )
            topics_str = result.stdout.strip()
            if not topics_str:
                messagebox.showerror("No Topics", "No ROS2 topics available!")
                return
            all_topics = topics_str.splitlines()
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Error", f"Failed to retrieve topics:\n{e}")
            return

        # Filter topics to include only those with type sensor_msgs/msg/Image.
        image_topics = []
        for topic in all_topics:
            try:
                type_result = subprocess.run(
                    ["ros2", "topic", "type", topic],
                    capture_output=True, text=True, check=True
                )
                topic_type = type_result.stdout.strip()
                if topic_type == "sensor_msgs/msg/Image":
                    image_topics.append(topic)
            except subprocess.CalledProcessError:
                # Skip the topic if there's an error getting its type.
                continue

        if not image_topics:
            messagebox.showerror("No Image Topics", "No image topics available!")
            return

        # Create a simple Tkinter dialog with a dropdown selection.
        root = tk.Tk()
        root.title("Select ROS2 Image Topic")
        root.geometry("300x100")

        selected_topic = tk.StringVar(value=image_topics[0])
        label = ttk.Label(root, text="Choose a ROS2 image topic:")
        label.pack(pady=5)

        combo = ttk.Combobox(root, textvariable=selected_topic, values=image_topics, state="readonly")
        combo.pack(pady=5)
        combo.current(0)

        def on_confirm():
            chosen_topic = selected_topic.get()
            print("Selected image topic:", chosen_topic)

            if self.ros_thread is not None:
                self.keep_node = False
                time.sleep(0.5)

            self.keep_node = True
            threading.Thread(target=main_ros_loop, args=(chosen_topic, self.ros_image_callback, self.keep_ros_node)).start()
            time.sleep(0.5)
            
            root.destroy()

        confirm_button = ttk.Button(root, text="Connect", command=on_confirm)
        confirm_button.pack(pady=5)

        root.mainloop()
        
    def keep_ros_node(self):
        return self.keep_node 
    
    def ros_image_callback(self, image):
        cv2.imwrite(str(_ROS_TMP_IMAGE), image)
        if not self.busy_processing_image:
            self.status_label.configure(text="Processing image ...", foreground="red")
            self.class_names1 = ["HEAT_EXCHANGER", "BRONZE", "LEAD", "STAINLESS_STEEL", 
                                 "ALUMINIUM", "COPPER", "BRASS", "CAPACITOR", "PCB"]
            threading.Thread(
                target=self.process_image_bottom,
                args=(str(_ROS_TMP_IMAGE), True, float(self.text_entry.get())),
            ).start()
            self.busy_processing_image = True
        

    def load_yolo_dataset(self):

        self.images = []
        self.labels = []
        for item in self.image_listbox.get_children():
            self.image_listbox.delete(item)

        folder_path = filedialog.askdirectory(title="Select YOLO Dataset Folder", initialdir="~/Documents")
        if not folder_path:
            return  

        images_folder = os.path.join(folder_path, "images")
        labels_folder = os.path.join(folder_path, "labels")
        class_names_path = os.path.join(folder_path, "classes.txt")

        if not os.path.isdir(images_folder) or not os.path.isdir(labels_folder):
            self.show_error_dialog("The selected folder must contain 'images' and 'labels' subfolders.")
            return

        image_files = [f for f in os.listdir(images_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            self.show_error_dialog("The 'images' subfolder must contain image files (.png, .jpg, .jpeg).")
            return
        
        if not os.path.isfile(class_names_path):
            self.show_error_dialog("There is no classes.txt file in the dataset folder.")
            return

        with open(class_names_path, mode='r') as file:
            self.class_names = []
            for line in file:
                self.class_names.append(line.strip())

        missing_labels = []
        for image_file in image_files:
            label_file = os.path.splitext(image_file)[0] + ".txt"
            if not os.path.isfile(os.path.join(labels_folder, label_file)):
                missing_labels.append(image_file)
            else:
                self.images.append(os.path.join(images_folder, image_file))
                self.labels.append(os.path.join(labels_folder, label_file))
                self.image_listbox.insert("", "end", values=(image_file,))  

        if missing_labels:
            self.images = []
            self.labels = []
            for item in self.image_listbox.get_children():
                self.image_listbox.delete(item)
            missing_count = len(missing_labels)
            total_count = len(image_files)
            error_message = (
                f"The following issues were found:\n"
                f" - Total images: {total_count}\n"
                f" - Missing labels: {missing_count}\n\n"
                f"Each image file in the 'images' folder must have a corresponding label file "
                f"with the same name in the 'labels' folder (e.g., 'images/img1.jpg' -> 'labels/img1.txt')."
            )
            self.show_error_dialog(error_message)
            return

        # messagebox.showinfo("Success", "The YOLO dataset was loaded successfully!")

        if not self.busy_processing_dataset:
            self.status_label.configure(text="Processing dataset ...", foreground="red")
            threading.Thread(target=self.process_dataset).start()

    def sim_acl(self):

        self.images_bottom = []
        self.labels_bottom = []
        for item in self.image_listbox_bottom.get_children():
            self.image_listbox_bottom.delete(item)

        folder_path = filedialog.askdirectory(title="Select YOLO Dataset Folder", initialdir="~/Documents" )
        if not folder_path:
            return  

        self.folder_path1 = folder_path
        images_folder = os.path.join(folder_path, "images")
        labels_folder = os.path.join(folder_path, "labels")
        class_names_path = os.path.join(folder_path, "classes.txt")

        if not os.path.isdir(images_folder) or not os.path.isdir(labels_folder):
            self.show_error_dialog("The selected folder must contain 'images' and 'labels' subfolders.")
            return

        image_files = [f for f in os.listdir(images_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            self.show_error_dialog("The 'images' subfolder must contain image files (.png, .jpg, .jpeg).")
            return
        
        if not os.path.isfile(class_names_path):
            self.show_error_dialog("There is no classes.txt file in the dataset folder.")
            return

        with open(class_names_path, mode='r') as file:
            self.class_names1 = []
            for line in file:
                self.class_names1.append(line.strip())

        missing_labels = []
        for image_file in image_files:
            label_file = os.path.splitext(image_file)[0] + ".txt"
            if not os.path.isfile(os.path.join(labels_folder, label_file)):
                missing_labels.append(image_file)
            else:
                self.images_bottom.append(os.path.join(images_folder, image_file))
                self.labels_bottom.append(os.path.join(labels_folder, label_file))
                self.image_listbox_bottom.insert("", "end", values=(image_file,))  

        if missing_labels:
            self.images_bottom = []
            self.labels_bottom = []
            for item in self.image_listbox.get_children():
                self.image_listbox_bottom.delete(item)
            missing_count = len(missing_labels)
            total_count = len(image_files)
            error_message = (
                f"The following issues were found:\n"
                f" - Total images: {total_count}\n"
                f" - Missing labels: {missing_count}\n\n"
                f"Each image file in the 'images' folder must have a corresponding label file "
                f"with the same name in the 'labels' folder (e.g., 'images/img1.jpg' -> 'labels/img1.txt')."
            )
            self.show_error_dialog(error_message)
            return
        
        file_path = filedialog.askopenfilename(filetypes=[("YAML files", "*.yaml")], initialdir="~/Documents")
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)

        queries_dict = {str(k): str(v) for k, v in data.items()}

        embeddings = self.icod.tokenize_queries(queries_dict.values())

        self.class_embeddings = {}
        for i, q in enumerate(queries_dict.keys()):

            self.class_embeddings[q] = [embeddings[i]]

        self.update_class_display()

        # messagebox.showinfo("Success", "The YOLO dataset was loaded successfully!")

        if not self.busy_processing_dataset:
            self.status_label.configure(text="Processing dataset ...", foreground="red")
            threading.Thread(target=self.simulate_acl).start()

    def simulate_acl(self):

        if self.icod is None:
            print("Could not process dataset without the model loaded.")
            return
            
        self.busy_processing_dataset = True

        all_bboxes = []
        all_gts = []

        merging_mode = self.selected_last_op.get()

        merging_mode = self.selected_last_op.get()
        output_folder = f"{self.folder_path1}/results/acl/{merging_mode}/{self.selected_mode.get()}_k{int(self.text_entry1.get())}"
        if os.path.isdir(output_folder):
            for filename in os.listdir(output_folder):
                file_path = os.path.join(output_folder, filename)
                os.remove(file_path) 
        os.makedirs(output_folder, exist_ok=True)


        for i, filepath in enumerate(self.images_bottom):
            self.status_label.configure(text=f"Processing dataset {i}/{len(self.images_bottom)}.", foreground="red")
            indexes, bboxes, scores, img = self.process_image_bottom(filepath, update_gui=False, conf_thresh=0.01)

            image_file = os.path.basename(filepath)
            output_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}.txt")
            
            objectness_threshold = float( self.text_entry.get() )
            with open(output_path, 'w') as f:
                for box, label, score in zip(bboxes, indexes, scores):
                    if score > objectness_threshold:
                        cx, cy, w, h = box
                        x_center = cx 
                        y_center = cy 
                        width = w 
                        height = h 
                        f.write(f"{label} {x_center} {y_center} {width} {height} {score}\n")
            img.save(output_path.replace(".txt", ".jpg"))

            draw = ImageDraw.Draw(img)

            img_1 = Image.open(filepath)
            orig_size = img_1.size
            h_ratio = orig_size[0] / orig_size[1]

            target_size_w = self.image_canvas_bottom.winfo_width()
            target_size_h = self.image_canvas_bottom.winfo_height()
            if target_size_w < 200:
                target_size_w = 200
            if target_size_h < 200:
                target_size_h = 200

            if len(self.labels_bottom) > 0:
                file_index = self.images_bottom.index(filepath)
                gt_data = read_yolo_label_file(self.labels_bottom[file_index])

                for gt_bboox in gt_data:

                    x_min_gt, y_min_gt, x_max_gt, y_max_gt = gt_bboox[:4]
                    x_min_gt, y_min_gt, x_max_gt, y_max_gt = x_min_gt*target_size_w, y_min_gt*target_size_h, x_max_gt*target_size_w, y_max_gt*target_size_h                  
                    draw.rectangle(((x_min_gt, y_min_gt), (x_max_gt, y_max_gt)), outline="yellow", width=4)

                    best_iou = 0
                    best_box = None
                    best_idx = None
                    best_id = None
                    for idx, (id, bbox, score) in enumerate(zip(indexes, bboxes, scores)):

                        cx, cy, w, h = bbox
                        cx, cy, w, h = cx, cy*h_ratio, w, h*h_ratio
                        x_min = cx-w/2; y_min = cy-h/2; x_max = cx+w/2; y_max = cy+h/2
                    
                        iou = calculate_iou([x_min, y_min, x_max, y_max], gt_bboox[:4])

                        if iou > best_iou and iou > 0.5:
                            best_iou = iou
                            best_box = bbox
                            best_idx = idx
                            best_id = id

                    if best_idx is not None:
                        color = "green"
                        if gt_bboox[4] != best_id:

                            best_name = self.class_names1[gt_bboox[4]]
                            if best_name not in self.class_embeddings:
                                self.class_embeddings[best_name] = []
                            self.class_embeddings[best_name].append(self.class_embeddings_full[best_idx])

                            color = "red"

                        cx, cy, w, h = best_box
                        cx, cy, w, h = cx*target_size_w, cy*h_ratio*target_size_h, w*target_size_w, h*h_ratio*target_size_h
                        x_min = int(cx-w/2)
                        y_min = int(cy-h/2)
                        x_max = int(cx+w/2)
                        y_max = int(cy+h/2)                    
                        draw.rectangle(((x_min, y_min), (x_max, y_max)), outline=color, width=3)


                self.update_class_display()

            self.image_display_bottom = ImageTk.PhotoImage(img)
            self.image_canvas_bottom.create_image(0, 0, anchor="nw", image=self.image_display_bottom)

        self.status_label.configure(text="Done processing the dataset.", foreground="green")
        self.busy_processing_dataset = False

        # plot_confusion_matrix(all_gts, all_bboxes,  self.class_names1)
        
    def load_yolo_dataset1(self):

        self.images_bottom = []
        self.labels_bottom = []
        for item in self.image_listbox_bottom.get_children():
            self.image_listbox_bottom.delete(item)

        folder_path = filedialog.askdirectory(title="Select YOLO Dataset Folder", initialdir="~/Documents" )
        if not folder_path:
            return  

        self.folder_path1 = folder_path
        images_folder = os.path.join(folder_path, "images")
        labels_folder = os.path.join(folder_path, "labels")
        class_names_path = os.path.join(folder_path, "classes.txt")

        if not os.path.isdir(images_folder) or not os.path.isdir(labels_folder):
            self.show_error_dialog("The selected folder must contain 'images' and 'labels' subfolders.")
            return

        image_files = [f for f in os.listdir(images_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            self.show_error_dialog("The 'images' subfolder must contain image files (.png, .jpg, .jpeg).")
            return
        
        if not os.path.isfile(class_names_path):
            self.show_error_dialog("There is no classes.txt file in the dataset folder.")
            return

        with open(class_names_path, mode='r') as file:
            self.class_names1 = []
            for line in file:
                self.class_names1.append(line.strip())

        missing_labels = []
        for image_file in image_files:
            label_file = os.path.splitext(image_file)[0] + ".txt"
            if not os.path.isfile(os.path.join(labels_folder, label_file)):
                missing_labels.append(image_file)
            else:
                self.images_bottom.append(os.path.join(images_folder, image_file))
                self.labels_bottom.append(os.path.join(labels_folder, label_file))
                self.image_listbox_bottom.insert("", "end", values=(image_file,))  

        if missing_labels:
            self.images_bottom = []
            self.labels_bottom = []
            for item in self.image_listbox.get_children():
                self.image_listbox_bottom.delete(item)
            missing_count = len(missing_labels)
            total_count = len(image_files)
            error_message = (
                f"The following issues were found:\n"
                f" - Total images: {total_count}\n"
                f" - Missing labels: {missing_count}\n\n"
                f"Each image file in the 'images' folder must have a corresponding label file "
                f"with the same name in the 'labels' folder (e.g., 'images/img1.jpg' -> 'labels/img1.txt')."
            )
            self.show_error_dialog(error_message)
            return

        # messagebox.showinfo("Success", "The YOLO dataset was loaded successfully!")

        if not self.busy_processing_dataset:
            self.status_label.configure(text="Processing dataset ...", foreground="red")
            threading.Thread(target=self.process_dataset1).start()

    def show_error_dialog(self, message):
        """
        Displays an error dialog with the specified message.
        """
        messagebox.showerror("Dataset Format Error", message)

    def display_image(self, event):
        selected_items = self.image_listbox.selection()  
        if not selected_items:
            return
        selected_item = selected_items[0] 
        item_index = self.image_listbox.index(selected_item)  
        filepath = self.images[item_index]  

        if not self.busy_processing_image:
            self.status_label.configure(text="Processing image ...", foreground="red")
            threading.Thread(target=self.process_image, args=(filepath, [""])).start()

    def load_images_bottom(self):
        filepaths = filedialog.askopenfilenames(title="Select Images", filetypes=[("Image Files", "*.png *.jpg *.jpeg")], initialdir="~/Documents")
        for filepath in filepaths:
            filename = filepath.split('/')[-1] 
            self.image_listbox_bottom.insert("", "end", values=(filename,)) 
            self.images_bottom.append(filepath)  

    def display_image_bottom(self, event):
        selected_items = self.image_listbox_bottom.selection() 
        if not selected_items:
            return
        selected_item = selected_items[0]  
        item_index = self.image_listbox_bottom.index(selected_item)  
        filepath = self.images_bottom[item_index] 

        if not self.busy_processing_image:
            self.status_label.configure(text="Processing image ...", foreground="red")
            threading.Thread(target=self.process_image_bottom, args=(filepath,True,float(self.text_entry.get()))).start()

    def load_icod(self):
        # Import here so the Tk window can map before JAX/TF and OWL weights load (avoids long noVNC black screen).
        try:
            from detection.OWL_VIT_v2.image_conditioned import ImageConditionedObjectDetector

            self.icod = ImageConditionedObjectDetector()
            self.root.after(
                0,
                lambda: self.status_label.configure(text="Object detector loaded!", foreground="green"),
            )
        except Exception as e:
            self.root.after(
                0,
                lambda err=str(e): self.status_label.configure(
                    text=f"Detector load failed: {err}", foreground="red"
                ),
            )

    def process_image(self, filepath, queries):
        self.busy_processing_image = True

        # t_queries = self.icod.tokenize_queries(queries)
        boxes, objectnesses, class_embeddings = self.icod.process(filepath)

        self.class_embeddings_full = class_embeddings  
        
        target_size_w = self.image_canvas.winfo_width()
        target_size_h = self.image_canvas.winfo_height()
        if target_size_w < 200:
            target_size_w = 200
        if target_size_h < 200:
            target_size_h = 200

        # objectness_threshold = np.partition(objectnesses, -top_k)[-top_k]
        objectness_threshold = float( self.text_entry.get() )

        img = Image.open(filepath)
        orig_size = img.size
        h_ratio = orig_size[0] / orig_size[1]

        img = Image.open(filepath)
        img = img.resize((target_size_w, target_size_h), Image.LANCZOS)  # Resize for display purposes

        draw = ImageDraw.Draw(img)
        self.idx_obj_map = {}
        bbox_mode = self.selected_var.get()
        prev_boxes = []
        for idx, (box, objectness) in enumerate(zip(boxes, objectnesses)):
            if bbox_mode == "From GT" and len(self.labels) > 0 and objectness > 0.01:
                best_iou = 0
                best_box = None
                t0 = time.time()
    
                file_index = self.images.index(filepath)
                gt_data = read_yolo_label_file(self.labels[file_index])
                for gt_bboox in gt_data: # For all labels in the file
                    cx, cy, w, h = box
                    cx, cy, w, h = cx, cy*h_ratio, w, h*h_ratio
                    x_min, y_min, x_max, y_max = cx-w/2, cy-h/2, cx+w/2, cy+h/2
                    iou = calculate_iou([x_min, y_min, x_max, y_max], gt_bboox[:4])
                    if iou > best_iou and iou > 0.7:
                        best_iou = iou
                        best_box = gt_bboox[:4]

                if best_box is not None: # Check if it wasn't detected previously
                    for prev_box in prev_boxes:
                        niou = calculate_iou(prev_box, best_box)
                        if niou > 0.8:
                            best_iou = 0
                    prev_boxes.append(best_box)

                if best_iou == 0:
                    continue
            else:
                if objectness < objectness_threshold:
                    continue

            cx, cy, w, h = box
            cx, cy, w, h = cx*target_size_w, cy*h_ratio*target_size_h, w*target_size_w, h*h_ratio*target_size_h
            x_min = int(cx-w/2)
            y_min = int(cy-h/2)
            x_max = int(cx+w/2)
            y_max = int(cy+h/2)
            self.idx_obj_map[idx] = ([x_min, y_min, x_max, y_max], objectness)

            cx_gt, cy_gt, w_gt, h_gt = gt_bboox[:4]
            x_gt_min, y_gt_min, x_gt_max, y_gt_max = cx_gt*target_size_w, cy_gt*target_size_h, w_gt*target_size_w, h_gt*target_size_h
            draw.rectangle(((x_min, y_min), (x_max, y_max)), outline="yellow", width=4)
            draw.rectangle(((x_gt_min, y_gt_min), (x_gt_max, y_gt_max)), outline="green", width=3)
            draw.text((x_min, y_min-10), f'Index {idx} - {objectness:1.2f}')

        if len(self.labels) > 0:
            file_index = self.images.index(filepath)
            gt_data = read_yolo_label_file(self.labels[file_index])

            for idx, obj in self.idx_obj_map.items():
                best_iou = 0
                best_id = None
                bbox = obj[0]
                for gt_bboox in gt_data:
                    iou = calculate_iou([bbox[0]/float(target_size_w), bbox[1]/float(target_size_h), bbox[2]/float(target_size_w), bbox[3]/float(target_size_h)], gt_bboox[:4])
                    if iou > best_iou and iou > 0.7:
                        best_iou = iou
                        best_id = gt_bboox[4]

                iou_with_img = calculate_iou([bbox[0]/float(target_size_w), bbox[1]/float(target_size_h), bbox[2]/float(target_size_w), bbox[3]/float(target_size_h)], 
                                             [0, 0, 1, 1])
                if best_id is not None and iou_with_img < 0.9:
                    best_name = self.class_names[best_id]
                    if best_name not in self.class_embeddings:
                        self.class_embeddings[best_name] = []
                    self.class_embeddings[best_name].append(self.class_embeddings_full[idx])

            self.update_class_display()

        self.image_display = ImageTk.PhotoImage(img)
        self.image_canvas.create_image(0, 0, anchor="nw", image=self.image_display)

        self.status_label.configure(text="Done processing the image.", foreground="green")
        self.busy_processing_image = False

    def validate_float(self, value):
        if value == "": 
            return True
        try:
            float(value)  
            return True
        except ValueError:
            return False
        
    def validate_int(self, value):
        if value == "": 
            return True
        try:
            int(value)  
            return True
        except ValueError:
            return False

    def copy_image_with_copy_suffix(self, src_path):
        # Get directory, base name, and extension of the file.
        dir_name, base_name = os.path.split(src_path)
        name, ext = os.path.splitext(base_name)
        
        # Create a new file name by appending '_copy' before the extension.
        new_file_name = f"{name}_copy{ext}"
        dst_path = os.path.join(dir_name, new_file_name)
        
        # Copy the file.
        shutil.copy(src_path, dst_path)
        print(f"Copied image to: {dst_path}")
        return dst_path


    def process_image_bottom(self, filepath, update_gui=True, conf_thresh=0.2):

        if len(self.class_embeddings) == 0:
            print("No embeddings are loaded.")
            return
        
        self.busy_processing_image = True

        merging_mode = self.selected_last_op.get()

        filepath = self.copy_image_with_copy_suffix(filepath)
        indexes, scores, bboxes, self.class_embeddings_full = self.icod.process_with_embeddings(filepath, self.class_embeddings, list(self.class_embeddings.keys()), conf_thresh=conf_thresh, avg_count = int(self.text_entry1.get()), merging_mode=merging_mode)

        target_size_w = self.image_canvas_bottom.winfo_width()
        target_size_h = self.image_canvas_bottom.winfo_height()
        if target_size_w < 200:
            target_size_w = 200
        if target_size_h < 200:
            target_size_h = 200

        img = Image.open(filepath)
        orig_size = img.size
        h_ratio = orig_size[0] / orig_size[1]

        img = img.resize((target_size_w, target_size_h), Image.LANCZOS)  # Resize for display purposes
        
        if update_gui:
            draw = ImageDraw.Draw(img)
            self.idx_bbox_map = {}
            classes = []
            class_names = self.class_names1
            for idx, (id, box, score) in enumerate(zip(indexes, bboxes, scores)):
                
                cx, cy, w, h = box
                cx, cy, w, h = cx*target_size_w, cy*h_ratio*target_size_h, w*target_size_w, h*h_ratio*target_size_h
                
                x_min = int(cx-w/2)
                y_min = int(cy-h/2)
                x_max = int(cx+w/2)
                y_max = int(cy+h/2)
                self.idx_bbox_map[idx] = [x_min, y_min, x_max, y_max]
                draw.rectangle(((x_min, y_min), (x_max, y_max)), outline="green", width=5)
                # draw.text((x_min, y_min-10), f'{class_names[id]}: {score*100:.1f}%')

            # Display processed image in bottom right panel
            self.image_display_bottom = ImageTk.PhotoImage(img)
            self.image_canvas_bottom.create_image(0, 0, anchor="nw", image=self.image_display_bottom)

        self.status_label.configure(text="Done processing the image.", foreground="green")
        self.busy_processing_image = False
        
        return indexes, bboxes, scores, img

    def process_dataset(self):
        
        if self.icod is None:
            print("Could not process dataset without the model loaded.")
            return
            
        self.busy_processing_dataset = True

        t0 = time.time()
        for i, filepath in enumerate(self.images):
            self.status_label.configure(text=f"Processing dataset {i}/{len(self.images)}.", foreground="red")
            self.process_image(filepath, "")
        print(f"Total time for processing dataset: {time.time()-t0:.3f}")

        self.status_label.configure(text="Done processing the dataset.", foreground="green")
        self.busy_processing_dataset = False

    def process_dataset1(self):
        
        if self.icod is None:
            print("Could not process dataset without the model loaded.")
            return
            
        self.busy_processing_dataset = True

        all_bboxes = []
        all_gts = []

        merging_mode = self.selected_last_op.get()
        output_folder = f"{self.folder_path1}/results/{self.selected_mode.get()}/{merging_mode}/{self.selected_mode.get()}_k{int(self.text_entry1.get())}"
        if os.path.isdir(output_folder):
            for filename in os.listdir(output_folder):
                file_path = os.path.join(output_folder, filename)
                os.remove(file_path) 
        os.makedirs(output_folder, exist_ok=True)

        for i, filepath in enumerate(self.images_bottom):
            self.status_label.configure(text=f"Processing dataset {i}/{len(self.images_bottom)}.", foreground="red")
            indexes, bboxes, scores, img = self.process_image_bottom(filepath, conf_thresh=float(self.text_entry.get()))

            all_bboxes.extend([[box[0], box[1], box[2], box[3], id] for box, id in zip(bboxes, indexes)])

            # file_index = self.images_bottom.index(filepath)
            # gt_data = read_yolo_label_file(self.labels_bottom[file_index])
            # all_gts.extend(gt_data)

            image_file = os.path.basename(filepath)
            output_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}.txt")
            
            with open(output_path, 'w') as f:
                for box, label, score in zip(bboxes, indexes, scores):
                    cx, cy, w, h = box
                    x_center = cx 
                    y_center = cy 
                    width = w 
                    height = h 
                    f.write(f"{label} {x_center} {y_center} {width} {height} {score}\n")
            
            img.save(output_path.replace(".txt", ".jpg"))
        
        self.status_label.configure(text="Done processing the dataset.", foreground="green")
        self.busy_processing_dataset = False

        # plot_confusion_matrix(all_gts, all_bboxes,  self.class_names1)

    def on_canvas_click(self, event):
        click_x, click_y = event.x, event.y
        print(click_x, click_y)
        smallest_bbox_idx = None
        smallest_bbox_area = float('inf')

        for idx, (x_min, y_min, x_max, y_max) in self.idx_bbox_map.items():
            if x_min <= click_x <= x_max and y_min <= click_y <= y_max:
                bbox_area = (x_max - x_min) * (y_max - y_min)
                if bbox_area < smallest_bbox_area:
                    smallest_bbox_area = bbox_area
                    smallest_bbox_idx = idx

        if smallest_bbox_idx is not None:
            self.prompt_for_class(smallest_bbox_idx)

    def prompt_for_class(self, bbox_idx):
        # Custom dialog window
        dialog = Toplevel(self.root)
        dialog.title("Class Name")
        dialog.geometry("300x200")
        
        # Ensure the dialog is visible before making it modal
        dialog.wait_visibility()  
        dialog.grab_set()  # Make the dialog modal

        selected_class = StringVar()  # Variable to hold the selected class name

        # Instructions
        Label(dialog, text="Choose an existing class or enter a new one:").pack(pady=5)

        # Add buttons for each existing class
        for class_name in self.class_embeddings.keys():
            Button(dialog, text=class_name, command=lambda cn=class_name: selected_class.set(cn)).pack(pady=2)

        # Entry box for new class name
        new_class_entry = Entry(dialog)
        new_class_entry.pack(pady=10)
        
        # OK button to confirm new class name entry
        def on_ok():
            if not selected_class.get():  # Only set the entry text if no button was clicked
                selected_class.set(new_class_entry.get())
            dialog.destroy()

        Button(dialog, text="OK", command=on_ok).pack(pady=5)
        
        # Wait for the dialog to close
        self.root.wait_window(dialog)

        # Get the final class name and proceed if it's non-empty
        class_name = selected_class.get().strip()
        if class_name:
            # Append the embedding to the corresponding class list in the dictionary
            if class_name not in self.class_embeddings:
                self.class_embeddings[class_name] = []
            self.class_embeddings[class_name].append(self.class_embeddings_full[bbox_idx])

            # Update the displayed class information
            self.update_class_display()

    def update_class_display(self):
        # Generate text with class names and the number of embeddings per class
        display_text = "\n".join([f"{class_name}: {len(embeddings)}" for class_name, embeddings in self.class_embeddings.items()])
        self.class_display.config(text=display_text)

    def save_class_embeddings(self):
        # Convert numpy arrays to lists for JSON serialization
        serializable_data = {k: [embedding.tolist() for embedding in v] for k, v in self.class_embeddings.items()}
        
        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")], initialdir="~/Documents")
        if file_path:
            with open(file_path, "w") as file:
                json.dump(serializable_data, file)
            self.status_label.configure(text="Classes saved successfully.", foreground="green")

    def load_class_embeddings(self):

        mode = self.selected_mode.get()
        if mode == "image_conditioned":
            file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")], initialdir="~/Documents")
            if file_path:
                with open(file_path, "r") as file:
                    # Load data and convert lists back to numpy arrays
                    loaded_data = json.load(file)
                    self.class_embeddings = {k: [np.array(embedding) for embedding in v] for k, v in loaded_data.items()}
                
                self.update_class_display()  # Update panel to reflect loaded data
                self.status_label.configure(text="Classes loaded successfully.", foreground="green")
        elif mode == "text_conditioned":
            file_path = filedialog.askopenfilename(filetypes=[("YAML files", "*.yaml")], initialdir="~/Documents")
            with open(file_path, 'r') as file:
                data = yaml.safe_load(file)

            queries_dict = {str(k): str(v) for k, v in data.items()}

            embeddings = self.icod.tokenize_queries(queries_dict.values())

            self.class_embeddings = {}
            for i, q in enumerate(queries_dict.keys()):

                self.class_embeddings[q] = [embeddings[i]]

            self.update_class_display()



root = tk.Tk()
# noVNC + Xvfb: default Tk size is tiny on a 1920x1080 framebuffer — looks like a black screen.
if os.environ.get("OPEN_ROBO_NOVNC"):
    root.geometry("1920x1080+0+0")
app = ImageAnnotatorApp(root)
root.mainloop()
