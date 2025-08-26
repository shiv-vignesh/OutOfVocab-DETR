from collections import defaultdict, Counter

import math
import numpy as np

from .nuscenes_dataset_preprocessing import NuScenesObjectDetectDataset
from .enums import Enums

def _init_dataset(dataset_kwargs):

    dataset = NuScenesObjectDetectDataset(
        table_blob_paths=dataset_kwargs["table_blob_paths"],
        root_dir=dataset_kwargs["root_dir"]
    )
    
    return dataset

def preprocess_labels(labels_2d_cam_front):    
    
    class_labels = []
    bboxes_2d = []
    fine_labels = []
    fine_labels_str = []
    
    for label_data in labels_2d_cam_front:
        attribute = label_data['category_name']
        if attribute in Enums.NUSCENES_TO_GENERAL_CLASSES:
            if Enums.NUSCENES_TO_GENERAL_CLASSES[attribute] not in Enums.nuscenes_label2Id:
                continue

            fine_label_id = Enums.attr_label2id[attribute]
            fine_labels.append(fine_label_id)
            fine_labels_str.append(attribute)
            
            bbox_corners = label_data['bbox_corners']
            left, top, right, bottom = bbox_corners
            class_label = Enums.NUSCENES_TO_GENERAL_CLASSES[attribute]
            class_id = Enums.nuscenes_label2Id[class_label]
            class_labels.append(class_id)
            bboxes_2d.append([float(left), float(top), float(right), float(bottom)])
            
    return class_labels, fine_labels, fine_labels_str, bboxes_2d

def compute_class_frequencies(dataset):

    class_counter = Counter()
    total_instances = 0
    
    for data_items in dataset:
        sample_token = data_items["sample_token"]
        lidar_top_fp = data_items["lidar_top_fp"]
        cam_front_fp = data_items["cam_front_fp"]
        labels_2d_cam_front = data_items["labels_2d_cam_front"]

        class_labels, _, _, _ = preprocess_labels(labels_2d_cam_front)
        
        class_counter.update(class_labels)
        total_instances += len(class_labels)
        
    # frequency per class = count / total_instances
    class_freq = {cls: count / total_instances for cls, count in class_counter.items()}
    return class_counter, class_freq

def compute_sample_weights(dataset, class_freq:dict, 
                           image_size:tuple,
                           use_class_weights_only:bool=True):
    
    """
    weight = class_weight * area_weight * aspect_ratio_weight
    """
    
    class_weights = {}
    sample_weights = []
    
    H, W = image_size

    for data_items in dataset:
        sample_token = data_items["sample_token"]
        labels_2d_cam_front = data_items["labels_2d_cam_front"]
        class_labels, _, _, bboxes_2d = preprocess_labels(labels_2d_cam_front)
        
        weights = []
        
        for class_label, bbox in zip(class_labels, bboxes_2d):
            cls_wt = 1.0/(class_freq[class_label] + 1e-3)
            
            if use_class_weights_only:
                weights.append(cls_wt)        
            else:
                x_min, y_min, x_max, y_max = bbox
                w = x_max - x_min
                h = y_max - y_min
                
                norm_area = (w * h) / (W * H)
                area_wt = 1.0 / math.sqrt(norm_area + 1e-3)

                aspect_ratio = w / (h + 1e-3)
                if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                    ar_wt = 1.5   # boost rare tall/wide objs
                else:
                    ar_wt = 1.0
                    
                total_wt = cls_wt * area_wt * ar_wt
                weights.append(total_wt)

        weights = np.array(weights)
        weights = weights / weights.sum() #normalize weights
                
        # represent the sample by the max weight of its objects
        if weights.tolist():
            sample_weights.append(max(weights))
        else:
            sample_weights.append(0.1)  # images with no objs get a tiny weight
    
    return np.array(sample_weights, dtype=np.float32)


if __name__ == "__main__":
    
    dataset_kwargs = {
        "table_blob_paths":['../data/trainval01_blobs_US/tables.json', '../data/trainval03_blobs_US/tables.json', 
                            '../data/trainval04_blobs_US/tables.json', '../data/trainval05_blobs_US/tables.json',
                            '../data/trainval06_blobs_US/tables.json' ,'../data/trainval07_blobs_US/tables.json'],
        "root_dir":'../data/',
        "batch_size":32,
        "shuffle":True,
        "apply_augmentation":True
    }
    
    original_size:tuple=(900, 1600)
    
    dataset = _init_dataset(dataset_kwargs)    
    
    class_counter, class_freq = compute_class_frequencies(dataset)
    sample_weights = compute_sample_weights(dataset, class_freq, original_size, use_class_weights_only=False)
    
    print(len(sample_weights))
    print(len(dataset))
