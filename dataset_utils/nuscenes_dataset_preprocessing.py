import os
import json
from collections import defaultdict
from typing import List, Dict, Iterable

import numpy as np

import torch
from torch.utils.data import Dataset

import cv2
import albumentations

import transformers
from transformers import BertTokenizer, ViTImageProcessor, AutoTokenizer, DetrImageProcessor

from .enums import Enums
from .data_augmentation import MosaicAugmentation

class NuScenesObjectDetectDataset(Dataset):    
    
    def __init__(self, table_blob_paths:list, root_dir:str):
        
        self.sample_tokens = []
        self.tables = {}
        self.table_blob_paths = table_blob_paths
        
        self.root_dir = root_dir
                
        for blob_table_path in table_blob_paths:
            self.parse_blob_table(blob_table_path)
    
    def parse_blob_table(self, blob_table_path:str):
        
        table = json.load(open(blob_table_path))        

        for sample_token in table:
            if table[sample_token]["labels_2d_cam_front"]:
                self.sample_tokens.extend([sample_token])
        
        # self.sample_tokens.extend([sample_token for sample_token in table])
        self.tables.update(table)

    def __len__(self):
        return len(self.sample_tokens)
    
    def __getitem__(self, idx):
        
        sample_token = self.sample_tokens[idx]
        lidar_top_fp = self.tables[sample_token]['lidar_top_fp']
        cam_front_fp = self.tables[sample_token]['cam_front_fp']
        labels_2d_cam_front = self.tables[sample_token]['labels_2d_cam_front']
        
        # splitting because 
        # ../data/nuscenes/trainval04_blobs_US/samples/CAM_FRONT/filename.jpg
        lidar_top_fp = f"{self.root_dir}/{'/'.join(lidar_top_fp.split('/')[3:])}"
        cam_front_fp = f"{self.root_dir}/{'/'.join(cam_front_fp.split('/')[3:])}"
        
        return {
            'sample_token':sample_token, 
            'lidar_top_fp':lidar_top_fp, 
            'cam_front_fp':cam_front_fp, 
            'labels_2d_cam_front':labels_2d_cam_front
        }


class NuScenesDETRPreprocessing:
    def __init__(self, image_resize:tuple=(800, 800), 
                original_size:tuple=(900, 1600),
                apply_augmentation=False, 
                tokenize_fine_labels:bool=True, eval_mode:bool=False):
        
        self.image_preprocessor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.apply_augmentation = apply_augmentation
        
        self.image_resize = image_resize
        self.original_size = original_size

        self.original_width = self.original_size[1]
        self.original_height = self.original_size[0]
        
        self.resized_width = self.image_resize[1]
        self.resized_height = self.image_resize[0]
        self.tokenize_fine_labels = tokenize_fine_labels

        self.eval_mode = eval_mode

        if self.tokenize_fine_labels:
            self.text_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        else:
            self.text_tokenizer = None

        if self.apply_augmentation:
            self.transformation = albumentations.Compose([
                albumentations.RandomBrightnessContrast(p=0.5),
                albumentations.HorizontalFlip(p=0.5),
                albumentations.RandomResizedCrop(
                    size=(self.resized_height, self.resized_width),
                    scale=(0.5, 1.0),   # don’t crop below 50% of image
                    ratio=(0.8, 1.25),  # allow mild aspect ratio jitter
                    p=0.5
                ),
                albumentations.LongestMaxSize(max_size=max(self.image_resize)), # redundant with RandomResizedCrop.
                albumentations.PadIfNeeded(
                    min_height=self.resized_height,
                    min_width=self.resized_width,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0
                ), 
                albumentations.CenterCrop(height=self.resized_height, width=self.resized_width, always_apply=True)                
            ], 
            bbox_params=albumentations.BboxParams(format='pascal_voc', label_fields=['class_labels']))
        else:
            self.transformation = albumentations.Compose(
                [albumentations.LongestMaxSize(max_size=max(self.image_resize)),
                albumentations.PadIfNeeded(
                    min_height=self.resized_height,
                    min_width=self.resized_width,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0
                ), 
                albumentations.CenterCrop(height=self.resized_height, width=self.resized_width, always_apply=True)
                ],                
                bbox_params=albumentations.BboxParams(format='pascal_voc', label_fields=['class_labels'])
            )           
            
        self.image_only_transformation = albumentations.Compose(
            [albumentations.Resize(height=self.resized_height, width=self.resized_width, always_apply=True)]
        )            
    
    def preprocess_labels(self, labels_2d_cam_front):
        """Extract Pascal VOC bboxes and class IDs."""
        class_labels = []
        bboxes_2d = []
        fine_labels = []
        fine_labels_str = []
        
        for label_data in labels_2d_cam_front:
            attribute = label_data['category_name']
            visibility = label_data['visibility_token']
            if attribute in Enums.NUSCENES_TO_GENERAL_CLASSES and visibility in ["3", "4"]:
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

    def transform_sample(self, image:np.array, label_bboxes:np.array=None, class_labels:np.array=None):

        if label_bboxes is not None and class_labels is not None:        
            transformed_dict = self.transformation(
                image=image, bboxes=label_bboxes, class_labels=class_labels
            )
        else:
            transformed_dict = self.image_only_transformation(
                image=image
            )
        return transformed_dict    

    def __call__(self, batch):
        
        images = []
        annotations = []
        batch_fine_labels = []
        batch_fine_labels_str = []

        for idx, data in enumerate(batch):
            cam_front_fp = data['cam_front_fp']
            labels_2d_cam_front = data['labels_2d_cam_front']            

            image = cv2.imread(cam_front_fp)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            class_labels, fine_labels, fine_labels_str, bboxes_2d = self.preprocess_labels(labels_2d_cam_front)

            transformed_dict = self.transform_sample(
                image, bboxes_2d, class_labels
            )

            image = transformed_dict['image']
            bboxes_2d = transformed_dict['bboxes']
            class_labels = transformed_dict['class_labels']

            fine_labels = torch.tensor(fine_labels, dtype=torch.int32)

            batch_fine_labels.extend(fine_labels)
            
            #used for tokenization
            batch_fine_labels_str.extend(fine_labels_str)
            
            coco_boxes = []

            for bbox, label in zip(bboxes_2d, class_labels):
                x_min, y_min, x_max, y_max = bbox
                w = x_max - x_min
                h = y_max - y_min

                coco_boxes.append({
                    "bbox": [
                        x_min,
                        y_min,
                        w, 
                        h
                    ],
                    "category_id": int(label),
                    "area": w*h
                })

            annotations.append({
                "image_id": idx,
                "annotations": coco_boxes
            })
            images.append(image)

        encoding = self.image_preprocessor(
            images=images,
            annotations=annotations,
            return_tensors="pt"
        )

        coarse_labels = encoding['labels']
        pixel_values = encoding['pixel_values']
        encoding = self.image_preprocessor.pad(pixel_values, return_tensors="pt")

        pixel_mask = encoding.pixel_mask
        pixel_values = encoding.pixel_values

        try:
            batch_fine_labels = torch.stack(batch_fine_labels, dim=0)
        except:
            print(annotations, batch_fine_labels)
            for data in batch:
                sample_token = data['sample_token']
                print(sample_token)
            exit(1)

        if self.tokenize_fine_labels:
            tokenized_dict = self.text_tokenizer(batch_fine_labels_str, padding=True, return_tensors="pt")
            fine_labels_input_ids, fine_labels_attention_mask = tokenized_dict['input_ids'], tokenized_dict['attention_mask']

            # total_num_labels = sum of num_labels (each batch_item)
            return {
                "pixel_values":pixel_values,
                "pixel_mask":pixel_mask,
                "coarse_labels":coarse_labels,
                "fine_labels":batch_fine_labels, #(total_num_labels)
                "fine_labels_input_ids":fine_labels_input_ids, #(total_num_labels, max_len)
                "fine_labels_attention_mask":fine_labels_attention_mask #(total_num_labels, max_len)
            }

        return {
            "pixel_values":pixel_values,
            "pixel_mask":pixel_mask,
            "labels":coarse_labels,
            "fine_labels":batch_fine_labels
        }

class NuScenesDETRAugPreprocessing(NuScenesDETRPreprocessing):
    
    def __init__(self, image_resize = (800, 800), original_size = (900, 1600), 
                apply_augmentation=False, tokenize_fine_labels = True, eval_mode = False):
        super().__init__(image_resize, original_size, apply_augmentation, tokenize_fine_labels, eval_mode)
        
        self.mosaic_augmentation = MosaicAugmentation(
            self.resized_width, self.resized_height
        )
            
    def __call__(self, batch):
        
        augmentation_data_items = defaultdict(lambda:defaultdict())
        
        images = []
        annotations = []
        batch_fine_labels = []
        batch_fine_labels_str = []

        for idx, data in enumerate(batch):
            cam_front_fp = data['cam_front_fp']
            labels_2d_cam_front = data['labels_2d_cam_front']

            image = cv2.imread(cam_front_fp)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            class_labels, fine_labels, fine_labels_str, bboxes_2d = self.preprocess_labels(labels_2d_cam_front)

            transformed_dict = self.transform_sample(
                image, bboxes_2d, class_labels
            )                        

            if (idx + 1) % 4 == 0:
                augmentation_data_items[idx] = {
                    "image":transformed_dict["image"],
                    "bboxes_2d": transformed_dict['bboxes'],
                    "class_labels": transformed_dict['class_labels'],
                    "cam_front_fp":cam_front_fp
                }
                mosaic_image, quadrant_bboxes, quadrant_labels = self.mosaic_augmentation(augmentation_data_items, 
                                                                                        use_random_quadrant_dimensions=False)
                coco_boxes = []
                for bbox, label in zip(quadrant_bboxes, quadrant_labels):
                    x_min, y_min, x_max, y_max = bbox
                    w = x_max - x_min
                    h = y_max - y_min

                    coco_boxes.append({
                        "bbox": [
                            x_min,
                            y_min,
                            w, 
                            h
                        ],
                        "category_id": int(label),
                        "area": w*h
                    })

                annotations.append({
                    "image_id": idx,
                    "annotations": coco_boxes
                })
                images.append(mosaic_image)

                augmentation_data_items = defaultdict(lambda:defaultdict())

            else:
                augmentation_data_items[idx] = {
                    "image":transformed_dict["image"],
                    "bboxes_2d": transformed_dict['bboxes'],
                    "class_labels": transformed_dict['class_labels'],
                    "cam_front_fp":cam_front_fp
                }
                
        encoding = self.image_preprocessor(
            images=images,
            annotations=annotations,
            return_tensors="pt"
        )
        
        coarse_labels = encoding['labels']
        pixel_values = encoding['pixel_values']
        encoding = self.image_preprocessor.pad(pixel_values, return_tensors="pt")

        pixel_mask = encoding.pixel_mask
        pixel_values = encoding.pixel_values
        
        return {
            "pixel_values":pixel_values,
            "pixel_mask":pixel_mask,
            "coarse_labels":coarse_labels
        }

if __name__ == "__main__":
    dataset =NuScenesObjectDetectDataset(
        table_blob_paths=['../data/trainval01_blobs_US/tables.json'],
        root_dir='../data/'
    )

    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=4,
        # collate_fn=NuScenesDETRPreprocessing()
        collate_fn=NuScenesDETRAugPreprocessing(),
        shuffle=True
    )

    for data in dataloader:
        print(data["labels"])
        exit(1)
