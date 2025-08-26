import random
import cv2
import numpy as np


class MosaicAugmentation(object):
    
    def __init__(self, image_w:int, image_h:int):
        self.image_width = image_w
        self.image_height = image_h
        self._init_quadrants_dimensions()
        
    def _init_quadrants_dimensions(self):
        h_mid = self.image_height // 2
        w_mid = self.image_width // 2
        
        self.quadrants = {
            "quad_1": {  # top-left
                "top_left": (0, 0),
                "bottom_right": (h_mid, w_mid),
            },
            "quad_2": {  # top-right
                "top_left": (0, w_mid),
                "bottom_right": (h_mid, self.image_width),
            },
            "quad_3": {  # bottom-left
                "top_left": (h_mid, 0),
                "bottom_right": (self.image_height, w_mid),
            },
            "quad_4": {  # bottom-right
                "top_left": (h_mid, w_mid),
                "bottom_right": (self.image_height, self.image_width),
            },
        }
        
    def _init_random_quadrants_dimensions(self):
        # Pick a random mosaic center anywhere inside the canvas
        cx = random.randint(int(0.25 * self.image_width), int(0.75 * self.image_width))
        cy = random.randint(int(0.25 * self.image_height), int(0.75 * self.image_height))

        self.center = (cx, cy)

        self.quadrants = {
            "quad_1": {  # top-left
                "top_left": (0, 0),
                "bottom_right": (cy, cx),
            },
            "quad_2": {  # top-right
                "top_left": (0, cx),
                "bottom_right": (cy, self.image_width),
            },
            "quad_3": {  # bottom-left
                "top_left": (cy, 0),
                "bottom_right": (self.image_height, cx),
            },
            "quad_4": {  # bottom-right
                "top_left": (cy, cx),
                "bottom_right": (self.image_height, self.image_width),
            },
        }
    
    def normalize_bboxes_2d(self, class_bboxes:list, img_w:int, img_h:int):

        for idx, bbox in enumerate(class_bboxes):
            left, top, right, bottom = bbox
            
            x_min = left/img_w
            y_min = top/img_h
            x_max = right/img_w
            y_max = bottom/img_h

            class_bboxes[idx] = (x_min, y_min, x_max, y_max)

        return class_bboxes
    
    def rescale_to_quadrant_dimensions(self, normalized_bboxes:list, quadrant:dict, ):

        (y1, x1) = quadrant["top_left"]
        (y2, x2) = quadrant["bottom_right"]

        quad_h = y2 - y1
        quad_w = x2 - x1

        for idx, bbox in enumerate(normalized_bboxes):

            x_min, y_min, x_max, y_max = bbox
            new_bbox = (
                int(x1 + x_min * quad_w),
                int(y1 + y_min * quad_h),
                int(x1 + x_max * quad_w),
                int(y1 + y_max * quad_h),
            )

            normalized_bboxes[idx] = new_bbox

        return normalized_bboxes
    
    def __call__(self, batch_data_items:dict, use_random_quadrant_dimensions:bool=True):
        
        if use_random_quadrant_dimensions:
            self._init_random_quadrants_dimensions()
        
        mosaic_image = np.zeros(shape=(self.image_height, self.image_width, 3), dtype=np.uint8)
        quadrant_bboxes = []  # Collect for visualization
        quadrant_labels = []

        for quadrant, data_items in zip(self.quadrants.values(), batch_data_items.values()):
            
            (y1, x1) = quadrant["top_left"]
            (y2, x2) = quadrant["bottom_right"]
            
            # cam_front_fp = data_items["cam_front_fp"]
            class_labels = data_items["class_labels"]
            
            quad_h = y2 - y1
            quad_w = x2 - x1
            
            resized_image = cv2.resize(data_items["image"], (quad_w, quad_h))
            mosaic_image[y1:y2, x1:x2] = resized_image
            
            h, w = data_items["image"].shape[:2]
            # print(data_items["bboxes_2d"])
            normalized_class_bboxes = self.normalize_bboxes_2d(data_items["bboxes_2d"], w, h)
            # print(data_items["bboxes_2d"])
            
            rescaled_bbox = self.rescale_to_quadrant_dimensions(normalized_class_bboxes, quadrant)
            # quadrant_bboxes.append((cam_front_fp, rescaled_bbox))
            quadrant_bboxes.extend(rescaled_bbox)
            quadrant_labels.extend(class_labels)

            
        return mosaic_image, quadrant_bboxes, quadrant_labels
        # self.plot_mosaic_with_bboxes(mosaic_image, quadrant_bboxes)
        # exit(1)
        
    def plot_mosaic_with_bboxes(self, mosaic_image, quadrant_bboxes):
        """
        Visualize the mosaic with bounding boxes for debugging.
        
        Args:
            mosaic_image (np.ndarray): The stitched mosaic image.
            quadrant_bboxes (list): List of tuples 
                [(cam_front_fp, [(x_min,y_min,x_max,y_max), ...]), ...]
        """
        vis_img = mosaic_image.copy()

        # Loop through quadrants and draw bboxes
        for cam_front_fp, bboxes in quadrant_bboxes:
            for (x_min, y_min, x_max, y_max) in bboxes:
                cv2.rectangle(vis_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                # cv2.putText(
                #     vis_img, 
                #     (x_min, max(y_min - 5, 15)), 
                #     cv2.FONT_HERSHEY_SIMPLEX, 
                #     0.5, 
                #     (0, 255, 0), 
                #     1, 
                #     cv2.LINE_AA
                # )

        cv2.imwrite('mosaic_image_sample.png', vis_img)
        
        # # Convert BGR → RGB for matplotlib
        # plt.figure(figsize=(10, 10))
        # plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        # plt.axis("off")
        # plt.savefig()        
