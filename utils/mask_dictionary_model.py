import numpy as np
import json
import torch
import copy
import os
import cv2
from dataclasses import dataclass, field

@dataclass
class MaskDictionaryModel:
    mask_name:str = ""
    mask_height: int = 1080
    mask_width:int = 1920
    promote_type:str = "mask"
    labels:dict = field(default_factory=dict)

    def add_new_frame_annotation(self, mask_list, box_list, label_list, background_value = 0):
        mask_img = torch.zeros(mask_list.shape[-2:])
        anno_2d = {}
        for idx, (mask, box, label) in enumerate(zip(mask_list, box_list, label_list)):
            final_index = background_value + idx + 1

            if mask.shape[0] != mask_img.shape[0] or mask.shape[1] != mask_img.shape[1]:
                raise ValueError("The mask shape should be the same as the mask_img shape.")
            # mask = mask
            mask_img[mask == True] = final_index
            # print("label", label)
            name = label
            box = box # .numpy().tolist()
            new_annotation = ObjectInfo(instance_id = final_index, mask = mask, class_name = name, x1 = box[0], y1 = box[1], x2 = box[2], y2 = box[3])
            anno_2d[final_index] = new_annotation

        # np.save(os.path.join(output_dir, output_file_name), mask_img.numpy().astype(np.uint16))
        self.mask_height = mask_img.shape[0]
        self.mask_width = mask_img.shape[1]
        self.labels = anno_2d

    def update_masks(self, tracking_annotation_dict, iou_threshold=0.8, objects_count=0):
        updated_masks = {}

        for seg_obj_id, seg_mask in self.labels.items():  # tracking_masks
            flag = 0 
            new_mask_copy = ObjectInfo()
            if seg_mask.mask.sum() == 0:
                continue
            
            for object_id, object_info in tracking_annotation_dict.labels.items():  # grounded_sam masks
                iou = self.calculate_iou(seg_mask.mask, object_info.mask)  # tensor, numpy
                # print("iou", iou)
                if iou > iou_threshold:
                    flag = object_info.instance_id
                    new_mask_copy.mask = seg_mask.mask
                    new_mask_copy.instance_id = object_info.instance_id
                    new_mask_copy.class_name = seg_mask.class_name
                    break
                
            if not flag:
                objects_count += 1
                flag = objects_count
                new_mask_copy.instance_id = objects_count
                new_mask_copy.mask = seg_mask.mask
                new_mask_copy.class_name = seg_mask.class_name
            updated_masks[flag] = new_mask_copy
        self.labels = updated_masks
        return objects_count

    def get_target_class_name(self, instance_id):
        return self.labels[instance_id].class_name

    def get_target_logit(self, instance_id):
        return self.labels[instance_id].logit
    
    @staticmethod
    def calculate_iou(mask1, mask2):
        # Convert masks to float tensors for calculations
        mask1 = mask1.to(torch.float32)
        mask2 = mask2.to(torch.float32)
        
        # Calculate intersection and union
        intersection = (mask1 * mask2).sum()
        union = mask1.sum() + mask2.sum() - intersection
        
        # Calculate IoU
        iou = intersection / union
        return iou

    def to_dict(self):
        return {
            "mask_name": self.mask_name,
            "mask_height": self.mask_height,
            "mask_width": self.mask_width,
            "promote_type": self.promote_type,
            "labels": {k: v.to_dict() for k, v in self.labels.items()}
        }
    
    def to_json(self, json_file):
        with open(json_file, "w") as f:
            json.dump(self.to_dict(), f, indent=4)
            
    def from_json(self, json_file):
        with open(json_file, "r") as f:
            data = json.load(f)
            self.mask_name = data["mask_name"]
            self.mask_height = data["mask_height"]
            self.mask_width = data["mask_width"]
            self.promote_type = data["promote_type"]
            self.labels = {int(k): ObjectInfo(**v) for k, v in data["labels"].items()}
        return self


@dataclass
class BoxDictionaryModel:
    box_name: str = ""
    image_height: int = 1080
    image_width: int = 1920
    promote_type: str = "bbox"
    labels: dict = field(default_factory=dict)

    def add_new_frame_annotation(self, box_list, label_list):
        anno_2d = {}
        for idx, (box, label) in enumerate(zip(box_list, label_list)):
            final_index = idx + 1
            name = label

            new_annotation = ObjectInfo(
                instance_id=final_index,
                class_name=name,
                x1=box[0],
                y1=box[1],
                x2=box[2],
                y2=box[3]
            )
            anno_2d[final_index] = new_annotation

        self.labels = anno_2d

    def update_boxes(self, tracking_annotation_dict, iou_threshold=0.8, objects_count=0):
        updated_boxes = {}

        for box_obj_id, box_info in self.labels.items():
            flag = 0
            new_box_copy = ObjectInfo()
            if (box_info.x2 - box_info.x1) * (box_info.y2 - box_info.y1) == 0:
                continue

            for object_id, object_info in tracking_annotation_dict.labels.items():
                iou = self.calculate_iou(box_info, object_info)
                if iou > iou_threshold:
                    flag = object_info.instance_id
                    new_box_copy.instance_id = object_info.instance_id
                    new_box_copy.class_name = box_info.class_name
                    new_box_copy.x1 = box_info.x1
                    new_box_copy.y1 = box_info.y1
                    new_box_copy.x2 = box_info.x2
                    new_box_copy.y2 = box_info.y2
                    break

            if not flag:
                objects_count += 1
                flag = objects_count
                new_box_copy.instance_id = objects_count
                new_box_copy.class_name = box_info.class_name
                new_box_copy.x1 = box_info.x1
                new_box_copy.y1 = box_info.y1
                new_box_copy.x2 = box_info.x2
                new_box_copy.y2 = box_info.y2
            updated_boxes[flag] = new_box_copy
        self.labels = updated_boxes
        return objects_count

    def get_target_class_name(self, instance_id):
        return self.labels[instance_id].class_name

    def get_target_logit(self, instance_id):
        return self.labels[instance_id].logit

    @staticmethod
    def calculate_iou(box1, box2):
        # Calculate intersection
        xA = max(box1.x1, box2.x1)
        yA = max(box1.y1, box2.y1)
        xB = min(box1.x2, box2.x2)
        yB = min(box1.y2, box2.y2)
        interArea = max(0, xB - xA) * max(0, yB - yA)

        # Calculate union
        box1Area = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
        box2Area = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)
        unionArea = box1Area + box2Area - interArea

        # Calculate IoU
        iou = interArea / unionArea
        return iou

    def to_dict(self):
        return {
            "box_name": self.box_name,
            "image_height": self.image_height,
            "image_width": self.image_width,
            "promote_type": self.promote_type,
            "labels": {k: v.to_dict() for k, v in self.labels.items()}
        }

    def to_json(self, json_file):
        with open(json_file, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    def from_json(self, json_file):
        with open(json_file, "r") as f:
            data = json.load(f)
            self.box_name = data["box_name"]
            self.image_height = data["image_height"]
            self.image_width = data["image_width"]
            self.promote_type = data["promote_type"]
            self.labels = {int(k): ObjectInfo(**v) for k, v in data["labels"].items()}
        return self



@dataclass
class ObjectInfo:
    instance_id:int = 0
    mask: any = None
    class_name:str = ""
    x1:int = 0
    y1:int = 0
    x2:int = 0
    y2:int = 0
    logit:float = 0.0

    def get_mask(self):
        return self.mask
    
    def get_id(self):
        return self.instance_id

    def update_box(self):
        nonzero_indices = torch.nonzero(self.mask)
        
        if nonzero_indices.size(0) == 0:
            return []
        
        y_min, x_min = torch.min(nonzero_indices, dim=0)[0]
        y_max, x_max = torch.max(nonzero_indices, dim=0)[0]
        
        # [x_min, y_min, x_max, y_max]
        bbox = [x_min.item(), y_min.item(), x_max.item(), y_max.item()]        
        self.x1 = bbox[0]
        self.y1 = bbox[1]
        self.x2 = bbox[2]
        self.y2 = bbox[3]
    
    def to_dict(self):
        return {
            "instance_id": self.instance_id,
            "class_name": self.class_name,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "logit": self.logit
        }