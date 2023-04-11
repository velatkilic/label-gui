import numpy as np
import torch
import os
from pathlib import Path
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

class Model:
    def __init__(self, sam_checkpoint = None, model_type=None, device=None):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if sam_checkpoint is None:
            sam_checkpoint = os.path.join(os.getcwd(), "models" ,"sam_vit_h_4b8939.pth")
        
        if model_type is None:
            model_type = "vit_h"
        
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)
        self.predictor = SamPredictor(self.sam)
        self.auto_mask_generator = None

        self.img_shape = None
        self.max_mask_region = None
    
    def set_cached_img_embed(self, original_size=None, input_size=None, features=None):
        self.predictor.reset_image()
        self.predictor.original_size = original_size
        self.predictor.input_size = input_size
        self.predictor.features = features
        self.predictor.is_image_set = True
    
    def get_img_embed(self):
        img_embed = {}
        img_embed["original_size"] = self.predictor.original_size
        img_embed["input_size"] = self.predictor.input_size
        img_embed["features"] = self.predictor.features
        return img_embed

    def set_image(self, img):
        self.img_shape = img.shape
        self.predictor.set_image(img)

    def set_auto_mask_generator(self, params):
        self.max_mask_region = params["max_mask_region"]
        del params["max_mask_region"]
        self.auto_mask_generator = SamAutomaticMaskGenerator(model=self.sam, **params)

    def predict(self, input_point=None, input_label=None, input_mask=None, input_box=None):
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            mask_input=input_mask
        )
        return masks, scores, logits

    def generate(self, img):
        annots = self.auto_mask_generator.generate(img)
        if self.max_mask_region is not None:
            filtered_annots = []
            for annot in annots:
                if annot["area"] < self.max_mask_region:
                    filtered_annots.append(annot)
            return filtered_annots
        else:
            return annots

class Annotation:
    def __init__(self) -> None:
        self.masks = {}
        self.labels = {}
        self.img_embed = {}
    
    def add_annotation(self, frame_id, mask, label):
        if mask is None:
            return
        elif frame_id in self.masks:
            self.masks[frame_id].append(mask)
            self.labels[frame_id].append(label)
        else:
            self.masks[frame_id] = [mask]
            self.labels[frame_id] = [label]

    def add_auto_detect_annot(self, frame_id, annots):
        masks = []
        label = ["unspecified",] * len(annots)
        for annot in annots:
            masks.append(annot["segmentation"])
        
        self.masks[frame_id] = masks
        self.labels[frame_id] = label

    def add_img_embed(self, frame_id, img_embed):
        self.img_embed[frame_id] = img_embed

    def get_mask(self, frame_id):
        if frame_id in self.masks:
            return np.array(self.masks[frame_id])
        else:
            return None