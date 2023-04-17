import numpy as np
import torch
import json
from utils import rle_encode, rle_decode, mask_to_bbox
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

    def add_auto_detect_annot(self, frame_id, masks):
        self.masks[frame_id] = masks
        self.labels[frame_id] = ["unspecified",] * len(masks)

    def set_annotations(self, masks, labels, img_embed):
        self.masks = masks
        self.labels = labels
        self.img_embed = img_embed

    def add_img_embed(self, frame_id, img_embed):
        self.img_embed[frame_id] = img_embed

    def get_mask(self, frame_id):
        if frame_id in self.masks:
            return np.array(self.masks[frame_id])
        else:
            return None
    
    def get_labels(self, frame_id):
        if frame_id in self.labels:
            return self.labels[frame_id]
        else:
            return None
    
    def set_label(self, frame_id, label_id, label):
        if self.labels[frame_id] is not None:
            self.labels[frame_id][label_id] = label

    def delete_mask(self, frame_idx, annot_idx):
        print(annot_idx)
        del self.masks[frame_idx][annot_idx]
        del self.labels[frame_idx][annot_idx]

    def load_annot_from_file(self, fname):
        with open(fname, "r") as file:
            annots = json.load(file)

        for annot in annots:
            frame_id = annot["image_id"]
            masks = rle_decode(annot["masks"])
            self.masks[frame_id] = masks
            self.labels[frame_id] = annot["labels"]

    def save_annotations(self, fname):
        annot = []
        for frame_id, mask in self.masks.items():
            boxes = [mask_to_bbox(255*m.astype(np.uint8))[0].tolist() for m in mask]
            masks = rle_encode(mask)
            
            frame_annot = {"image_id":frame_id,
                           "masks": masks,
                           "boxes": boxes,
                           "labels": self.labels[frame_id]}
            annot.append(frame_annot)

        # save data dict as json
        with open(fname, "w") as file:
            json.dump(annot, file)

    def load_embed_from_torch(self, fname):
        img_embed = torch.load(fname)
        for i, embed in enumerate(img_embed):
            self.img_embed[i] = embed
    
    def save_embed(self, fname):
        img_embed = []
        for frame_id in self.img_embed:
            img_embed.append(self.img_embed[frame_id])
        torch.save(img_embed, fname)