import numpy as np

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
        del self.masks[frame_idx][annot_idx]
        del self.labels[frame_idx][annot_idx]