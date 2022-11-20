import pyqtgraph as pg

class RectROI(pg.RectROI):
    def __init__(self, label, pos, size, centered=False, sideScalers=False, **args):
        super().__init__(pos, size, centered, sideScalers, **args)
        self.label = label

def rois_to_annot(rois: dict[list[RectROI]]) -> list[dict]:
    annot = []

    # loop over rois dict
    for idx, roi in rois.items():
        # key is the image id and value is a list of rois
        frame_annot = {"image_id": idx}
        boxes = []
        labels = []
        colors = []
        for r in roi:
            # convert roi coordinates to bbox
            x1,y1 = r.pos()
            w,h = r.size()
            bbox = [x1,y1,x1+w,y1+h]
            boxes.append(bbox)

            # get label
            labels.append(r.label)

            # get color
            color = r.pen.color()
            colors.append(color.getRgb())
        
        frame_annot["boxes"] = boxes
        frame_annot["labels"] = labels
        frame_annot["colors"] = colors
        annot.append(frame_annot)
    
    return annot

def annot_to_rois(annot: list[dict], remove_roi: callable) -> dict[list[RectROI]]:
    rois = {}
    # iteratre over each fram
    for frame_annot in annot:
        # parse data
        idx = frame_annot["image_id"]
        boxes = frame_annot["boxes"]
        labels = frame_annot["labels"]
        colors = frame_annot["colors"]

        # make roi for each bounding box
        # TODO: handle when colors is not present (e.g COCO)
        roi_list = []
        for j, b in enumerate(boxes):
            pen = pg.mkPen(width=1, color=colors[j])
            roi = RectROI(labels[j], b[0:2], [b[2]-b[0], b[3]-b[1]], pen=pen, 
                          removable=True, rotatable=False)
            roi.sigRemoveRequested.connect(remove_roi)
            roi_list.append(roi)
        rois[idx] = roi_list
    return rois

def annot_to_label(annot: list[dict]) -> dict:
    # this is mainly for updating the label list
    label_dict = {}
    
    for frame_annot in annot:
        labels = frame_annot["labels"]
        colors = frame_annot["colors"]
        
        # label_dict will iteratre over all label-color combo
        # TODO: handle when label text is the same
        for i, label in enumerate(labels):
            label_dict[label] = colors[i]
    
    return label_dict