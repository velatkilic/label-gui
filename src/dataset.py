import imageio
from skimage.io import imread_collection
from pathlib import Path
import glob
import os

class Dataset:
    def __init__(self, video_name=None, image_folder=None):
        if video_name is not None:
            self.reader = imageio.get_reader(video_name)
        elif image_folder is not None:
            self.reader = imread_collection(image_folder, conserve_memory=False)
        else:
            raise NotImplementedError("Dataset requires either a video or image folder input")
        self.image_folder = image_folder
        self.video_name = video_name

    def get_img(self, idx):
        if self.video_name is not None:
            img = self.reader.get_data(idx)
        else:
            img = self.reader[idx]
        return img

    def length(self):
        if self.video_name is not None:
            return self.reader.count_frames()
        elif self.image_folder is not None:
            return len(self.reader)
        else:
            return 0

def dataset_from_filename(fname: Path)->Dataset:
    # check video extensions
    ext_list = ["mp4", "avi"]
    for ext in ext_list:
        video_name = glob.glob(os.path.join(fname, "*."+ext))
        if len(video_name)>0:
            return Dataset(video_name=video_name[0])
    
    # check image extensions
    ext_list = ["tif","tiff","bmp", "png"]
    for ext in ext_list:
        image_folder = glob.glob(os.path.join(fname, "*."+ext))
        if len(image_folder)>0:
            return Dataset(image_folder=image_folder[0])
    
    return None