import os
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from data.utils.images_transforms import SoftLabelResize
from data.utils import resize_right, interp_methods
from data.video.video_dataset import compute_homo_cv


class ImageDataset(Dataset):
    def __init__(
            self, 
            data_folder,  
            video_indices, 
            DATASET,
            data_cfg,
            training=True, 
            joint_transforms=None,
            img_transforms=None,
            segmentation_transforms=None,
            ):
        super(ImageDataset, self).__init__()
        self.data_folder = data_folder
        self.n_classes = DATASET.n_classes
        self.ignore_index = DATASET.ignore_index
        self.label_extension = DATASET.label_extension
        self.img_extension = DATASET.img_extension
        self.frame_folder = DATASET.frame_folder
        self.mask_folder = DATASET.mask_folder
        self.convert_labels = DATASET.convert_labels
        self.crop_size = data_cfg["crop_size"]
        self.labeled_frames_per_vid = data_cfg.get("labeled_frames_per_vid", None)
        self.val_skip_frames = data_cfg.get("val_skip_frames", 1)
        self.min_vid_len = data_cfg.get("min_vid_len", 0)
        self.soft_labels = data_cfg.get("soft_labels", False)
        self.training = training

        self.samples = []
        v_idx = 0
        for v_name in video_indices:
            frames = os.listdir(os.path.join(data_folder, v_name, self.frame_folder))
            frames.sort()

            # Check if length of video is sufficient (for video model: fair comparison)
            if len(frames) < self.min_vid_len + 1:
                continue
            
            for (i, f) in enumerate(frames):
                if self.islabeled(f, v_name, i, len(frames)):
                    sample = (f, v_name)
                    self.samples.append(sample)

            v_idx += 1

        self.joint_transforms = joint_transforms
        self.img_transforms = img_transforms
        self.segmentation_transforms = segmentation_transforms

    def __getitem__(self, index):
        sample = self.samples[index]
        (f, v_name) = sample
        label_name = self.name_to_labelname(f)

        # Read images
        frame = self.read_image(f, v_name, self.data_folder)
        label = self.read_mask(label_name, v_name, self.data_folder)

        # Data augmentation and transforms
        if self.joint_transforms is not None:
            transformed = self.joint_transforms(image=frame, mask=label)
            frame, label = transformed['image'], transformed['mask']
        if self.img_transforms is not None:
            frame = self.img_transforms(image=frame)
            frame = frame["image"]
        if self.segmentation_transforms is not None:
            label = self.segmentation_transforms(label)

        return frame, label

    def __len__(self):
        return len(self.samples)

    def islabeled(self, f, v_name, idx_in_vid, len_vid):
        if not self.training:
            if idx_in_vid%self.val_skip_frames == 0:
                labelname = self.name_to_labelname(f)
                return labelname in os.listdir(os.path.join(self.data_folder, v_name, self.mask_folder))
            else:
                return False
        if self.labeled_frames_per_vid is None:
            labelname = self.name_to_labelname(f)
            if labelname in os.listdir(os.path.join(self.data_folder, v_name, self.mask_folder)):
                return True
        else:
            n_labeled_frames = min(self.labeled_frames_per_vid, len_vid)
            labels_idx = np.linspace(0,len_vid-1, n_labeled_frames, dtype=int)
            if idx_in_vid in labels_idx:
                return True
        return False

    def name_to_labelname(self, name):
        return name.split(".")[0] + self.label_extension
    
    def labelname_to_name(self, labelname):
        return labelname.split(".")[0] + self.img_extension
    
    def read_image(self, img, v_name, data_folder):
        i = Image.open(os.path.join(data_folder, v_name, self.frame_folder, img))
        i = i.resize((self.crop_size[1], self.crop_size[0]), Image.BILINEAR)
        i = np.array(i)
        return i
    
    def read_mask(self, mask, v_name, data_folder):
        l = Image.open(os.path.join(data_folder, v_name, self.mask_folder, mask))
        if self.soft_labels:
            l = SoftLabelResize(self.n_classes, self.crop_size, self.convert_labels, self.ignore_index, "bilinear")(l)
        else:
            l = l.resize((self.crop_size[1], self.crop_size[0]), Image.NEAREST)
            l = np.array(l)
        return l
    


class ImageInferenceDataset(Dataset):
    def __init__(
            self, 
            data_folder,  
            video_indices, 
            DATASET,
            data_cfg,
            img_transforms=None,
            segmentation_transforms=None,
            val_skip_frames=1
            ):
        super(ImageInferenceDataset, self).__init__()
        self.data_folder = data_folder
        self.n_classes = DATASET.n_classes
        self.label_extension = DATASET.label_extension
        self.img_extension = DATASET.img_extension
        self.frame_folder = DATASET.frame_folder
        self.mask_folder = DATASET.mask_folder
        self.reg_folder = DATASET.reg_folder
        self.crop_size = data_cfg["crop_size"]
        self.min_vid_len = data_cfg.get("min_vid_len", 0)
        self.val_skip_frames = val_skip_frames
        self.opencv_homos = data_cfg.get("opencv_homos", False)
        self.opencv_model_type = data_cfg.get("opencv_model_type", "sift")

        self.videos = []
        for v_name in video_indices:
            frames = os.listdir(os.path.join(data_folder, v_name, self.frame_folder))
            frames.sort()

            # Check if length of video is sufficient (for video model: fair comparison)
            if len(frames) < self.min_vid_len + 1:
                continue
            
            self.videos.append((frames, v_name))

        self.img_transforms = img_transforms
        self.segmentation_transforms = segmentation_transforms

    def __getitem__(self, index):
        frames_names, v_name = self.videos[index]
        frames_names = [f for (i, f) in enumerate(frames_names) if self.isinfered(i)]
        labels_names = [self.name_to_labelname(f) for (i, f) in enumerate(frames_names) if self.islabeled(f, v_name)]

        # Read images
        frames = [self.read_image(f, v_name, self.data_folder) for f in frames_names]
        labels = [self.read_mask(l, v_name, self.data_folder) for l in labels_names]

        # Precomputed homo
        #if self.precomputed_homos:
        #    homos_names = [os.path.join(self.data_folder, v_name, self.reg_folder, f.split(".")[0] + ".pt") for f in frames_names[1:]]
        #    homos = [torch.load(h, weights_only=True) for h in homos_names]
        #    homos = [None] + homos
        #else:
        #    homos = [None] +  [torch.tensor(compute_homo_cv(frames[i-1], frame), dtype=torch.float32) for (i, frame) in enumerate(frames[1:])]

        if self.opencv_homos:
            homos = [None] + [torch.tensor(compute_homo_cv(frames[i-1], frames[i], self.opencv_model_type), dtype=torch.float32).unsqueeze(0) for i in range(1, len(frames))]
        else:
            homos = [None for _ in frames]

        # Transforms
        if self.img_transforms is not None:
            frames = [self.img_transforms(image=frame)["image"] for frame in frames]
        if self.segmentation_transforms is not None:
            labels = [self.segmentation_transforms(Image.fromarray(label)) for label in labels]

        return frames_names, frames, labels_names, labels, v_name, homos

    def __len__(self):
        return len(self.videos)
    
    def isinfered(self, idx_in_vid):
        return idx_in_vid%self.val_skip_frames == 0 or idx_in_vid%self.val_skip_frames == 1
    
    def islabeled(self, f, v_name):
        labelname = self.name_to_labelname(f)
        return labelname in os.listdir(os.path.join(self.data_folder, v_name, self.mask_folder))

    def name_to_labelname(self, name):
        return name.split(".")[0] + self.label_extension
    
    def labelname_to_name(self, labelname):
        return labelname.split(".")[0] + self.img_extension
    
    def read_image(self, img, v_name, data_folder):
        i = Image.open(os.path.join(data_folder, v_name, self.frame_folder, img))
        i = i.resize((self.crop_size[1], self.crop_size[0]), Image.BILINEAR)
        i = np.array(i)
        return i
    
    def read_mask(self, mask, v_name, data_folder):
        l = Image.open(os.path.join(data_folder, v_name, self.mask_folder, mask))
        l = l.resize((self.crop_size[1], self.crop_size[0]), Image.NEAREST)
        l = np.array(l)
        return l
    

class ImageLogitsDataset(Dataset):
    def __init__(
            self, 
            data_folder,  
            video_indices, 
            DATASET,
            data_cfg,
            training=True, 
            joint_transforms=None,
            img_transforms=None,
            segmentation_transforms=None,
            ):
        super(ImageLogitsDataset, self).__init__()
        self.data_folder = data_folder
        self.logit_folder = data_cfg["logits_folder"]
        self.n_classes = DATASET.n_classes
        self.ignore_index = DATASET.ignore_index
        self.label_extension = ".npz"
        self.img_extension = DATASET.img_extension
        self.frame_folder = DATASET.frame_folder
        self.mask_folder = ""
        self.convert_labels = DATASET.convert_labels
        self.crop_size = data_cfg["crop_size"]
        self.labeled_frames_per_vid = data_cfg.get("labeled_frames_per_vid", None)
        self.val_skip_frames = data_cfg.get("val_skip_frames", 1)
        self.train_skip_frames = data_cfg.get("train_skip_frames", None)
        self.min_vid_len = data_cfg.get("min_vid_len", 0)
        self.training = training

        self.samples = []
        v_idx = 0
        for v_name in video_indices:
            frames = os.listdir(os.path.join(data_folder, v_name, self.frame_folder))
            frames.sort()

            # Check if length of video is sufficient (for video model: fair comparison)
            if len(frames) < self.min_vid_len + 1:
                continue
            
            for (i, f) in enumerate(frames):
                if self.islabeled(f, v_name, i, len(frames)):
                    sample = (f, v_name)
                    self.samples.append(sample)
            v_idx += 1

        self.joint_transforms = joint_transforms
        self.img_transforms = img_transforms
        self.segmentation_transforms = segmentation_transforms

    def __getitem__(self, index):
        sample = self.samples[index]
        (f, v_name) = sample
        label_name = self.name_to_labelname(f)

        # Read images
        frame = self.read_image(f, v_name, self.data_folder)
        label = self.read_mask(label_name, v_name, self.logit_folder)

        # Data augmentation and transforms
        if self.joint_transforms is not None:
            transformed = self.joint_transforms(image=frame, mask=label)
            frame, label = transformed['image'], transformed['mask']
        if self.img_transforms is not None:
            frame = self.img_transforms(image=frame)
            frame = frame["image"]
        if self.segmentation_transforms is not None:
            label = self.segmentation_transforms(label)

        return frame, label

    def __len__(self):
        return len(self.samples)

    def islabeled(self, f, v_name, idx_in_vid, len_vid):
        if not self.training:
            if idx_in_vid%self.val_skip_frames == 0:
                return True
            else:
                return False
        if self.train_skip_frames is not None:
            labels_idx = np.arange(1, len_vid, step=self.train_skip_frames)
            if idx_in_vid in labels_idx:
                return True
            else:
                return False
        if self.labeled_frames_per_vid is None:
                return True
        else:
            n_labeled_frames = min(self.labeled_frames_per_vid, len_vid)
            #labels_idx = np.linspace(0,len_vid-1, n_labeled_frames, dtype=int)
            labels_idx = np.arange(1, n_labeled_frames*(len_vid//n_labeled_frames)+1, step=len_vid//n_labeled_frames)
            if idx_in_vid in labels_idx:
                return True
        return False

    def name_to_labelname(self, name):
        return name.split(".")[0] + self.label_extension
    
    def labelname_to_name(self, labelname):
        return labelname.split(".")[0] + self.img_extension
    
    def read_image(self, img, v_name, data_folder):
        i = Image.open(os.path.join(data_folder, v_name, self.frame_folder, img))
        i = i.resize((self.crop_size[1], self.crop_size[0]), Image.BILINEAR)
        i = np.array(i)
        return i
    
    def read_mask(self, mask, v_name, data_folder):
        l = np.load(os.path.join(data_folder, v_name, self.mask_folder, mask))["arr_0"].transpose(1,2,0)
        l = resize_right.resize(l, out_shape=(self.crop_size[0], self.crop_size[1]), interp_method=interp_methods.linear, support_sz=2, antialiasing=True)
        return l