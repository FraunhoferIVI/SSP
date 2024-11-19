import os
import PIL.Image as Image
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from data.utils.images_transforms import SoftLabelResize
from data.utils import resize_right, interp_methods


def compute_homo_cv(last_frame, frame, model_type):
    # Detect and describe points of interest in the 2 images (idenpendantly)
    if model_type == "sift":
        detdescr = cv2.SIFT_create(8000)
    elif model_type == "akaze":
        detdescr = cv2.AKAZE_create()
    elif model_type == "orb":
        detdescr = cv2.ORB_create()
    #detdescr = cv2.AKAZE_create()
    #detdescr = cv2.ORB_create()
    kpts1, desc1 = detdescr.detectAndCompute(frame, None)
    kpts2, desc2 = detdescr.detectAndCompute(last_frame, None)

    # Match the points between the 2 images
    if model_type == "sift":
        matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    else:
        matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
    nn_matches = matcher.knnMatch(desc1, desc2, 2)

    # Filter the matches on a threshold
    matched1 = []
    matched2 = []
    nn_match_ratio = 0.8
    for m, n in nn_matches:
        if m.distance < nn_match_ratio * n.distance:
            matched1.append(kpts1[m.queryIdx])
            matched2.append(kpts2[m.trainIdx])

    # Find the homography based on the corresponding points
    ransac_thresh = 2.5
    src_pts = np.float32([m.pt for m in matched2]).reshape(-1,1,2)
    dst_pts = np.float32([m.pt for m in matched1]).reshape(-1,1,2)
    if len(matched1) >= 4:
        homo_cv, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
    else:
        print("not computed")
        homo_cv = np.array([[1,0,0], [0,1,0], [0,0,1]])
    return homo_cv


class VideoDataset(Dataset):
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
        super(VideoDataset, self).__init__()
        self.data_folder = data_folder
        self.n_classes = DATASET.n_classes
        self.ignore_index = DATASET.ignore_index
        self.label_extension = DATASET.label_extension
        self.img_extension = DATASET.img_extension
        self.frame_folder = DATASET.frame_folder
        self.mask_folder = DATASET.mask_folder
        self.reg_folder = DATASET.reg_folder
        self.convert_labels = DATASET.convert_labels
        self.crop_size = data_cfg["crop_size"]
        self.adjacent_frames = data_cfg["adjacent_frames"]
        self.labeled_frames_per_vid = data_cfg.get("labeled_frames_per_vid", None)
        self.val_skip_frames = data_cfg.get("val_skip_frames", 1)
        self.min_vid_len = data_cfg.get("min_vid_len", 0)
        self.soft_labels = data_cfg.get("soft_labels", False)
        self.opencv_homos = data_cfg.get("opencv_homos", False)
        self.opencv_model_type = data_cfg.get("opencv_model_type", "sift")

        self.training = training

        self.samples = []
        self.videos = []
        self.video_names = []
        v_idx = 0
        for v_name in video_indices:
            frames = os.listdir(os.path.join(data_folder, v_name, self.frame_folder))
            frames.sort()
            
            min_vid_len = max(self.min_vid_len, max(0, max(self.adjacent_frames)) - min(0, min(self.adjacent_frames))) if len(self.adjacent_frames)>0 else self.min_vid_len
            # Check if length of video is sufficient
            if len(frames) < min_vid_len + 1:
                continue
            
            frames_iter = frames[:-max(0, max(self.adjacent_frames))] if max(0, max(self.adjacent_frames)) > 0 else frames
            for (i, f) in enumerate(frames_iter, -min(0, min(self.adjacent_frames))):
                if self.islabeled(f, v_name, i, len(frames_iter)):
                    adj_frames = [frames[i-(-min(0, min(self.adjacent_frames)))+k] for k in self.adjacent_frames]
                    sample = (f, adj_frames, (v_idx, i))
                    self.samples.append(sample)

            self.videos.append(frames)
            self.video_names.append(v_name)
            v_idx += 1

        self.joint_transforms = joint_transforms
        self.img_transforms = img_transforms
        self.segmentation_transforms = segmentation_transforms

    def __getitem__(self, index):
        sample = self.samples[index]
        (f, adj_frames, (v_idx, i)) = sample
        label_name = self.name_to_labelname(f)

        # Read images
        v_name = self.video_names[v_idx]
        frame = self.read_image(f, v_name, self.data_folder)
        adj_frames = [self.read_image(adj_f, v_name, self.data_folder) for adj_f in adj_frames]
        label = self.read_mask(label_name, v_name, self.data_folder)

        # Data augmentation and transforms
        if self.joint_transforms is not None:
            frame, adj_frames, _, label = self.joint_transforms(frame, adj_frames, None, label)

        # Precomputed homo
        if self.opencv_homos:
            homo = compute_homo_cv(adj_frames[0], frame, self.opencv_model_type)
            homo = torch.tensor(homo, dtype=torch.float32).unsqueeze(0)
        else:
            homo = torch.tensor([0])

        # Tensor conversion
        if self.img_transforms is not None:
            frame, adj_frames, _ = self.img_transforms(frame, adj_frames, None)
        if self.segmentation_transforms is not None:
            label = self.segmentation_transforms(label)
        
        return frame, adj_frames, label, homo

    def __len__(self):
        return len(self.samples)
    
    def islabeled(self, f, v_name, idx_in_vid, len_vid):
        if not self.training:
            labelname = self.name_to_labelname(f)
            if idx_in_vid%self.val_skip_frames == 0:
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
        return l
    

class VideoLogitsDataset(Dataset):
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
        super(VideoLogitsDataset, self).__init__()
        self.data_folder = data_folder
        self.logit_folder = data_cfg["logits_folder"]
        self.n_classes = DATASET.n_classes
        self.ignore_index = DATASET.ignore_index
        self.label_extension = ".npz"
        self.img_extension = DATASET.img_extension
        self.frame_folder = DATASET.frame_folder
        self.mask_folder = ""
        self.reg_folder = DATASET.reg_folder
        self.convert_labels = DATASET.convert_labels
        self.crop_size = data_cfg["crop_size"]
        self.adjacent_frames = data_cfg["adjacent_frames"]
        self.labeled_frames_per_vid = data_cfg.get("labeled_frames_per_vid", None)
        self.val_skip_frames = data_cfg.get("val_skip_frames", 1)
        self.train_skip_frames = data_cfg.get("train_skip_frames", None)
        self.min_vid_len = data_cfg.get("min_vid_len", 0)
        self.opencv_homos = data_cfg.get("opencv_homos", False)
        self.opencv_model_type = data_cfg.get("opencv_model_type", "sift")

        self.training = training

        self.samples = []
        self.videos = []
        self.video_names = []
        v_idx = 0
        for v_name in video_indices:
            frames = os.listdir(os.path.join(data_folder, v_name, self.frame_folder))
            frames.sort()
            
            min_vid_len = max(self.min_vid_len, max(0, max(self.adjacent_frames)) - min(0, min(self.adjacent_frames))) if len(self.adjacent_frames)>0 else self.min_vid_len
            # Check if length of video is sufficient
            if len(frames) < min_vid_len + 1:
                continue
            
            frames_iter = frames[:-max(0, max(self.adjacent_frames))] if max(0, max(self.adjacent_frames)) > 0 else frames
            for (i, f) in enumerate(frames_iter, -min(0, min(self.adjacent_frames))):
                f_idx = i-(-min(0, min(self.adjacent_frames)))
                if self.islabeled(f, v_name, f_idx, len(frames_iter)):
                    adj_frames = [frames[f_idx+k] for k in self.adjacent_frames]
                    sample = (f, adj_frames, (v_idx, f_idx))
                    self.samples.append(sample)
            self.videos.append(frames)
            self.video_names.append(v_name)
            v_idx += 1

        self.joint_transforms = joint_transforms
        self.img_transforms = img_transforms
        self.segmentation_transforms = segmentation_transforms

    def __getitem__(self, index):
        sample = self.samples[index]
        (f, adj_fs, (v_idx, i)) = sample
        label_name = self.name_to_labelname(f)

        # Read images
        v_name = self.video_names[v_idx]
        frame = self.read_image(f, v_name, self.data_folder)
        adj_frames = [self.read_image(adj_f, v_name, self.data_folder) for adj_f in adj_fs]
        label = self.read_mask(label_name, v_name, self.logit_folder)
        adj_labels = [self.read_mask(self.name_to_labelname(adj_f), v_name, self.logit_folder) for adj_f in adj_fs]

        # Data augmentation and transforms
        if self.joint_transforms is not None:
            frame, adj_frames, _, label, adj_labels = self.joint_transforms(frame, adj_frames, None, label, adj_labels)

        # Precomputed homo
        if self.opencv_homos:
            homo = compute_homo_cv(adj_frames[0], frame, self.opencv_model_type)
            homo = torch.tensor(homo, dtype=torch.float32).unsqueeze(0)
        else:
            homo = torch.tensor([0])

        # Tensor conversion
        if self.img_transforms is not None:
            frame, adj_frames, _ = self.img_transforms(frame, adj_frames, None)
        if self.segmentation_transforms is not None:
            label = self.segmentation_transforms(label)
            adj_labels = [self.segmentation_transforms(adj_l) for adj_l in adj_labels]
        
        return frame, adj_frames, label, adj_labels, homo

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
            labels_idx = np.arange(n_labeled_frames*(len_vid//n_labeled_frames), step=len_vid//n_labeled_frames)
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

