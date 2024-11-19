import numpy as np
from data.image.image_dataset import ImageDataset, ImageInferenceDataset, ImageLogitsDataset
from data.video.video_dataset import VideoDataset, VideoLogitsDataset
from data.dataset_utils import parse_datasets, get_transforms


def prep_video_dataset(data_cfg):
    DATASET, data_folder, video_train_idx, video_val_idx = parse_datasets(data_cfg["dataset"], path=data_cfg["path"])

    augmentations, frame_transforms, frame_transforms_val, mask_transforms = get_transforms("ssltm", 
                                                                                            data_cfg["crop_size"], 
                                                                                            DATASET, 
                                                                                            data_augmentation=data_cfg.get("data_augmentation", False),
                                                                                            soft_labels=data_cfg.get("soft_labels", False),
                                                                                            square_crop=data_cfg.get("square_crop", False))

    traindatasetclass = VideoLogitsDataset if data_cfg.get("logit_distillation", False) else VideoDataset
    train_dataset = traindatasetclass(
        data_folder, 
        video_train_idx, 
        DATASET,
        data_cfg,
        training=True, 
        joint_transforms=augmentations,
        img_transforms=frame_transforms,
        segmentation_transforms=mask_transforms,
        )
    val_dataset = VideoDataset(
        data_folder, 
        video_val_idx, 
        DATASET,
        data_cfg, 
        training=False, 
        img_transforms=frame_transforms_val,
        segmentation_transforms=mask_transforms,
        )

    print(f"Train dataset: {len(train_dataset)} samples, Validation dataset: {len(val_dataset)} samples")
    return train_dataset, val_dataset, DATASET


def prep_image_dataset(data_cfg):
    DATASET, data_folder, video_train_idx, video_val_idx = parse_datasets(data_cfg["dataset"], path=data_cfg["path"])

    augmentations, frame_transforms, frame_transforms_val, mask_transforms = get_transforms("image", 
                                                                                            data_cfg["crop_size"], 
                                                                                            DATASET, 
                                                                                            data_augmentation=data_cfg.get("data_augmentation", False),
                                                                                            soft_labels=data_cfg.get("soft_labels", False),
                                                                                            square_crop=data_cfg.get("square_crop", False))

    traindatasetclass = ImageLogitsDataset if data_cfg.get("logit_distillation", False) else ImageDataset
    train_dataset = traindatasetclass(
        data_folder,
        video_train_idx,
        DATASET,
        data_cfg,
        training=True,
        joint_transforms=augmentations,
        img_transforms=frame_transforms,
        segmentation_transforms=mask_transforms,
    )

    val_dataset = ImageDataset(
        data_folder,
        video_val_idx,
        DATASET,
        data_cfg,
        training=False,
        joint_transforms=None,
        img_transforms=frame_transforms_val,
        segmentation_transforms=mask_transforms,
    )

    print(f"Train dataset contains {len(train_dataset)} samples")
    print(f"Validation dataset contains {len(val_dataset)} samples")

    return train_dataset, val_dataset, DATASET



def prep_infer_image_dataset(data_cfg, split="val", val_skip_frames=1):
    DATASET, data_folder, video_train_idx, video_val_idx = parse_datasets(data_cfg["dataset"], path=data_cfg["path"], split=split)
    video_indices = video_train_idx if split == "train" else video_val_idx
    _, _, frame_transforms, mask_transforms = get_transforms("image", data_cfg["crop_size"], DATASET, data_augmentation=False)
    video_dataset = ImageInferenceDataset(
        data_folder, 
        video_indices, 
        DATASET, 
        data_cfg,
        img_transforms=frame_transforms, 
        segmentation_transforms=mask_transforms,
        val_skip_frames=val_skip_frames,
        )

    print(f"Inference dataset contains {len(video_dataset)} videos")

    return video_dataset, DATASET


