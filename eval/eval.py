import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import os
    import PIL.Image as Image
    import numpy as np
    from tqdm import tqdm
    import torch
    import yaml

    from vis_utils.visualization import color_predictions, inverse_normalize, pred_to_mask, mask_to_pred
    from utils.metrics import (pixel_accuracy, meanIoU, weightedIoU, class_meanIoU, video_consistency, 
                            temporal_consistency, temporal_consistency_forwardbackward, temporal_consistency_nframes, temporal_consistency_kskip, 
                            get_flow_model, MetricMeter, PerClassMetricMeter)
    from data.dataset_prep import prep_infer_image_dataset


def main(config, checkpoint_name, split, checkpoint_folder):
    device = torch.device("cuda:0")
    with open(config, 'r') as cfg_file:
        cfg = yaml.load(cfg_file, Loader=yaml.FullLoader)
    data_cfg = cfg["data_cfg"]

    vis_dir = os.path.join(cfg["results_dir"], checkpoint_folder, checkpoint_name)
    save_folder = os.path.join(vis_dir, split)

    # Dataset prep (data_utils.py)
    video_dataset, DATASET = prep_infer_image_dataset(data_cfg)

    # Flow prediction model for TC
    model_raft = get_flow_model()
    model_raft = model_raft.to(device)

    # Metrics
    mIoU1 = MetricMeter()
    mIoU2 = MetricMeter()
    wIoU = MetricMeter()
    accuracy = MetricMeter()
    tc = MetricMeter()
    tc_fb = MetricMeter()
    tc_n = MetricMeter()
    tc_10 = MetricMeter()
    n_frames_vc = DATASET.n_frames_vc
    vc = []
    classes_mIoU = PerClassMetricMeter(DATASET.n_classes)
    preds_for_iou = []
    labels_for_iou = []
    
    # evaluation
    video_iter = tqdm(video_dataset)
    for (frames_names, frames, labels_names, labels, v_name, homos) in video_iter:
        frame_list = []
        label_list = []
        pred_list = []
        
        frames_iter = frames[:-data_cfg["min_vid_len"]] if data_cfg["min_vid_len"] > 0 else frames
        for (i, frame) in enumerate(frames_iter):
            frame_name = frames_names[i]
            label_name = video_dataset.name_to_labelname(frame_name)
            labeled = label_name in labels_names
            pred = mask_to_pred(torch.LongTensor(np.array(Image.open(os.path.join(save_folder, v_name, label_name)))), ignore_index=DATASET.ignore_index)
            frame_list.append(frame.numpy())
            pred_list.append(pred.numpy())
            # Compute metrics if label is available
            if labeled:
                label = labels[labels_names.index(label_name)]
                mIoU1.update(meanIoU(pred, label, DATASET.n_classes, ignore_index=DATASET.ignore_index))
                valid_classes = torch.unique(label[label<255])
                valid_classes = np.array(torch.nn.functional.one_hot(valid_classes, num_classes=DATASET.n_classes).sum(0))
                classes_mIoU.update(np.array(class_meanIoU(pred, label, DATASET.n_classes, ignore_index=DATASET.ignore_index)), valid_classes)
                mIoU2.update(classes_mIoU.last_values.sum()/valid_classes.astype(float).sum())
                wIoU.update(weightedIoU(pred, label, DATASET.n_classes, ignore_index=DATASET.ignore_index))
                accuracy.update(pixel_accuracy(pred, label))
                label_list.append(label.numpy())
                preds_for_iou.append(pred)
                labels_for_iou.append(label)
                
        if len(frame_list) > 1:
            tc.update(temporal_consistency(frame_list, pred_list, model_raft, DATASET.n_classes, device))
            #tc_n.update(temporal_consistency_nframes(frame_list, pred_list, model_raft, DATASET.n_classes, device, n_window=3))
            #tc_10.update(temporal_consistency_kskip(frame_list, pred_list, model_raft, DATASET.n_classes, device, k_skip=10))
            #tc_fb.update(temporal_consistency_forwardbackward(frame_list, pred_list, model_raft, DATASET.n_classes, device))
            if len(label_list) > n_frames_vc and len(label_list) == len(pred_list):
                vc.extend(video_consistency(label_list, pred_list, n_frames_vc))

        video_iter.set_description(f"{v_name}: TC = {tc.avg:.4f}")
        per_classes_mIoU = {v: classes_mIoU.avg[k-1] for (k,v) in DATASET.classes.items() if k>0}
        with open(os.path.join(vis_dir, f"log_metrics_pervideo_{split}.txt"), "a") as f:
            f.write(f"Metrics avg at video {v_name}: ")
            f.write(f"TC = {tc.avg:.4f} ||| ")
            for (k, v) in per_classes_mIoU.items():
                f.write(f" {k}: {v:.5f} | ")
            f.write("\n")
        
    vc = np.array(vc)
    vc = np.nanmean(vc)
    per_classes_mIoU = {v: classes_mIoU.avg[k-1] for (k,v) in DATASET.classes.items() if k>0} if DATASET.ignore_index > 0 else {v: classes_mIoU.avg[k].item() for (k,v) in DATASET.classes.items()}
    preds_for_iou = torch.stack(preds_for_iou, dim=0)
    labels_for_iou = torch.stack(labels_for_iou, dim=0)
    global_miou = meanIoU(preds_for_iou, labels_for_iou, DATASET.n_classes, ignore_index=DATASET.ignore_index)
    global_classes_iou = class_meanIoU(preds_for_iou, labels_for_iou, DATASET.n_classes, ignore_index=DATASET.ignore_index)
    global_per_classes_mIoU = {v: global_classes_iou[k-1].item() for (k,v) in DATASET.classes.items() if k>0} if DATASET.ignore_index > 0 else {v: global_classes_iou[k].item() for (k,v) in DATASET.classes.items()}
    print(f"mIoU = {global_miou:.4f} | Temporal Consistency = {tc.avg:.4f}")
    print(per_classes_mIoU)
    print(global_per_classes_mIoU)
    with open(os.path.join(vis_dir, f"log_metrics_{split}.txt"), "r") as f:
        first_line = f.readline().strip()
    with open(os.path.join(vis_dir, f"log_metrics_{split}.txt"), "a") as f:
        f.write(first_line + "\n")
        #f.write(f"Mean IoU 1 = {mIoU1.avg:.4f}\n")
        #f.write(f"Mean IoU 2 = {mIoU2.avg:.4f}\n")
        f.write(f"mIoU = {global_miou:.4f}\n")
        #f.write(f"Weighted = {wIoU.avg:.4f}\n")
        f.write(f"Temporal Consistency = {tc.avg:.4f}\n")
        #f.write(f"Video Consistency ({n_frames_vc}) = {vc:.4f}\n")
        #f.write(f"pixel accuracy = {accuracy.avg:.4f}\n")
        #f.write(f"Per class mIoU:\n")
        #for (k, v) in per_classes_mIoU.items():
        #    f.write(f"  {k}: {v:.5f}\n")
        f.write(f"per class mIoU:\n")
        for (k, v) in global_per_classes_mIoU.items():
            f.write(f"  {k}: {v:.5f}\n")


import argparse
def parse_args():
    """Parse command line arguments."""
    default_video_save_dir = "checkpoints"
    default_img_save_dir = "checkpoints"
    parser = argparse.ArgumentParser(description="Evaluation Parameters")
    parser.add_argument("checkpoint_name", metavar="C", type=str, help="Checkpoint to visualize")
    parser.add_argument("--save-dir", required=False, type=str, default=default_video_save_dir,
                         help="Folder where checkpoint (and its config) is located. Should be in config file. Default is video save--dir")
    parser.add_argument("--checkpoint-folder", required=False, type=str, default="", help="Subfolder of checkpoint")
    parser.add_argument("--split", required=False, type=str, default="val", help="Data split to visualize")
    parser.add_argument("--image", action="store_true", help="Use the stored image save-dir if no --save-dir is provided")
    args = parser.parse_args()

    if args.image and args.save_dir == default_video_save_dir:
        args.save_dir = default_img_save_dir
        
    return args

if __name__=='__main__':
    args = parse_args()
    save_dir = args.save_dir
    checkpoint_name = args.checkpoint_name
    checkpoint_folder = args.checkpoint_folder
    config = os.path.join(save_dir, checkpoint_folder, checkpoint_name, checkpoint_name.split("@")[-1] + "_config.yaml")
    split = args.split
    main(config, checkpoint_name, split, checkpoint_folder)
