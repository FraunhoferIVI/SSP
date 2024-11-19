import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import os
    import PIL.Image as Image
    import numpy as np
    from tqdm import tqdm
    import torch
    import torch.nn as nn
    import yaml

    from vis_utils.visualization import color_predictions, inverse_normalize, pred_to_mask
    from utils.metrics import pixel_accuracy, meanIoU, weightedIoU, class_meanIoU, video_consistency, temporal_consistency, temporal_consistency_forwardbackward, get_flow_model, MetricMeter, PerClassMetricMeter
    from models.image.models import get_model as get_image_model
    from models.opt_flow import get_flow_model
    from models.video.models_consistency import get_model as get_video_model
    from data.dataset_prep import prep_infer_image_dataset


def main(config, checkpoint_name, checkpoint_folder, split, evaluation, best_model, write_res):
    device = torch.device("cuda")
    with open(config, 'r') as cfg_file:
        cfg = yaml.load(cfg_file, Loader=yaml.FullLoader)
    data_cfg = cfg["data_cfg"]
    image_model_cfg = cfg["image_model_cfg"]
    video_model_cfg = cfg["video_model_cfg"]

    save_dir = cfg["save_dir"]
    if os.path.exists(os.path.join(save_dir, checkpoint_folder + checkpoint_name, checkpoint_name.split("@")[-1] + ".pth.tar")):
        if best_model:
            checkpoint = torch.load(os.path.join(save_dir, checkpoint_folder + checkpoint_name, "best_model_3_" + checkpoint_name.split("@")[-1] + ".pth.tar"), map_location="cpu")
            print("Loaded best checkpoint at epoch {}".format(checkpoint["epoch"]))
        else:
            checkpoint = torch.load(os.path.join(save_dir, checkpoint_folder + checkpoint_name, checkpoint_name.split("@")[-1] + ".pth.tar"), map_location="cpu")
            print("Loaded last checkpoint")
    else:
        checkpoint = None

    vis_dir = os.path.join(cfg["results_dir"], checkpoint_name)
    save_folder = os.path.join(vis_dir, split)
    save_folder_colored = os.path.join(vis_dir, split + "_colored")
    save_folder_blended = os.path.join(vis_dir, split + "_blended")
    save_folder_labels = os.path.join(vis_dir, split + "_labels")
    save_folder_labels_blended = os.path.join(vis_dir, split + "_labels_blended")
    save_folder_gif = os.path.join(vis_dir, split + "_gif")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if write_res:
        if not os.path.exists(save_folder_colored):
            os.makedirs(save_folder_colored)
        if not os.path.exists(save_folder_blended):
            os.makedirs(save_folder_blended)
        if not os.path.exists(save_folder_labels):
            os.makedirs(save_folder_labels)
        if not os.path.exists(save_folder_labels_blended):
            os.makedirs(save_folder_labels_blended)
        if not os.path.exists(save_folder_gif):
            os.makedirs(save_folder_gif)

    # Dataset 
    video_dataset, DATASET = prep_infer_image_dataset(data_cfg, split=split)

    # Trained image model
    image_save_dir = image_model_cfg["image_save_dir"]
    img_checkpoint_folder = image_model_cfg["checkpoint_folder"]
    img_checkpoint_name = image_model_cfg["checkpoint_name"]
    img_best_model = image_model_cfg["best_model"]
    with open(os.path.join(image_save_dir, img_checkpoint_folder + img_checkpoint_name, img_checkpoint_name.split("@")[-1] + "_config.yaml"), 'r') as cfg_file:
        image_cfg = yaml.load(cfg_file, Loader=yaml.FullLoader)
    seg_model = get_image_model(image_cfg["model_cfg"], DATASET.n_classes)
    seg_model.to(device)
    if img_best_model:
        img_checkpoint = torch.load(os.path.join(image_cfg["save_dir"], img_checkpoint_folder + img_checkpoint_name, "best_model_" + img_checkpoint_name.split("@")[-1] + ".pth.tar"), map_location="cpu")
    else:
        img_checkpoint = torch.load(os.path.join(image_cfg["save_dir"], img_checkpoint_folder + img_checkpoint_name, img_checkpoint_name.split("@")[-1] + ".pth.tar"), map_location="cpu")
    seg_model.load_state_dict(img_checkpoint["model"])

    # Optical flow model
    flow_model = get_flow_model()
    flow_model.to(device)

    # Video model
    model = get_video_model(video_model_cfg, seg_model, flow_model, DATASET.n_classes)
    model.to(device)
    print(f"Model has {sum([p.numel() for p in model.parameters()]):,} parameters")
    if checkpoint is not None:
        state_dict = checkpoint["model"]
        model.load_state_dict(state_dict)

    # Init evaluation metrics
    if evaluation:
        # Flow prediction model for TC
        model_raft = get_flow_model()
        model_raft = model_raft.to(device)

        mIoU1 = MetricMeter()
        mIoU2 = MetricMeter()
        wIoU = MetricMeter()
        accuracy = MetricMeter()
        tc = MetricMeter()
        tc_fb = MetricMeter()
        n_frames_vc = DATASET.n_frames_vc
        vc = []
        classes_mIoU = PerClassMetricMeter(DATASET.n_classes)
        preds_for_iou = []
        labels_for_iou = []

    # Predictions + evaluation
    video_iter = tqdm(video_dataset)
    for (frames_names, frames, labels_names, labels, v_name, homos) in video_iter:
        frame_list = []
        label_list = []
        pred_list = []
        
        # Predictions
        preds = model.infer_video(frames, homos, device)
        frame_list = [f.numpy() for f in frames]
        label_list = [l.numpy() for l in labels]
        pred_list = [p.numpy() for p in preds]

        # Iter over video frames
        frames_iter = frame_list[:-data_cfg["min_vid_len"]] if data_cfg["min_vid_len"] > 0 else frame_list
        for (i, frame) in enumerate(frames_iter):
            frame_name = frames_names[i]
            label_name = video_dataset.name_to_labelname(frame_name)
            labeled = label_name in labels_names
            if labeled:
                label = labels[labels_names.index(label_name)]
            pred = preds[i]

            # Compute metrics
            if evaluation and labeled:
                mIoU1.update(meanIoU(pred, label, DATASET.n_classes, ignore_index=DATASET.ignore_index))
                valid_classes = torch.unique(label[label<255])
                valid_classes = np.array(torch.nn.functional.one_hot(valid_classes, num_classes=DATASET.n_classes).sum(0))
                classes_mIoU.update(np.array(class_meanIoU(pred, label, DATASET.n_classes, ignore_index=DATASET.ignore_index)), valid_classes)
                mIoU2.update(classes_mIoU.last_values.sum()/valid_classes.astype(float).sum())
                wIoU.update(weightedIoU(pred, label, DATASET.n_classes, ignore_index=DATASET.ignore_index))
                accuracy.update(pixel_accuracy(pred, label))
                preds_for_iou.append(pred)
                labels_for_iou.append(label)

            # Save predictions (normal and colored)
            if write_res:
                pred_pil = Image.fromarray(pred_to_mask(pred.numpy(), ignore_index=DATASET.ignore_index).astype(np.uint8))
                pred_pil_colored = Image.fromarray(color_predictions(pred.numpy(), colors=DATASET.colors, ignore_index=DATASET.ignore_index).astype(np.uint8))
                pred_pil_blended = color_predictions(pred.numpy(), colors=DATASET.colors, ignore_index=DATASET.ignore_index, blend_img=inverse_normalize(frame))[1]
                if not os.path.exists(os.path.join(save_folder, v_name)):
                    os.makedirs(os.path.join(save_folder, v_name))
                if not os.path.exists(os.path.join(save_folder, v_name, label_name)):
                    pred_pil.save(os.path.join(save_folder, v_name, label_name))
                if not os.path.exists(os.path.join(save_folder_colored, v_name)):
                    os.makedirs(os.path.join(save_folder_colored, v_name))
                if not os.path.exists(os.path.join(save_folder_colored, v_name, label_name)):
                    pred_pil_colored.save(os.path.join(save_folder_colored, v_name, label_name))
                if not os.path.exists(os.path.join(save_folder_blended, v_name)):
                    os.makedirs(os.path.join(save_folder_blended, v_name))
                if not os.path.exists(os.path.join(save_folder_blended, v_name, label_name)):
                    pred_pil_blended.save(os.path.join(save_folder_blended, v_name, label_name))

                if labeled:
                    label_pil = color_predictions(label.numpy(), colors=DATASET.colors, ignore_index=DATASET.ignore_index, blend_img=inverse_normalize(frame))[0]
                    label_pil_blended = color_predictions(label.numpy(), colors=DATASET.colors, ignore_index=DATASET.ignore_index, blend_img=inverse_normalize(frame))[1]
                    #if not os.path.exists(os.path.join(save_folder_labels, v_name)):
                    #    os.makedirs(os.path.join(save_folder_labels, v_name))
                    #if not os.path.exists(os.path.join(save_folder_labels, v_name, label_name)):
                    #    label_pil.save(os.path.join(save_folder_labels, v_name, label_name))
                    #if not os.path.exists(os.path.join(save_folder_labels_blended, v_name)):
                    #    os.makedirs(os.path.join(save_folder_labels_blended, v_name))
                    #if not os.path.exists(os.path.join(save_folder_labels_blended, v_name, label_name)):
                    #    label_pil_blended.save(os.path.join(save_folder_labels_blended, v_name, label_name))

        # Save gifs
        if write_res:
            if not os.path.exists(os.path.join(save_folder_gif, v_name)):
                os.makedirs(os.path.join(save_folder_gif, v_name))
            frame_gif_list = [Image.fromarray(inverse_normalize(frame)) for frame in frame_list]
            pred_gif_list = [color_predictions(pred, colors=DATASET.colors, ignore_index=DATASET.ignore_index, blend_img=inverse_normalize(frame_list[n]))[0] for (n,pred) in enumerate(pred_list)]
            pred_gif_list_blend = [color_predictions(pred, colors=DATASET.colors, ignore_index=DATASET.ignore_index, blend_img=inverse_normalize(frame_list[n]))[1] for (n,pred) in enumerate(pred_list)]
            #if not os.path.exists(os.path.join(save_folder_gif, v_name, "frames.gif")):
            #    frame_gif_list[0].save(os.path.join(save_folder_gif, v_name, "frames.gif"), save_all=True, append_images=frame_gif_list[1:], duration=(1000/DATASET.fps), loop=0)
            if not os.path.exists(os.path.join(save_folder_gif, v_name, "preds.gif")):
                pred_gif_list[0].save(os.path.join(save_folder_gif, v_name, "preds.gif"), save_all=True, append_images=pred_gif_list[1:], duration=(1000/DATASET.fps), loop=0)
            if not os.path.exists(os.path.join(save_folder_gif, v_name, "preds_blended.gif")):
                pred_gif_list_blend[0].save(os.path.join(save_folder_gif, v_name, "preds_blended.gif"), save_all=True, append_images=pred_gif_list_blend[1:], duration=(1000/DATASET.fps), loop=0)

            if len(label_list)>1:
                label_gif_list = [color_predictions(label, colors=DATASET.colors, ignore_index=DATASET.ignore_index, blend_img=inverse_normalize(frame_list[n]))[0] for (n,label) in enumerate(label_list)]
                label_gif_list_blend = [color_predictions(label, colors=DATASET.colors, ignore_index=DATASET.ignore_index, blend_img=inverse_normalize(frame_list[n]))[1] for (n,label) in enumerate(label_list)]
                if not os.path.exists(os.path.join(save_folder_gif, v_name, "labels.gif")):
                    label_gif_list[0].save(os.path.join(save_folder_gif, v_name, "labels.gif"), save_all=True, append_images=label_gif_list[1:], duration=(1000/DATASET.fps), loop=0)
                if not os.path.exists(os.path.join(save_folder_gif, v_name, "labels_blended.gif")):
                    label_gif_list_blend[0].save(os.path.join(save_folder_gif, v_name, "labels_blended.gif"), save_all=True, append_images=label_gif_list_blend[1:], duration=(1000/DATASET.fps), loop=0)

        if evaluation:
            if len(frame_list) <= 1:
                continue
            tc.update(temporal_consistency(frame_list, pred_list, model_raft, DATASET.n_classes, device))
            #tc_fb.update(temporal_consistency_forwardbackward(frame_list, pred_list, model_raft, DATASET.n_classes, device))
            if len(label_list) > n_frames_vc and len(label_list) == len(pred_list):
                vc.extend(video_consistency(label_list, pred_list, n_frames_vc))
            video_iter.set_description(f"{v_name}: TC = {tc.avg:.4f}")
    
    if evaluation:
        vc = np.array(vc)
        vc = np.nanmean(vc)
        per_classes_mIoU = {v: classes_mIoU.avg[k-1] for (k,v) in DATASET.classes.items() if k>0} if DATASET.ignore_index > 0 else {v: classes_mIoU.avg[k].item() for (k,v) in DATASET.classes.items()}
        preds_for_iou = torch.stack(preds_for_iou, dim=0)
        labels_for_iou = torch.stack(labels_for_iou, dim=0)
        global_miou = meanIoU(preds_for_iou, labels_for_iou, DATASET.n_classes, ignore_index=DATASET.ignore_index)
        global_classes_iou = class_meanIoU(preds_for_iou, labels_for_iou, DATASET.n_classes, ignore_index=DATASET.ignore_index)
        global_per_classes_mIoU = {v: global_classes_iou[k-1].item() for (k,v) in DATASET.classes.items() if k>0} if DATASET.ignore_index > 0 else {v: global_classes_iou[k].item() for (k,v) in DATASET.classes.items()}
        print(f"mIoU = {global_miou:.4f} | Temporal Consistency = {tc.avg:.4f}")
        #print(per_classes_mIoU)
        print(global_per_classes_mIoU)
        metrics_name = f"log_metrics_{split}_best_model_3.txt" if best_model else f"log_metrics_{split}.txt"
        with open(os.path.join(vis_dir, metrics_name), "a") as f:
            if checkpoint is not None:
                f.write("Checkpoint {} at epoch {}\n".format(checkpoint_name, checkpoint["epoch"]))
            #f.write(f"Mean IoU 1 = {mIoU1.avg:.4f}\n")
            #f.write(f"Mean IoU 2 = {mIoU2.avg:.4f}\n")
            f.write(f"mIoU = {global_miou:.4f}\n")
            #f.write(f"Weighted = {wIoU.avg:.4f}\n")
            f.write(f"Temporal Consistency = {tc.avg:.4f}\n")
            #f.write(f"Temporal Consistency FB = {tc_fb.avg:.4f}\n")
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
    parser = argparse.ArgumentParser(description="Visualization Parameters")
    parser.add_argument("checkpoint_name", metavar="C", type=str, help="Checkpoint to visualize")
    parser.add_argument("--save-dir", required=False, type=str, default="checkpoints",
                         help="Folder where checkpoint (and its config) is located. Should be in config file")
    parser.add_argument("--checkpoint-folder", required=False, type=str, default="", help="Subfolder of checkpoint")
    parser.add_argument("--split", required=False, type=str, default="val", help="Data split to visualize")
    parser.add_argument('--evaluation', dest='evaluation', action='store_true', help='Compute metrics (default)')
    parser.add_argument('--no-evaluation', dest='evaluation', action='store_false', help='Don\'t compute metrics')
    parser.add_argument('--best-model', dest='best_model', action='store_true', help='Use best checkpoint')
    parser.add_argument('--no-best-model', dest='best_model', action='store_false', help='Use last checkpoint (default)')
    parser.add_argument('--write-res', dest='write_res', action='store_true', help='Write results to disk (default)')
    parser.add_argument('--no-write-res', dest='write_res', action='store_false', help='Do not write results to disk')
    parser.set_defaults(best_model=False)
    parser.set_defaults(evaluation=True)
    parser.set_defaults(write_res=True)
    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()
    save_dir = args.save_dir
    checkpoint_name = args.checkpoint_name
    checkpoint_folder = args.checkpoint_folder
    config = os.path.join(save_dir, checkpoint_folder + checkpoint_name, checkpoint_name.split("@")[-1] + "_config.yaml")
    split = args.split
    evaluation = args.evaluation
    best_model = args.best_model
    write_res = args.write_res
    main(config, checkpoint_name, checkpoint_folder, split, evaluation, best_model, write_res)
