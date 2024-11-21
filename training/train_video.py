import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import PIL.Image as Image
    import os
    import numpy as np
    from tqdm import tqdm
    import datetime
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    import time
    import yaml

    from data.dataset_prep import prep_video_dataset
    from utils.optim_utils import get_criterion, get_optimizer_scheduler
    from training.utils.utils_video import save_model

    from models.image.models import get_model as get_image_model
    from models.opt_flow import get_flow_model
    from models.video.models_consistency import get_model as get_video_model

import argparse
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Training Parameters")
    parser.add_argument("config", metavar="C", type=str, help="Name of config file")
    return parser.parse_args()

def main(config):
    date = datetime.datetime.now()
    device = torch.device("cuda")
    # Config file
    with open("config/video/" + config, 'r') as cfg_file:
        cfg = yaml.load(cfg_file, Loader=yaml.FullLoader)
    
    resume_training = cfg["resume_training"] is not None
    if resume_training:
        checkpoint_name = cfg["resume_training"]
        config = os.path.join(cfg["save_dir"], checkpoint_name, checkpoint_name.split("@")[-1] + "_config.yaml")
        with open(config, 'r') as cfg_file:
            cfg = yaml.load(cfg_file, Loader=yaml.FullLoader)
        date = datetime.datetime.strptime(checkpoint_name.split("@")[-1], "%d-%m_%H-%M") 
        resume_checkpoint = torch.load(os.path.join(cfg["save_dir"], checkpoint_name, checkpoint_name.split("@")[-1] + ".pth.tar"), map_location="cpu")
        print(date)

    data_cfg = cfg["data_cfg"]
    image_model_cfg = cfg["image_model_cfg"]
    video_model_cfg = cfg["video_model_cfg"]
    training_cfg = cfg["training_cfg"]
    optim_cfg = cfg["optim_cfg"]
    loss_cfg = cfg["loss_cfg"]

    # Dataset
    train_dataset, val_dataset, DATASET = prep_video_dataset(data_cfg)

    # Dataloader
    batch_size = training_cfg["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=training_cfg["num_workers"], persistent_workers=True, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=training_cfg["num_workers"], persistent_workers=True, drop_last=True)

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

    # Video model
    flow_model = get_flow_model()
    flow_model.to(device)

    model = get_video_model(video_model_cfg, seg_model, flow_model, DATASET.n_classes)
    model.to(device)
    print(f"Model has {sum([p.numel() for p in model.parameters()]):,} parameters")

    # Load pre-trained checkpoint
    if cfg["pretrained_checkpoint"] is not None:
        try:
            pretrained_checkpoint = torch.load(os.path.join(cfg["save_dir"], cfg["pretrained_checkpoint"], 
                                                            cfg["pretrained_checkpoint"].split("@")[-1] + ".pth.tar"), map_location="cpu")
            state_dict = pretrained_checkpoint["model"]
        except:
            state_dict = torch.load(cfg["pretrained_checkpoint"])
        model.load_state_dict(state_dict, strict=False)

    # Optim
    iter_per_epoch = len(train_loader)
    criterion = get_criterion(loss_cfg, DATASET.ignore_index, device=device, soft_labels=data_cfg.get("soft_labels", False))
    if training_cfg["trained_layer"] is not None:
        param_groups = [{'params': (p for (n, p) in model.named_parameters() if (training_cfg["trained_layer"] not in n)), 'lr': training_cfg["secondary_lr"]}, 
                        {'params': (p for (n, p) in model.named_parameters() if (training_cfg["trained_layer"] in n)), 'lr': training_cfg["lr"]}]
    else:
        param_groups = model.parameters()
    
    optimizer, scheduler = get_optimizer_scheduler(optim_cfg, param_groups, iter_per_epoch, lr=training_cfg["lr"], num_epochs=training_cfg["num_epochs"])
    print(f"{iter_per_epoch} iterations per epochs of {batch_size} batches each")

    start_epoch = 0
    train_losses = []
    train_const_losses = []
    val_losses = []
    val_const_losses = []
    val_global_miou = []
    val_miou1 = []
    val_miou2 = []
    val_classes_mIoU = []

    if resume_training:
        start_epoch = resume_checkpoint["epoch"]
        model.load_state_dict(resume_checkpoint["model"])
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        scheduler.load_state_dict(resume_checkpoint["scheduler"])

        train_losses = resume_checkpoint["train_losses"]
        train_const_losses = resume_checkpoint["train_const_losses"]
        val_losses = resume_checkpoint["val_losses"]
        val_const_losses = resume_checkpoint["val_const_losses"]
        val_global_miou = resume_checkpoint["val_global_miou"]
        val_miou1 = resume_checkpoint["val_miou1"]
        val_miou2 = resume_checkpoint["val_miou2"]
        val_classes_mIoU = resume_checkpoint["val_classes_mIoU"]

    epochs = training_cfg["early_stopping"] if training_cfg["early_stopping"] is not None else training_cfg["num_epochs"]
    for epoch in range(start_epoch, epochs):
        if data_cfg.get("logit_distillation", False):
            train_loss, const_loss = model.kd_train_one_epoch(
            train_loader, 
            optimizer, 
            criterion, 
            scheduler, 
            device,
            )
        else:
            train_loss, const_loss = model.train_one_epoch(
                train_loader, 
                optimizer, 
                criterion, 
                scheduler, 
                device,
                )
        train_losses.append(train_loss)
        train_const_losses.append(const_loss)

        results = model.evaluate_with_metrics(
            val_loader, 
            criterion, 
            device,
            n_classes=DATASET.n_classes,
            ignore_index=DATASET.ignore_index,
            )
        val_losses.append(results['val_loss'])
        val_const_losses.append(results['const_loss'])
        val_global_miou.append(results["global_miou"])
        val_miou1.append(results["mIoU1"])
        val_miou2.append(results["mIoU2"])
        val_classes_mIoU.append(results["classes_mIoU"])
        per_classes_mIoU = {v: results["classes_mIoU"][k-1].item() for (k,v) in DATASET.classes.items() if k>0} if DATASET.ignore_index > 0 else {v: results["classes_mIoU"][k].item() for (k,v) in DATASET.classes.items()}
        print(f"Epoch {epoch+1}: train loss = {train_loss:.4f}, train consist loss = {const_loss:.4f}, validation loss = {val_losses[-1]:.4f}, val consist loss = {val_const_losses[-1]:.4f}, val mIoU = {val_global_miou[-1]:.4f}")
        print("-"*70)
        
        save_dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch+1,
            "train_losses": train_losses,
            "train_const_losses": train_const_losses,
            "val_losses": val_losses,
            "val_const_losses": val_const_losses,
            "val_global_miou": val_global_miou,
            "val_miou1": val_miou1,
            "val_miou2": val_miou2,
            "val_classes_mIoU": val_classes_mIoU
        }
        save_name = save_model(save_dict, cfg, cfg["save_dir"], date, per_classes_mIoU=per_classes_mIoU)

    return save_name


if __name__=="__main__":
    args = parse_args()
    config = args.config
    save_name = main(config)
    print(save_name)