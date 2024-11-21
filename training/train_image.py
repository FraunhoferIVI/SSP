import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import os
    import numpy as np
    import torch
    import yaml
    from torch.utils.data import DataLoader
    import datetime

    from data.dataset_prep import prep_image_dataset
    from utils.optim_utils import get_criterion, get_optimizer_scheduler
    from models.image.models import get_model
    from training.utils.utils_image import train_one_epoch, train_with_metrics, evaluate, evaluate_with_metrics, save_model

import argparse
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Training Parameters")
    parser.add_argument("config", metavar="C", type=str, help="Name of config file")
    return parser.parse_args()

def main(config):
    date = datetime.datetime.now()
    # Config file
    with open("config/image/" + config, 'r') as cfg_file:
        cfg = yaml.load(cfg_file, Loader=yaml.FullLoader)

    save_dir = cfg["save_dir"]
    resume_training = cfg["resume_training"] is not None
    if resume_training:
        checkpoint_name = cfg["resume_training"]
        config = os.path.join(save_dir, checkpoint_name, checkpoint_name.split("@")[-1] + "_config.yaml")
        with open(config, 'r') as cfg_file:
            cfg = yaml.load(cfg_file, Loader=yaml.FullLoader)
        date = datetime.datetime.strptime(checkpoint_name.split("@")[-1], "%d-%m_%H-%M") 
        resume_checkpoint = torch.load(os.path.join(save_dir, checkpoint_name, checkpoint_name.split("@")[-1] + ".pth.tar"), map_location="cpu")

    data_cfg = cfg["data_cfg"]
    model_cfg = cfg["model_cfg"]
    training_cfg = cfg["training_cfg"]
    optim_cfg = cfg["optim_cfg"]
    loss_cfg = cfg["loss_cfg"]

    # Dataset
    train_dataset, val_dataset, DATASET = prep_image_dataset(data_cfg)

    # Dataloader
    batch_size = training_cfg["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=training_cfg["num_workers"], persistent_workers=True, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=training_cfg["num_workers"], persistent_workers=True, drop_last=True)

    # Model
    device = torch.device("cuda")
    model = get_model(
        model_cfg,
        DATASET.n_classes,
        )
    model.to(device)

    # Load pre-trained checkpoint
    if cfg["pretrained_checkpoint"] is not None:
        checkpoint = torch.load(os.path.join(cfg["save_dir"], cfg["pretrained_checkpoint"], "best_model_" + cfg["pretrained_checkpoint"].split("@")[-1] + ".pth.tar"), map_location="cpu")
        state_dict = checkpoint["model"]
        state_dict = {k: v for (k, v) in state_dict.items() if "classifier" not in k}
        model.load_state_dict(state_dict, strict=False)

    # Optim
    iter_per_epoch = len(train_loader)
    criterion = get_criterion(loss_cfg, DATASET.ignore_index, device=device, soft_labels=data_cfg.get("soft_labels", False))
    optimizer, scheduler = get_optimizer_scheduler(optim_cfg, model.parameters(), iter_per_epoch, lr=training_cfg["lr"], num_epochs=training_cfg["num_epochs"])
    print(f"{iter_per_epoch} iterations per epochs of {batch_size} batches each")

    start_epoch = 0
    train_losses = []
    train_global_miou = []
    train_classes_mIoU = []
    val_losses = []
    val_global_miou = []
    val_miou1 = []
    val_miou2 = []
    val_classes_mIoU = []

    if cfg.get("no_training", False):
        save_dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": 0,
            "train_losses": [torch.tensor(0)],
            "train_global_miou": [torch.tensor(0)],
            "train_classes_mIoU": [[torch.tensor(0) for _ in range(DATASET.n_classes)]],
            "val_losses": [torch.tensor(0)],
            "val_global_miou": [torch.tensor(0)],
            "val_miou1": [torch.tensor(0)],
            "val_miou2": [torch.tensor(0)],
            "val_classes_mIoU": [[torch.tensor(0) for _ in range(DATASET.n_classes)]]
        }
        save_name = save_model(save_dict, cfg, cfg["save_dir"], date, DATASET)
        return save_name


    if resume_training:
        model.load_state_dict(resume_checkpoint["model"])
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        scheduler.load_state_dict(resume_checkpoint["scheduler"])
        start_epoch = resume_checkpoint["epoch"]

        train_losses = resume_checkpoint["train_losses"]
        train_global_miou = resume_checkpoint["train_global_miou"]
        train_classes_mIoU = resume_checkpoint["train_classes_mIoU"]
        val_losses = resume_checkpoint["val_losses"]
        val_global_miou = resume_checkpoint["val_global_miou"]
        val_miou1 = resume_checkpoint["val_miou1"]
        val_miou2 = resume_checkpoint["val_miou2"]
        val_classes_mIoU = resume_checkpoint["val_classes_mIoU"]

    epochs = training_cfg["early_stopping"] if training_cfg["early_stopping"] is not None else training_cfg["num_epochs"]
    for epoch in range(start_epoch, epochs):
        train_loss = train_one_epoch(
            train_loader, 
            model,
            optimizer, 
            criterion, 
            scheduler, 
            device,
            )
        train_global_miou_ep = torch.tensor(0)
        train_classes_mIoU_ep = [torch.tensor(0) for _ in range(DATASET.n_classes)]
        # Compute metrics during training:
        #train_loss, train_global_miou_ep, train_classes_mIoU_ep = train_with_metrics(
        #    train_loader, 
        #    model,
        #    optimizer, 
        #    criterion, 
        #    scheduler, 
        #    device,
        #    DATASET.n_classes,
        #    DATASET.ignore_index
        #    )
        train_losses.append(train_loss)
        train_global_miou.append(train_global_miou_ep)
        train_classes_mIoU.append(train_classes_mIoU_ep)

        results = evaluate_with_metrics(
            val_loader, 
            model, 
            criterion, 
            device,
            n_classes=DATASET.n_classes,
            ignore_index=DATASET.ignore_index,
            )
        val_losses.append(results['val_loss'])
        val_global_miou.append(results["global_miou"])
        val_miou1.append(results["mIoU1"])
        val_miou2.append(results["mIoU2"])
        val_classes_mIoU.append(results["classes_mIoU"])
        per_classes_mIoU = {v: results["classes_mIoU"][k-1].item() for (k,v) in DATASET.classes.items() if k>0} if DATASET.ignore_index > 0 else {v: results["classes_mIoU"][k].item() for (k,v) in DATASET.classes.items()}
        print(f"Epoch {epoch+1}: train loss = {train_loss:.4f}, validation loss = {val_losses[-1]:.4f}, val mIoU = {val_global_miou[-1]:.4f}")
        print("-"*70)
        
        save_dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch+1,
            "train_losses": train_losses,
            "train_global_miou": train_global_miou,
            "train_classes_mIoU": train_classes_mIoU,
            "val_losses": val_losses,
            "val_global_miou": val_global_miou,
            "val_miou1": val_miou1,
            "val_miou2": val_miou2,
            "val_classes_mIoU": val_classes_mIoU
        }
        save_name = save_model(save_dict, cfg, cfg["save_dir"], date, DATASET)

    return save_name



if __name__=="__main__":
    args = parse_args()
    config = args.config
    save_name = main(config)
    print(save_name)