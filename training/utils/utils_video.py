import torch
from tqdm import tqdm
import os
import torch.nn as nn
import yaml
import datetime
import numpy as np


def save_model(save_dict, cfg, save_dir, date, per_classes_mIoU=None):
    date_str = date.strftime("%d-%m_%H-%M")
    ep = save_dict["epoch"]
    train_loss = save_dict["train_losses"][-1]
    train_const_loss = save_dict["train_const_losses"][-1]
    val_loss = save_dict["val_losses"][-1]
    val_const_loss = save_dict["val_const_losses"][-1]
    val_global_miou = save_dict["val_global_miou"][-1]
    save_name = f"{date_str}"
    if not os.path.exists(os.path.join(save_dir, save_name)):
        os.mkdir(os.path.join(save_dir, save_name))
    with open(os.path.join(save_dir, save_name, save_name + "_config.yaml"), "w") as file:
        yaml.dump(cfg, file, default_flow_style=False, sort_keys=False)

    ep_date = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    with open(os.path.join(save_dir, save_name, save_name + "_logs.txt"), "a") as f:
        f.write(ep_date)
        f.write(" -- ")
        f.write(f"Epoch {ep}")
        f.write(" -- ")
        f.write(f"train loss = {train_loss:.4f}, train consist loss = {train_const_loss:.4f}, validation loss = {val_loss:.4f}, val consist loss = {val_const_loss:.4f}, val mIoU = {val_global_miou:.4f}")
        f.write("\n")
        if per_classes_mIoU is not None:
            f.write(f"Per class mIoU: {per_classes_mIoU}\n")

    torch.save(save_dict, os.path.join(save_dir, save_name, save_name + ".pth.tar"))
    return save_name