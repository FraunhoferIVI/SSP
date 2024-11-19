import torch
from tqdm import tqdm
import os
import torch.nn as nn
import yaml
import datetime
import numpy as np
from utils.metrics import pixel_accuracy, meanIoU, weightedIoU, class_meanIoU, MetricMeter, PerClassMetricMeter
from data.utils.images_transforms import soft_to_hard_labels

def train_one_epoch(
        train_loader,
        model,
        optimizer,
        criterion,
        scheduler,
        device
):
    train_loss = 0
    train_iter = tqdm(train_loader)
    model.train()
    for (images, labels) in train_iter:
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.to(device)

        out = model(images)
        loss = criterion(out, labels)

        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()
        train_iter.set_description(desc=f"train loss = {loss.item():.4f}")

    train_loss = train_loss/len(train_loader)
    return train_loss


def train_with_metrics(
        train_loader,
        model,
        optimizer,
        criterion,
        scheduler,
        device,
        n_classes,
        ignore_index
):
    train_loss = 0
    train_iter = tqdm(train_loader)
    model.train()
    preds_for_iou = []
    labels_for_iou = []
    for (images, labels) in train_iter:
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.to(device)

        out = model(images)
        loss = criterion(out, labels)

        preds = out.argmax(1)
        preds = preds.detach().cpu()
        if labels.dim() != preds.dim():
            labels = soft_to_hard_labels(labels, ignore_index)
        labels = labels.detach().cpu()
        preds_for_iou.append(preds)
        labels_for_iou.append(labels)

        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()
        train_iter.set_description(desc=f"train loss = {loss.item():.4f}")

    train_loss = train_loss/len(train_loader)
    preds_for_iou = torch.cat(preds_for_iou, dim=0)
    labels_for_iou = torch.cat(labels_for_iou, dim=0)
    global_miou = meanIoU(preds_for_iou, labels_for_iou, n_classes, ignore_index=ignore_index)
    global_classes_iou = class_meanIoU(preds_for_iou, labels_for_iou, n_classes, ignore_index=ignore_index)
    return train_loss, global_miou, global_classes_iou


def evaluate(
        val_loader,
        model,
        criterion,
        device
):
    val_loss = 0
    val_iter = tqdm(val_loader)
    model.eval()
    for (images, labels) in val_iter:
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            out = model(images)
            loss = criterion(out, labels)

        val_loss += loss.item()
        val_iter.set_description(desc=f"val loss = {loss.item():.4f}")

    val_loss = val_loss/len(val_loader)
    return val_loss


def evaluate_with_metrics(
        val_loader,
        model,
        criterion,
        device,
        n_classes,
        ignore_index=255
):
    val_loss = 0
    val_iter = tqdm(val_loader)
    mIoU1 = MetricMeter()
    mIoU2 = MetricMeter()
    #classes_mIoU = PerClassMetricMeter(n_classes)
    preds_for_iou = []
    labels_for_iou = []
    model.eval()
    for (images, labels) in val_iter:
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            out = model(images)
            loss = criterion(out, labels)

        preds = out.argmax(1)
        preds = preds.detach().cpu()
        if labels.dim() != preds.dim():
            labels = soft_to_hard_labels(labels, ignore_index)
        labels = labels.detach().cpu()
        preds_for_iou.append(preds)
        labels_for_iou.append(labels)

        # Old metrics computation
        #batch_size = preds.size(0)
        #for b in range(batch_size):
        #    mIoU1.update(meanIoU(preds[b], labels[b], n_classes, ignore_index=ignore_index))
        #    valid_classes = torch.unique(labels[b][labels[b]<255])
        #    valid_classes = np.array(torch.nn.functional.one_hot(valid_classes, num_classes=n_classes).sum(0))
        #    classes_mIoU.update(np.array(class_meanIoU(preds[b], labels[b], n_classes, ignore_index=ignore_index)), valid_classes)
        #    mIoU2.update(classes_mIoU.last_values.sum()/valid_classes.astype(float).sum())

        val_loss += loss.item()
        val_iter.set_description(desc=f"val loss = {loss.item():.4f}")

    preds_for_iou = torch.cat(preds_for_iou, dim=0)
    labels_for_iou = torch.cat(labels_for_iou, dim=0)
    global_miou = meanIoU(preds_for_iou, labels_for_iou, n_classes, ignore_index=ignore_index)
    global_classes_iou = class_meanIoU(preds_for_iou, labels_for_iou, n_classes, ignore_index=ignore_index)

    val_loss = val_loss/len(val_loader)

    results = {
        "val_loss": val_loss,
        "mIoU1": mIoU1.avg,
        "mIoU2": mIoU2.avg,
        "global_miou": global_miou,
        "classes_mIoU": global_classes_iou
    }

    return results


def save_model(save_dict, cfg, save_dir, date, DATASET):
    date_str = date.strftime("%d-%m_%H-%M")
    ep = save_dict["epoch"]
    train_loss = save_dict["train_losses"][-1]
    train_global_miou = save_dict["train_global_miou"][-1]
    val_loss = save_dict["val_losses"][-1]
    val_global_miou = save_dict["val_global_miou"][-1]
    val_mIoU1 = save_dict["val_miou1"][-1]
    val_mIoU2 = save_dict["val_miou2"][-1]
    val_classes_mIoU = save_dict["val_classes_mIoU"][-1]
    train_classes_mIoU = save_dict["train_classes_mIoU"][-1]
    val_per_classes_mIoU = {v: val_classes_mIoU[k-1].item() for (k,v) in DATASET.classes.items() if k>0} if DATASET.ignore_index > 0 else {v: val_classes_mIoU[k].item() for (k,v) in DATASET.classes.items()}
    train_per_classes_mIoU = {v: train_classes_mIoU[k-1].item() for (k,v) in DATASET.classes.items() if k>0} if DATASET.ignore_index > 0 else {v: train_classes_mIoU[k].item() for (k,v) in DATASET.classes.items()}

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
        f.write(f"train loss = {train_loss:.4f}, train mIoU = {train_global_miou:.4f}, validation loss = {val_loss:.4f}, val mIoU = {val_global_miou:.4f}")
        f.write("\n")
        if train_per_classes_mIoU is not None:
            f.write(f"Train per class mIoU: {train_per_classes_mIoU}\n")
        if val_per_classes_mIoU is not None:
            f.write(f"Val per class mIoU: {val_per_classes_mIoU}\n")

    torch.save(save_dict, os.path.join(save_dir, save_name, save_name + ".pth.tar"))
    return save_name