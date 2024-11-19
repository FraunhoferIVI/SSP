import torch
import torch.nn as nn
from utils.losses import KLDivSoftLabelLoss
from utils.schedulers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

def get_criterion(loss_cfg, ignore_index, device="cuda", soft_labels=False, class_weights=None):
    loss_func = loss_cfg["loss_func"]
    class_weights = loss_cfg.get("class_weights", None)
    if class_weights is not None:
        class_weights = torch.FloatTensor(class_weights).to(device)
    if loss_func == "crossentropy":
        criterion = nn.CrossEntropyLoss(ignore_index=ignore_index if not soft_labels else -100, weight=class_weights)
    if loss_func == "kldiv":
        criterion = KLDivSoftLabelLoss(reduction='batchmean', softmax_on_target=loss_cfg["softmax_on_target"], temperature=loss_cfg["temperature"])
    return criterion


def get_optimizer_scheduler(optim_cfg, param_groups, iter_per_epoch, lr, num_epochs):
    opt_name = optim_cfg["optimizer"].lower()
    if opt_name == "sgd":
        optimizer = torch.optim.SGD(param_groups, lr=lr, momentum=optim_cfg["momentum"], weight_decay=optim_cfg["weight_decay"])
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(param_groups, lr=lr, weight_decay=optim_cfg["weight_decay"])
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=optim_cfg["weight_decay"])
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(param_groups, lr=lr, weight_decay=optim_cfg["weight_decay"])

    sch_name = optim_cfg["scheduler"].lower()
    if sch_name == "constant":
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=iter_per_epoch*optim_cfg["warmup_epochs"], start_lr=optim_cfg["start_lr"])
    elif sch_name == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=iter_per_epoch*optim_cfg["warmup_epochs"], 
        num_training_steps=iter_per_epoch*num_epochs, start_lr=optim_cfg["start_lr"], final_lr=optim_cfg["final_lr"])
    elif sch_name == "poly":
        scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=num_epochs*iter_per_epoch, power=optim_cfg["scheduler_power"])

    return optimizer, scheduler