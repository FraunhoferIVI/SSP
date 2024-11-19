import math
from functools import partial
from torch.optim.lr_scheduler import LambdaLR


def _get_cosine_schedule_with_warmup_lr_lambda(current_step, num_warmup_steps, num_training_steps, num_cycles, start_lr, final_lr):
    if current_step < num_warmup_steps:
        progress = float(current_step) / float(max(1, num_warmup_steps))
        return start_lr + progress * (1 - start_lr)
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(final_lr, final_lr + (1 - final_lr) * 0.5 * (1. + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1, start_lr=0, final_lr=0):
    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        start_lr=start_lr,
        final_lr=final_lr
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def _get_constant_schedule_with_warmup_lr_lambda(current_step, num_warmup_steps, start_lr):
    if current_step < num_warmup_steps:
        progress = float(current_step) / float(max(1, num_warmup_steps))
        return start_lr + progress * (1 - start_lr)
    return 1.

def get_constant_schedule_with_warmup(optimizer, num_warmup_steps, last_epoch=-1, start_lr=0):
    lr_lambda = partial(
        _get_constant_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        start_lr=start_lr
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)
