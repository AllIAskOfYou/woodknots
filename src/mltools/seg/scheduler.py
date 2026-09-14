import math
import torch
from torch.optim.lr_scheduler import LambdaLR

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    num_warmup_epochs: int,
    total_iters: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    Create a schedule with linear warmup followed by cosine decay.
    This is the standard implementation used in Hugging Face Transformers.
    """
    def lr_lambda(current_step):
        batches_per_epoch = total_iters // num_epochs
        current_epoch = current_step // batches_per_epoch
        num_warmup_steps = num_warmup_epochs * batches_per_epoch
        current_step_in_epoch = current_step % batches_per_epoch
        current_step = current_epoch * batches_per_epoch + current_step_in_epoch
        # 1. Linear warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # 2. Cosine decay phase
        progress = float(current_step - num_warmup_steps) / float(
            max(1, total_iters - num_warmup_steps)
        )
        # Cosine annealing calculation (0.5 * (1 + cos(pi * progress))
        cosine_lr_multiple = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
        return max(0.0, cosine_lr_multiple)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


# The schedules below share get_cosine_schedule_with_warmup's signature and its
# warmup, so an experiment can swap train.scheduler._target_ and nothing else:
# Hydra merges the experiment's scheduler node into the default one rather than
# replacing it, so num_epochs/num_warmup_epochs/num_cycles are always passed and
# must be accepted (num_cycles is simply unused here). Keeping the warmup
# identical means the three differ only in the shape of the decay.


def get_constant_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    num_warmup_epochs: int,
    total_iters: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """Linear warmup, then a constant learning rate."""
    num_warmup_steps = num_warmup_epochs * (total_iters // num_epochs)

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_linear_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    num_warmup_epochs: int,
    total_iters: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """Linear warmup, then linear decay to zero at the end of training."""
    num_warmup_steps = num_warmup_epochs * (total_iters // num_epochs)

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, total_iters - num_warmup_steps)
        )
        return max(0.0, 1.0 - progress)

    return LambdaLR(optimizer, lr_lambda, last_epoch)
