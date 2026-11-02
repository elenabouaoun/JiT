import math


def adjust_learning_rate(optimizer, epoch, args):
    """Epoch schedule sampled at optimizer-update starts, not microbatch starts.

    The caller supplies epoch + update_index / ceil(microbatches / accum_steps).
    Thus warmup_epochs and epochs retain their epoch meaning under accumulation;
    a short final window counts as one update. The first update uses progress zero
    and the final training update precedes the exact cosine endpoint.
    """
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        if args.lr_schedule == "constant":
            lr = args.lr
        elif args.lr_schedule == "cosine":
            lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
                (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
        else:
            raise NotImplementedError
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
