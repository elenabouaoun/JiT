import math
import time
from contextlib import nullcontext
import os
import shutil

import torch
import numpy as np
import cv2

import util.misc as misc
import util.lr_sched as lr_sched
import torch_fidelity
import copy


def train_one_epoch(model, model_without_ddp, data_loader, optimizer, device, epoch,
                    log_writer=None, args=None, max_optimizer_steps=None,
                    warmup_steps=0, step_callback=None):
    """Accumulate sample-weighted gradients, including a final short window.

    DDP ranks must consume equal-sized batches (as with DistributedSampler).
    max_optimizer_steps bounds pilots; this function never writes checkpoints.
    """
    device = torch.device(device)
    accum_steps = getattr(args, 'accum_steps', 1)
    if accum_steps < 1 or len(data_loader) == 0:
        raise ValueError('Positive accum_steps and a nonempty data loader are required')
    if max_optimizer_steps is not None and max_optimizer_steps < 1:
        raise ValueError('max_optimizer_steps must be positive')
    model.train(True)
    optimizer.zero_grad(set_to_none=True)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    window_samples = 0
    window_loss = 0.0
    updates = 0
    records = []
    updates_per_epoch = math.ceil(len(data_loader) / accum_steps)

    for data_iter_step, (x, labels) in enumerate(data_loader):
        if data_iter_step % accum_steps == 0:
            lr_sched.adjust_learning_rate(optimizer, epoch + data_iter_step / len(data_loader), args)
        boundary = ((data_iter_step + 1) % accum_steps == 0 or
                    data_iter_step + 1 == len(data_loader))
        sync = nullcontext() if boundary or not hasattr(model, 'no_sync') else model.no_sync()
        x = x.to(device, non_blocking=True).to(torch.float32).div_(255).mul_(2).sub_(1)
        labels = labels.to(device, non_blocking=True)
        batch_samples = x.size(0)
        # no_sync must cover both forward and backward.
        with sync:
            autocast = (torch.amp.autocast('cuda', dtype=torch.bfloat16)
                        if device.type == 'cuda' else nullcontext())
            with autocast:
                loss = model(x, labels)
            loss_value = misc.all_reduce_mean(loss.detach().item())
            if not math.isfinite(loss_value):
                raise FloatingPointError(f'Nonfinite training loss: {loss_value}')
            (loss * batch_samples).backward()
        window_samples += batch_samples
        window_loss += loss_value * batch_samples
        if not boundary:
            continue

        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(window_samples)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        model_without_ddp.update_ema()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        updates += 1
        record = dict(optimizer_step=updates, loss=window_loss / window_samples,
                      lr=optimizer.param_groups[0]['lr'], seconds=elapsed,
                      images=window_samples * misc.get_world_size(),
                      warmup=updates <= warmup_steps)
        record['images_per_second'] = record['images'] / elapsed
        if device.type == 'cuda':
            record.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        records.append(record)
        if step_callback is not None:
            step_callback(record)
        if (updates - 1) % args.log_freq == 0:
            print(f'Epoch {epoch} update {updates}: loss={record["loss"]:.6f}, '
                  f'images/s={record["images_per_second"]:.2f}')
        if log_writer is not None:
            step = epoch * updates_per_epoch + updates
            log_writer.add_scalar('train_loss', record['loss'], step)
            log_writer.add_scalar('lr', record['lr'], step)
        window_samples = 0
        window_loss = 0.0
        if max_optimizer_steps is not None and updates >= max_optimizer_steps:
            break
        started = time.perf_counter()

    measured = [record for record in records if not record['warmup']]
    seconds = sum(record['seconds'] for record in measured)
    return dict(optimizer_steps=updates, measured_steps=len(measured),
                images_per_second=sum(record['images'] for record in measured) / seconds if seconds else None,
                mean_loss=(sum(record['loss'] * record['images'] for record in measured) /
                           sum(record['images'] for record in measured)) if measured else None,
                steps=records)


def evaluate(model_without_ddp, args, epoch, batch_size=64, log_writer=None):

    model_without_ddp.eval()
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    num_steps = args.num_images // (batch_size * world_size) + 1

    # Construct the folder name for saving generated images.
    save_folder = os.path.join(
        args.output_dir,
        "{}-steps{}-cfg{}-interval{}-{}-image{}-res{}".format(
            model_without_ddp.method, model_without_ddp.steps, model_without_ddp.cfg_scale,
            model_without_ddp.cfg_interval[0], model_without_ddp.cfg_interval[1], args.num_images, args.img_size
        )
    )
    print("Save to:", save_folder)
    if misc.get_rank() == 0 and not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # switch to ema params, hard-coded to be the first one
    model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
        assert name in ema_state_dict
        ema_state_dict[name] = model_without_ddp.ema_params1[i]
    print("Switch to ema")
    model_without_ddp.load_state_dict(ema_state_dict)

    # ensure that the number of images per class is equal.
    class_num = args.class_num
    assert args.num_images % class_num == 0, "Number of images per class must be the same"
    class_label_gen_world = np.arange(0, class_num).repeat(args.num_images // class_num)
    class_label_gen_world = np.hstack([class_label_gen_world, np.zeros(50000)])

    for i in range(num_steps):
        print("Generation step {}/{}".format(i, num_steps))

        start_idx = world_size * batch_size * i + local_rank * batch_size
        end_idx = start_idx + batch_size
        labels_gen = class_label_gen_world[start_idx:end_idx]
        labels_gen = torch.Tensor(labels_gen).long().cuda()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            sampled_images = model_without_ddp.generate(labels_gen)

        torch.distributed.barrier()

        # denormalize images
        sampled_images = (sampled_images + 1) / 2
        sampled_images = sampled_images.detach().cpu()

        # distributed save images
        for b_id in range(sampled_images.size(0)):
            img_id = i * sampled_images.size(0) * world_size + local_rank * sampled_images.size(0) + b_id
            if img_id >= args.num_images:
                break
            gen_img = np.round(np.clip(sampled_images[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255))
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(os.path.join(save_folder, '{}.png'.format(str(img_id).zfill(5))), gen_img)

    torch.distributed.barrier()

    # back to no ema
    print("Switch back from ema")
    model_without_ddp.load_state_dict(model_state_dict)

    # compute FID and IS
    if log_writer is not None:
        if args.img_size == 256:
            fid_statistics_file = 'fid_stats/jit_in256_stats.npz'
        elif args.img_size == 512:
            fid_statistics_file = 'fid_stats/jit_in512_stats.npz'
        else:
            raise NotImplementedError
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=save_folder,
            input2=None,
            fid_statistics_file=fid_statistics_file,
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=False,
        )
        fid = metrics_dict['frechet_inception_distance']
        inception_score = metrics_dict['inception_score_mean']
        postfix = "_cfg{}_res{}".format(model_without_ddp.cfg_scale, args.img_size)
        log_writer.add_scalar('fid{}'.format(postfix), fid, epoch)
        log_writer.add_scalar('is{}'.format(postfix), inception_score, epoch)
        print("FID: {:.4f}, Inception Score: {:.4f}".format(fid, inception_score))
        shutil.rmtree(save_folder)

    torch.distributed.barrier()
