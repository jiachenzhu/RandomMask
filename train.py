import os
import argparse

import torch
import torch.distributed as dist
from torchvision import datasets, transforms
from einops import rearrange

from helper import read_config, save_checkpoint, super_print, visualize_kernel
from augmentation import MaskGenerator, train_transform
from models import Encoder
from lars import LARS
from scheduler import Scheduler
from loss import VICRegLossModule

def main():
    parser = argparse.ArgumentParser(description='RandomMask')
    parser.add_argument('config_paths', nargs='+')
    args = parser.parse_args()
    config = read_config(args.config_paths)
    
    ngpus_per_node = torch.cuda.device_count()
    config.world_size = int(os.getenv('SLURM_NNODES', '1')) * ngpus_per_node
    
    if 'SLURM_JOB_NODELIST' in os.environ:
        host_name = os.getenv('SLURM_JOB_NODELIST').split(',')[0].strip()
        config.dist_url = f'tcp://{host_name}:56384'
    else:
        config.dist_url = f'tcp://localhost:56384'
    
    print(config)
    print(f"node id: {int(os.getenv('SLURM_NODEID', '0'))} {ngpus_per_node}")
    
    torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))

def main_worker(gpu, ngpus_per_node, config):
    device = torch.device(f"cuda:{gpu}")
    config.rank = gpu + int(os.getenv('SLURM_NODEID', '0')) * ngpus_per_node
    super_print(f"Rank {config.rank}")

    dist.init_process_group(backend='nccl', init_method=config.dist_url, world_size=config.world_size, rank=config.rank)
    torch.backends.cudnn.benchmark = True

    mask_generator = MaskGenerator(
        blur_mask_transform=transforms.GaussianBlur(config.blur_kernel_size, config.blur_sigma),
        mask_percentage=config.mask_percentage,
    )
    train_dataset = datasets.ImageFolder(f"{config.dataset_dir}/train", train_transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    
    device_batch_size = config.batch_size // config.world_size
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        sampler=train_sampler,
        batch_size=device_batch_size,
        num_workers=config.num_workers,
        drop_last=True,
        pin_memory=True,
    )
    
    model = Encoder(config.backbone_type, config.planes_multipliers).to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])
    model.train()

    optimizer = LARS(
        model.parameters(),
        lr=config.start_lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        eta=config.eta,
        weight_decay_filter=config.weight_decay_filter,
        lars_adaptation_filter=config.lars_adaptation_filter
    )
    
    lr_scheduler = Scheduler(
        "lr",
        config.num_epochs, len(train_loader),
        config.start_lr * config.batch_size / 256, config.end_lr * config.batch_size / 256,
        config.lr_num_warmup_epochs,
        decay=config.lr_decay
    )

    loss_module = VICRegLossModule()

    super_print(f"Number of Steps per Epoch: {len(train_loader)}", rank=config.rank)
    if not os.path.exists(os.path.join(f"{config.checkpoint_dir}/{config.comment}", "checkpoint.pt")):
        if config.rank == 0:
            checkpoint = {
                'epoch': 0,
                'config': config,
                'encoder_state_dict': model.module.backbone.state_dict(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            save_checkpoint(
                checkpoint=checkpoint,
                directory=f"{config.checkpoint_dir}/{config.comment}",
                filename=f"checkpoint.pt"
            )
    dist.barrier()
    
    # load checkpoint
    ckpt = torch.load(os.path.join(f"{config.checkpoint_dir}/{config.comment}", "checkpoint.pt"), map_location='cpu')
    start_epoch = ckpt['epoch'] + 1
    super_print(f'resuming from checkpoint {start_epoch}', rank=config.rank)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    
    for epoch in range(start_epoch, config.num_epochs + 1):
        train_sampler.set_epoch(epoch - 1)
        
        super_print(f"epoch {epoch} starts", rank=config.rank)
        for step, inputs in enumerate(train_loader, start=(epoch - 1) * len(train_loader)):
            # get lr
            lr = lr_scheduler.get_value(step)
            for g in optimizer.param_groups:
                g['lr'] = lr

            optimizer.zero_grad()
            
            x, _ = inputs
            x = x.to(device, non_blocking=True)
            mask_1, mask_2 = mask_generator(x)
            x1 = x * mask_1 + (torch.randn_like(x) * (1 - mask_1) * config.gaussian_noise)
            x2 = x * mask_2 + (torch.randn_like(x) * (1 - mask_2) * config.gaussian_noise)

            p1 = model(x1)
            p2 = model(x2)

            p1 = rearrange(p1, "b c h w -> (b h w) c")
            p2 = rearrange(p2, "b c h w -> (b h w) c")
            
            loss = loss_module(p1, p2, config.sim_coeff, config.std_coeff, config.cov_coeff)

            loss.backward()
            optimizer.step()

            super_print(f"{epoch:03d}-{step:04d} {lr_scheduler} -- {loss_module}", rank=config.rank)

        loss_module.reset_meters()

        if config.rank == 0:
            checkpoint = {
                'epoch': epoch,
                'config': config,
                'encoder_state_dict': model.module.backbone.state_dict(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            save_checkpoint(
                checkpoint=checkpoint,
                directory=f"{config.checkpoint_dir}/{config.comment}",
                filename=f"checkpoint.pt"
            )
            if epoch % config.checkpoint_frequency == 0:
                save_checkpoint(
                    checkpoint=checkpoint,
                    directory=f"{config.checkpoint_dir}/{config.comment}",
                    filename=f"checkpoint_{epoch}.pt"
                )
            
            visualize_kernel(
                model.module.backbone[0].weight.detach().cpu(),
                directory=f"vis/{config.comment}",
                filename=f"vis_{epoch}.png"
            )
    
    dist.barrier()
    dist.destroy_process_group()
            
if __name__ == '__main__':
    main()    