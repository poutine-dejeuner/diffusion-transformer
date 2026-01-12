'''
Author: Emilio Morales (mil.mor.mor@gmail.com)
        Dec 2023
'''
import torch
from torch import optim
from torch.utils.data import DataLoader
import time
import os
import warnings
from copy import deepcopy
from collections import OrderedDict
import argparse
import wandb
import yaml
from types import SimpleNamespace
from fid import get_fid
from image_datasets import create_loader, create_dataset
from dit import DiT
from utils import *
from diff_utils import *

from icecream import ic, install
install()

warnings.filterwarnings("ignore")


@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def train(model_dir, fid_real_dir,
          iter_interval, fid_interval, conf):
    # Initialize wandb
    if not conf.d:
        wandb.init(
            project="diffusion-transformer",
            config=vars(conf),
            name=os.path.basename(model_dir)
        )

    data_dir = conf.data_dir
    if fid_real_dir == None:
        fid_real_dir = data_dir
    img_size = conf.img_size
    if isinstance(img_size, (tuple, list)):
        img_h, img_w = img_size
    else:
        img_h, img_w = img_size, img_size
    batch_size = conf.batch_size
    lr = conf.lr
    dim = conf.dim
    ema_decay = conf.ema_decay
    patch_size = conf.patch_size
    depth = conf.depth
    heads = conf.heads
    mlp_dim = conf.mlp_dim
    k = conf.k
    fid_batch_size = conf.fid_batch_size
    gen_batch_size = conf.gen_batch_size
    steps = conf.steps
    n_fid_real = conf.n_fid_real
    n_fid_gen = conf.n_fid_gen
    n_iter = conf.n_iter
    plot_shape = conf.plot_shape
    seed = conf.seed
    channels = conf.channels
    grad_clip = getattr(conf, 'grad_clip', 1.0)
    warmup_steps = getattr(conf, 'warmup_steps', 1000)
    weight_decay = getattr(conf, 'weight_decay', 0.0)

    # dataset
    dataset = create_dataset(
        data_dir, (img_h, img_w), batch_size, channels
    )
    train_loader = DataLoader(dataset, conf.batch_size)

    # model
    model = DiT(img_size, dim, patch_size,
                depth, heads, mlp_dim, k, in_channels=channels)
    diffusion = Diffusion()
    optimizer = optim.AdamW(model.parameters(), lr=lr,
                            betas=(0.9, 0.999), weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # create ema
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)

    # logs and ckpt config
    gen_dir = os.path.join(model_dir, 'fid')
    log_img_dir = os.path.join(model_dir, 'log_img')
    log_dir = os.path.join(model_dir, 'log_dir')
    os.makedirs(gen_dir, exist_ok=True)
    os.makedirs(log_img_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    last_ckpt = os.path.join(model_dir, './last_ckpt.pt')
    best_ckpt = os.path.join(model_dir, './best_ckpt.pt')

    if os.path.exists(last_ckpt):
        ckpt = torch.load(last_ckpt)
        start_iter = ckpt['iter'] + 1  # start from iter + 1
        best_fid = ckpt['best_fid']
        model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['opt'])
        print(f'Checkpoint restored at iter {start_iter}; '
              f'best FID: {best_fid}')
    else:
        start_iter = 1
        best_fid = 1000.  # init with big value
        print(f'New model')

    # plot shape
    sz = (plot_shape[0] * plot_shape[1], channels, img_h, img_w)

    # train
    start = time.time()
    train_loss = 0.0
    grad_norm_total = 0.0
    update_ema(ema, model, decay=ema_decay)
    model.train()
    ema.eval()  # EMA model should always be in eval mode
    train_iter = iter(train_loader)
    for idx in range(n_iter):
        i = idx + start_iter

        # Learning rate warmup
        if i <= warmup_steps:
            lr_scale = i / warmup_steps
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * lr_scale
        try:
            inputs = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            inputs = next(train_iter)

        inputs = inputs.to(device)
        xt, t, target = diffusion.diffuse(inputs)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(xt, t)
        loss = loss_fn(outputs, target)
        loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), grad_clip)
        grad_norm_total += grad_norm.item()

        optimizer.step()
        update_ema(ema, model)
        train_loss += loss.item()

        if i % iter_interval == 0:
            # plot
            gen_batch = diffusion.sample(ema, sz, steps=steps, seed=seed)
            plot_path = os.path.join(log_img_dir, f'{i:04d}.png')
            plot_batch(deprocess(gen_batch), plot_shape,
                       plot_path, img_size=(img_h, img_w))
            # metrics
            train_loss /= iter_interval
            grad_norm_avg = grad_norm_total / iter_interval
            elapsed_time = time.time() - start
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Time for iter {i} is {elapsed_time:.4f}sec '
                  f'Train loss: {train_loss:.4f} Grad norm: {grad_norm_avg:.4f} '
                  f'LR: {current_lr:.6f}')

            # wandb logging
            if not conf.d:
                wandb.log({
                    'train_loss': train_loss,
                    'grad_norm': grad_norm_avg,
                    'learning_rate': current_lr,
                    'iteration': i,
                }, step=i)

            train_loss = 0.0
            grad_norm_total = 0.0
            start = time.time()
            model.train()

        if i % fid_interval == 0:
            # fid
            print('Generating eval batches...')
            gen_batches(
                diffusion, ema, n_fid_real, gen_batch_size,
                steps, gen_dir, (img_h, img_w), channels
            )
            fid = get_fid(
                fid_real_dir, gen_dir, n_fid_real, n_fid_gen,
                device, fid_batch_size
            )
            print(f'FID: {fid}')

            # wandb logging
            if not conf.d:
                wandb.log({
                    'FID': fid,
                    'best_FID': min(fid, best_fid)
                }, step=i)

            # ckpt
            ckpt_data = {
                'iter': i,
                'model': model.state_dict(),
                'ema': ema.state_dict(),
                'opt': optimizer.state_dict(),
                'fid': fid,
                'best_fid': min(fid, best_fid),
                'train_loss': train_loss
            }

            torch.save(ckpt_data, last_ckpt)
            print(f'Checkpoint saved at iter {i}')

            if fid <= best_fid:
                torch.save(ckpt_data, best_ckpt)
                best_fid = fid
                print(f'Best checkpoint saved at iter {i}')

            start = time.time()
            model.train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', action="store_true", default=False)
    parser.add_argument('--config', type=str, default="config.yaml")
    args = parser.parse_args()

    # Load config from YAML or Python file
    conf = load_config_from_yaml(args.config)

    conf.d = args.d

    train(
        conf.model_dir, conf.fid_real_dir,
        conf.iter_interval, conf.fid_interval, conf
    )


if __name__ == '__main__':
    main()
