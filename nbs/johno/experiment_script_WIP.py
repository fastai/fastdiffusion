# pip install --upgrade -q fastcore fastai diffusers datasets lpips pytorch-fid ema-pytorch wandb clean-fid
#python experiment_script_WIP.py --dataset_name faces --batch_size 256 --img_size 64 --num_epochs 5 --comments "5epch speedrun" --job_type "Quick Run" --calc_fid_every_n_steps 1000 --use_device cuda:0 --perceptual_loss_scale 0.05 --n_samples_for_FID 5000 --log_samples_every_n_steps 1000 --ema --ema_beta 0.998 --blur

import wandb
import torch
import torchvision
import lpips
import os
import time
from torch import nn
from torchvision import transforms as T
from datasets import load_dataset
from fastai.data.all import *
from fastai.vision.all import *
from fastai.callback.wandb import *
from diffusers import UNet2DModel
from PIL import Image as Image_PIL
from fastcore.script import *
from fastcore.basics import patch_to
from ema_pytorch import EMA
from cleanfid import fid

# Class for crappified image
class PILImageNoised(PILImage): pass
class TensorImageNoised(TensorImage):
    def show(self, ctx=None, **kwargs):
        super().show(ctx=ctx, **kwargs)
PILImageNoised._tensor_cls = TensorImageNoised

# Transform (TODO experiment)
class Crappify(Transform):
    def __init__(self, blur=True):
        super().__init__()
        self.blur = blur
    def encodes(self, x:TensorImageNoised): 
        x = IntToFloatTensor()(x)
        if self.blur:
            x = T.GaussianBlur(3)(x) # Add some random blur
        noise_amount = torch.rand(x.shape[0], device=x.device)
        noise = torch.rand_like(x, device=x.device)
        x = torch.lerp(x, noise, noise_amount.view(-1, 1, 1, 1)) * 255
        return x

# The UNET Model (wraps diffusers unet)
class Unetwrapper(Module):
    def __init__(self, in_channels=3, out_channels=3, sample_size=64,
                block_out_channels=(32, 64, 128, 128,),
                down_block_types=("DownBlock2D","DownBlock2D","AttnDownBlock2D","AttnDownBlock2D",),
                up_block_types=("AttnUpBlock2D","AttnUpBlock2D","UpBlock2D","UpBlock2D",)):
        super().__init__()
        self.net = UNet2DModel(
            sample_size=sample_size,  # the target image resolution
            in_channels=in_channels,  # the number of input channels, 3 for RGB images
            out_channels=out_channels,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=block_out_channels,  # <<< Experiment with number of layers and how many
            down_block_types=down_block_types, 
            up_block_types=up_block_types,
        )
    def forward(self, x): return self.net(x, 0).sample # Timestep cond always set to 0

# Callback for logging samples
class LogSamplesBasicCallback(Callback):
    def __init__(self, n_sampling_steps=40, n_samples_row=8, img_size=64,
                log_samples_after_epoch=False, log_samples_every_n_steps=1000):
        super().__init__()
        self.n_sampling_steps = n_sampling_steps
        self.n_samples_row = n_samples_row
        self.img_size = img_size
        self.log_samples_every_n_steps=log_samples_every_n_steps
        self.log_samples_after_epoch = log_samples_after_epoch
        
    def log_samples(self):
        print('log samples')
        model = self.learn.model
        n_steps = self.n_sampling_steps
        x = torch.rand(self.n_samples_row**2, 3, self.img_size, self.img_size).to(model.net.device)

        for i in range(n_steps):
            with torch.no_grad():
                pred = model(x)
            mix_factor = 1/(n_steps - i)
            x = x*(1-mix_factor) + pred*mix_factor

        im = torchvision.utils.make_grid(x.detach().cpu(), nrow=self.n_samples_row).permute(1, 2, 0).clip(0, 1) * 255
        im = Image_PIL.fromarray(np.array(im).astype(np.uint8))
        wandb.log({'Sample generations basic':wandb.Image(im)})

    def after_epoch(self):
        if self.log_samples_after_epoch:
            self.log_samples()


    def after_step(self):
        if self.train_iter%self.log_samples_every_n_steps == 0 and self.train_iter>0:
            self.log_samples()

# TODO use better system here so multiple experiments running at once don't ever clash by saving to the same folder!
class FIDCallback(Callback):
    
    def __init__(self, n_sampling_steps=40, n_samples_for_FID=500, img_size=64,
                fid_after_epoch=False, calc_fid_every_n_steps=1000, batch_size=64):
        super().__init__()
        self.n_sampling_steps = n_sampling_steps
        self.n_samples_for_FID = n_samples_for_FID
        self.img_size = img_size
        self.calc_fid_every_n_steps = calc_fid_every_n_steps
        self.fid_after_epoch = fid_after_epoch
        self.batch_size = batch_size

    def log_fid(self):
        print('log FID')
        model = self.learn.model
        n_steps = self.n_sampling_steps
        os.system('rm -rf generated_samples;mkdir generated_samples')
        for start in range(0, self.n_samples_for_FID, self.batch_size):
            end = min(start+self.batch_size, self.n_samples_for_FID)
            n = end-start
            if n > 0:
                x = torch.rand(n, 3, self.img_size, self.img_size).to(model.net.device)
                for i in range(n_steps):
                    with torch.no_grad():
                        pred = model(x)
                    mix_factor = 1/(n_steps - i)
                    x = x*(1-mix_factor) + pred*mix_factor
            for i, im in enumerate(x):
                im = im.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255
                im = Image_PIL.fromarray(np.array(im).astype(np.uint8))
                im.save(f'generated_samples/{start+i:06}.jpeg', quality=90)
                
        fid_score = fid.compute_fid('generated_samples/', 'valid_samples/')
        wandb.log({'FID':fid_score})

    def after_epoch(self):
        if self.fid_after_epoch:
            self.log_fid()

    def after_step(self):
        if self.train_iter%self.calc_fid_every_n_steps == 0 and self.train_iter>0:
            self.log_fid()
            
# EMA
class EMAFIDCallback(Callback):
    def __init__(self, model, beta = 0.9999,              # exponential moving average factor
                update_after_step = 100,    # only after this number of .update() calls will it start updating
                update_every = 10,
                n_sampling_steps=40, n_samples_for_FID=500, img_size=64,
                fid_after_epoch=False, calc_fid_every_n_steps=1000, batch_size=64):
        super().__init__()
        self.n_sampling_steps = n_sampling_steps
        self.n_samples_for_FID = n_samples_for_FID
        self.img_size = img_size
        self.calc_fid_every_n_steps = calc_fid_every_n_steps
        self.fid_after_epoch = fid_after_epoch
        self.batch_size = batch_size
        self.ema = EMA(
            model,
            beta = beta,              # exponential moving average factor
            update_after_step = update_after_step,    # only after this number of .update() calls will it start updating
            update_every = update_every,          # how often to actually update, to save on compute (updates every 10th .update() call)
        )
        
    def log_fid(self):
        print('log FID')
        model = self.ema # Instead of self.learn.model
        n_steps = self.n_sampling_steps
        os.system('rm -rf generated_samples_ema;mkdir generated_samples_ema')
        for start in range(0, self.n_samples_for_FID, self.batch_size):
            end = min(start+self.batch_size, self.n_samples_for_FID)
            n = end-start
            if n > 0:
                x = torch.rand(n, 3, self.img_size, self.img_size).to(self.learn.model.net.device)
                for i in range(n_steps):
                    with torch.no_grad():
                        pred = model(x)
                    mix_factor = 1/(n_steps - i)
                    x = x*(1-mix_factor) + pred*mix_factor
            for i, im in enumerate(x):
                im = im.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255
                im = Image_PIL.fromarray(np.array(im).astype(np.uint8))
                im.save(f'generated_samples_ema/{start+i:06}.jpeg', quality=90)

        fid_score = fid.compute_fid('generated_samples_ema/', 'valid_samples/')
        wandb.log({'FID_EMA':fid_score})


    def after_epoch(self):
        if self.fid_after_epoch:
            self.log_fid()

    def after_step(self):
        self.ema.update() # Updates the EMA model
        if self.train_iter%self.calc_fid_every_n_steps == 0 and self.train_iter>0:
            self.log_fid()
        
        

@call_parse
def main(dataset_name = 'cifar10', # Dataset name: faces, flowers or cifar10
         img_size = 32, # Image size in pixels
         blur = True, # Add blur as a crappifier option
         batch_size=64,
         perceptual_loss_net = 'alex', # 'vgg' or 'alex' - used for perceptual loss
         mse_loss_scale = 1, # How much should we weight the MSE loss in the loss calc
         perceptual_loss_scale = 0.1, # ow much should we weight the perceptual loss in the loss calc
         num_epochs = 10, 
         job_type = 'script test', # For W&B
         comments = 'Script dev',  # For W&B
         lr_max = 1e-4,
         n_sampling_steps = 40,
         n_samples_row = 8,
         log_samples_every_n_steps = 1000,
         calc_fid_every_n_steps = 5000,
         n_samples_for_FID = 1500, # Larger = better measure but slower
         log_samples_after_epoch = False, # Log samples every epoch as well as every log_samples_every_n_steps
         fid_after_epoch = False, # Calc FID after every epoch as well as every calc_fid_every_n_steps
         use_device = 'cuda:0', # Which device should we use? Usually cuda:0
         ema=False, # Use EMA?
         ema_beta=0.999, # EMA factor
        ):
    
    # Not sure how to do lists as args, setting manually for now
    
    # Defaults for a lot of weekend runs:
#     block_out_channels=[32, 64, 128, 128], # UNET block channels
#     down_block_types=["DownBlock2D","DownBlock2D","AttnDownBlock2D","AttnDownBlock2D"], # UNET downblock types
#     up_block_types=["AttnUpBlock2D","AttnUpBlock2D","UpBlock2D","UpBlock2D"], # UNET upblock types
    # New test
    block_out_channels=[128, 256, 512], # UNET block channels
    down_block_types=["DownBlock2D","DownBlock2D","AttnDownBlock2D"], # UNET downblock types
    up_block_types=["AttnUpBlock2D","UpBlock2D","UpBlock2D"], # UNET upblock types
            
            
    device = torch.device(use_device if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}') 

    # Load dataset
    tfm = T.Compose([T.Resize(img_size), T.CenterCrop(img_size)])
    def transforms(examples):
        examples["image"] = [tfm(image.convert("RGB")) for image in examples["image"]]
        return examples
    if dataset_name == 'flowers':
        dataset = load_dataset('huggan/flowers-102-categories')
        dataset = dataset.with_transform(transforms)
        validation_set = dataset['train'][:n_samples_for_FID]
        dataset = dataset['train'][n_samples_for_FID:]
    elif dataset_name == 'faces':
        dataset = load_dataset('huggan/CelebA-faces')
        dataset = dataset.with_transform(transforms)
        validation_set = dataset['train'][:n_samples_for_FID]
        dataset = dataset['train'][n_samples_for_FID:] 
    elif dataset_name == 'afhq2':
        dataset = load_dataset('huggan/AFHQv2')
        dataset = dataset.with_transform(transforms)
        validation_set = dataset['train'][:n_samples_for_FID]
        dataset = dataset['train'][n_samples_for_FID:] 
    elif dataset_name == 'cifar10':
        dataset = load_dataset('cifar10')
        dataset = dataset.with_transform(transforms)
        dataset = dataset.remove_columns("label").rename_column("img", "image")
        validation_set = dataset['train'][:n_samples_for_FID]
        dataset = dataset['train'][:]

    # Save some samples for FID
    os.system('rm -rf valid_samples;mkdir valid_samples')
    for i, im in enumerate(validation_set['image']):
        im.save(f'valid_samples/{i:06}.jpeg')


    # Dataloader
    dblock = DataBlock(blocks=(ImageBlock(cls=PILImageNoised),ImageBlock(cls=PILImage)),
                       get_items=lambda pth: range(len(dataset['image'])), # Gets the indexes
                       getters=[lambda idx: np.array(dataset['image'][idx])]*2,
                       batch_tfms=[Crappify])
    dls = dblock.dataloaders('', bs=batch_size, device=device)
    dls.show_batch()

    # Model
    model = Unetwrapper(sample_size=img_size, 
                        block_out_channels=block_out_channels,
                        down_block_types=down_block_types, 
                        up_block_types=up_block_types).to(device)

    # Loss function
    loss_fn_perceptual = lpips.LPIPS(net=perceptual_loss_net).to(device)
    loss_fn_perceptual.net.to(device)
    loss_fn_mse = MSELossFlat()
    def combined_loss(preds, y):
        return  loss_fn_mse(preds, y) * mse_loss_scale + loss_fn_perceptual(preds, y).mean() * perceptual_loss_scale

    # Learner
    learn = Learner(dls, model, loss_func=combined_loss)
    learn.to(device) 

    # Config with all the settings
    cfg = dict(model.net.config)
    cfg['num_epochs'] = num_epochs
    cfg['lr_max'] = lr_max
    cfg['comments'] = comments
    cfg['dataset'] = dataset_name
    cfg['perceptual_loss_scale'] = perceptual_loss_scale
    cfg['mse_loss_scale'] = mse_loss_scale

    # Training!
    wandb.init(project='fastdiffusion', job_type=job_type, config=cfg)
    
    # I only want to commit logs to wandb every 50 iters otherwise things get too slow
    @patch_to(WandbCallback)
    def after_batch(self):
        "Log hyper-parameters and training loss"
        if self.training:
            batch_time = time.perf_counter() - self.ti_batch
            self._wandb_step += 1
            self._wandb_epoch += 1/self.n_iter
            hypers = {f'{k}_{i}':v for i,h in enumerate(self.opt.hypers) for k,v in h.items()}
            if self.train_iter%50 == 0: # Only logging every 50th train iter
                wandb.log({'epoch': self._wandb_epoch, 'train_loss': self.smooth_loss, 
                           'raw_loss': self.loss, 'train_samples_per_sec': len(self.xb[0]) / batch_time,
                           **hypers}, step=self._wandb_step)
    
    # logging callbacks can probably be consolidated...
    callbacks = [
        WandbCallback(n_preds=8), 
        LogSamplesBasicCallback(n_sampling_steps=n_sampling_steps, n_samples_row=8, img_size=img_size,
                log_samples_after_epoch=log_samples_after_epoch, log_samples_every_n_steps=log_samples_every_n_steps), 
        FIDCallback(n_sampling_steps=n_sampling_steps, calc_fid_every_n_steps=calc_fid_every_n_steps, img_size=img_size,
                fid_after_epoch=fid_after_epoch, n_samples_for_FID=n_samples_for_FID, batch_size = batch_size),
        ]
    if ema:
        callbacks.append(EMAFIDCallback(learn.model, beta=ema_beta, n_sampling_steps=n_sampling_steps, 
                                        calc_fid_every_n_steps=calc_fid_every_n_steps, img_size=img_size,
                                        fid_after_epoch=fid_after_epoch, n_samples_for_FID=n_samples_for_FID, batch_size = batch_size))
    learn.fit_one_cycle(cfg['num_epochs'], lr_max=cfg['lr_max'], cbs=callbacks)
    wandb.finish()