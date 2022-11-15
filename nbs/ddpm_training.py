from copy import deepcopy

import torchvision
import wandb
from fastai.callback.wandb import *
from fastai.vision.all import *
from fastai.vision.gan import *
from fastcore.script import call_parse
from unet import Unet


class DDPMCallback(Callback):
    def __init__(self, n_steps, beta_min, beta_max, tensor_type=TensorImage):
        store_attr()

    def before_fit(self):
        self.beta = torch.linspace(self.beta_min, self.beta_max, self.n_steps).to(self.dls.device) # variance schedule, linearly increased with timestep
        self.alpha = 1. - self.beta 
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.sigma = torch.sqrt(self.beta)


    def before_batch_training(self):
        eps = self.tensor_type(self.xb[0]) # noise, x_T
        x0 = self.yb[0] # original images, x_0
        batch_size = x0.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long) # select random timesteps
        alpha_bar_t = self.alpha_bar[t].reshape(-1, 1, 1, 1)
        
        xt =  torch.sqrt(alpha_bar_t)*x0 + torch.sqrt(1-alpha_bar_t)*eps #noisify the image
        self.learn.xb = (xt, t) # input to our model is noisy image and timestep
        self.learn.yb = (eps,) # ground truth is the noise 


    def before_batch_sampling(self):
        xt = self.tensor_type(self.xb[0])
        for t in reversed(range(self.n_steps)):
            t_batch = torch.full((xt.shape[0],), t, device=xt.device, dtype=torch.long)
            z = torch.randn(xt.shape, device=xt.device) if t > 0 else torch.zeros(xt.shape, device=xt.device)
            alpha_t = self.alpha[t] # get noise level at current timestep
            alpha_bar_t = self.alpha_bar[t]
            sigma_t = self.sigma[t]
            alpha_bar_t_1 = self.alpha_bar[t-1]  if t > 0 else torch.tensor(1, device=xt.device)
            beta_bar_t = 1 - alpha_bar_t
            beta_bar_t_1 = 1 - alpha_bar_t_1
            x0hat = (xt - torch.sqrt(beta_bar_t) * self.model(xt, t_batch))/torch.sqrt(alpha_bar_t)
            x0hat = torch.clamp(x0hat, -1, 1)
            xt = x0hat * torch.sqrt(alpha_bar_t_1)*(1-alpha_t)/beta_bar_t + xt * torch.sqrt(alpha_t)*beta_bar_t_1/beta_bar_t + sigma_t*z    
            #xt = 1/torch.sqrt(alpha_t) * (xt - (1-alpha_t)/torch.sqrt(1-alpha_bar_t) * self.model(xt, t_batch))  + sigma_t*z # predict x_(t-1) in accordance to Algorithm 2 in paper
        self.learn.pred = (xt,)
        raise CancelBatchException

    def before_batch(self):
        if not hasattr(self, 'gather_preds'): self.before_batch_training()
        else: self.before_batch_sampling()




class ModelEma(nn.Module):
    def __init__(self, model, decay=0.9999, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

class EMACallback(Callback):
    order,run_valid,remove_on_fetch = MixedPrecision.order+1,False,True
    "Callback to implment Model Exponential Moving Average from PyTorch Image Models in fast.ai"
    def __init__(self, decay=0.9999, ema_device=None):
        store_attr()

    @torch.no_grad()
    def before_fit(self):
        self.ema_model = ModelEma(self.learn.model, self.decay, self.ema_device)

    def after_batch(self):
        self.ema_model.update(self.learn.model)

    def before_validate(self):
        self.temp_model = self.learn.model
        self.learn.model = self.ema_model.module

    def after_validate(self):
        self.learn.model = self.temp_model

    @torch.no_grad()
    def after_fit(self):
        self.learn.model = self.ema_model.module
        self.ema_model = None
        self.remove_cb(EMACallback)


def save_grid(preds, experiment_name, epoch, wandb_to_log=False):
    grid = torchvision.utils.make_grid(preds, nrow=math.ceil(preds.shape[0] ** 0.5), padding=0)
    file_name = f'{experiment_name}_{epoch}_samples.png'
    Image.fromarray(grid.permute(1,2,0).numpy().astype(np.uint8)).save(file_name)
    if wandb_to_log: 
        return wandb.Image(file_name)

@patch
def log_predictions(self:WandbCallback):
    try:
        inp,preds,targs,out = self.learn.fetch_preds.preds
        preds = self.dls.after_batch.decode((preds,))[0]
        wandb_img = save_grid(preds, self.experiment_name, self.learn.epoch+1, wandb_to_log=True)
        wandb.log({"samples": [wandb_img]}, step=self._wandb_step)
    except: pass

@patch
def before_validate(self:WandbCallback):
    if (self.epoch+1) % self.demo_every == 0: self.learn.add_cb(FetchPredsCallback(dl=self.valid_dl, with_input=True, with_decoded=True, reorder=self.reorder))
    else: self.learn.remove_cb(FetchPredsCallback)

@call_parse
def main(
    experiment_name:str="ddpm_cifar10", # name of the experiment
    dataset_name:str="CIFAR", # name of the dataset
    batch_size:int=2048, # batch size
    image_size:int=32, # image size
    num_channels:int=3, # number of channels
    n_steps:int=1000, # number of steps in noise schedule
    beta_min:float=0.0001, # minimum noise level
    beta_max:float=0.02, # maximum noise level
    lr:float=2e-4, # learning rate
    epochs:int=1000, # number of epochs for training
    dropout:float=0., # dropout rate
    dim_mults:Param("List of channel multipliers", int, nargs='+')=[1,2,4,8], # channel multipliers
    ema_decay:float=0.9999, # decay rate for EMA
    demo_every:int=10, # demo every n epochs
    demo_end:bool=True, # demo at the end of training
    gpu:int=0, # gpu id
    wandb_project:str="fastai_ddpm", # wandb project name
    wandb_entity:str="tmabraham", # wandb entity name
    ):
    # setup wandb
    wandb.init(project=wandb_project, entity=wandb_entity, config=locals(), save_code=True)

    # gpu setup
    torch.cuda.set_device(gpu)

    # setup data
    if dataset_name == "CIFAR":
        path = untar_data(URLs.CIFAR)
    if dataset_name == "FashionMNIST": # if the dataset is on URLs, could be as simple as getattr(URLs, dataset_name)
        path  = untar_data('https://github.com/DeepLenin/fashion-mnist_png/raw/master/data.zip')

    if num_channels == 1: image_block = ImageBlock(cls=PILImageBW)
    else: image_block = ImageBlock(cls=PILImage)
    dblock = DataBlock(blocks = (TransformBlock, image_block),
                       get_x = partial(generate_noise, size=(num_channels,image_size,image_size)),
                       get_items = get_image_files,
                       splitter = IndexSplitter(range(batch_size)),
                       item_tfms=Resize(image_size), 
                       batch_tfms = [Normalize.from_stats(torch.tensor([0.5]), torch.tensor([0.5])), Flip()])
    dls = dblock.dataloaders(path, path=path, bs=batch_size)

    # setup model
    model = Unet(dim=image_size, dropout=dropout, dim_mults = tuple(dim_mults), channels=num_channels)

    # setup learner
    cbs=[DDPMCallback(n_steps=n_steps, beta_min=beta_min, beta_max=beta_max),ReduceLROnPlateau(patience=10,factor=2),WandbCallback(log_model=True, log_preds_every_epoch=True)]
    if ema_decay!=0:
        cbs.append(EMACallback(decay=ema_decay))
    ddpm_learner = Learner(dls, model, cbs=cbs, loss_func=nn.MSELoss())
    ddpm_learner.experiment_name = experiment_name
    ddpm_learner.demo_every = demo_every

    # start training 
    ddpm_learner.fit(epochs,lr)
    ddpm_learner.save(f'{experiment_name}_{epochs}')

    # demo at the end of training
    if demo_end:
        preds, targ = ddpm_learner.get_preds()
        preds = ddpm_learner.dls.after_batch.decode((preds,))[0]
        wandb_img = save_grid(preds, experiment_name, 'end', wandb_to_log=True)
        wandb.log({"samples": [wandb_img]})
