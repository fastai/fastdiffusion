from copy import deepcopy

import torchvision
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

@call_parse
def main(
    experiment_name:str="ddpm_cifar10", # name of the experiment
    epochs:int=1000, # epochs
    batch_size:int=2048, # batch size
    image_size:int=32, # image size
    n_steps:int=1000, # number of steps in noise schedule
    beta_min:float=0.0001, # minimum noise level
    beta_max:float=0.02, # maximum noise level
    lr:float=3e-4, # learning rate
    gpu:int=0, # gpu id
    ):

    # gpu setup
    torch.cuda.set_device(gpu)

    # setup data
    path = untar_data(URLs.CIFAR)
    dblock = DataBlock(blocks = (TransformBlock, ImageBlock),
                       get_x = partial(generate_noise, size=(3,image_size,image_size)),
                       get_items = get_image_files,
                       splitter = IndexSplitter(range(batch_size)),
                       item_tfms=Resize(image_size), 
                       batch_tfms = Normalize.from_stats(torch.tensor([0.5]), torch.tensor([0.5])))
    dls = dblock.dataloaders(path, path=path, bs=batch_size)

    # setup model
    model = Unet(dim=32)

    # setup learner
    ddpm_learner = Learner(dls, model, cbs=[DDPMCallback(n_steps=n_steps, beta_min=beta_min, beta_max=beta_max)], loss_func=nn.MSELoss())
    ddpm_learner.load(f'{experiment_name}_{epochs}')
    #ddpm_learner.load('model')

    # setup sampling
    preds, targ = ddpm_learner.get_preds()
    preds = dls.after_batch.decode((preds,))[0][:64]

    grid = torchvision.utils.make_grid(preds, nrow=8)
    Image.fromarray(grid.permute(1,2,0).numpy().astype(np.uint8)).save(f'{experiment_name}_{epochs}_gen_samples.png')
