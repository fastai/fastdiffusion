# fastdiffusion

## Useful resources

- [Simple diffusion from Johno](https://colab.research.google.com/drive/12xmTDBYssfFVMUs0XhGXWP42SN6mijtt?usp=sharing)

## Improvements on simple diffusion

- Better denoising autoencoder (diffusion model)
  - Unet
  - Attention
- Predict noise / gradient (Score based diffusion)
- Latent diffusion (can not be a unet)
  - Attention
- Better loss functions
  - Perceptual + MSE + GAN (in the VAE)
- Preconditioning/scaling inputs and outputs
- Other crappifiers
- Data augmentation
- Better samplers / optimisers
- Initialisers such as pixelshuffle
- Learnable blur

## Applications

- Style transfer
- Super-res
- Colorisation
- Remove jpeg noise
- Remove watermarks
- Deblur
- CycleGAN / Pixel2Pixel -> change subject/location/weather/etc

## Other model ideas

- Latent space models
  - Imagenet
  - CLIP
  - Noisy clip
