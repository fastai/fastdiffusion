# fastdiffusion

- Big resource list: [What's the score? Review of latest Score Based Generative Modeling papers.](https://scorebasedgenerativemodeling.github.io/)
- [labml.ai Annotated PyTorch Paper Implementations](https://nn.labml.ai/)

## Useful resources

- [Stable Diffusion with 🧨 Diffusers](https://huggingface.co/blog/stable_diffusion)
- [Simple diffusion from Johno](https://colab.research.google.com/drive/12xmTDBYssfFVMUs0XhGXWP42SN6mijtt?usp=sharing)
- [Introduction to Diffusion Models for Machine Learning](https://www.assemblyai.com/blog/diffusion-models-for-machine-learning-introduction/) - AssemblyAI
- [Tutorial - What is a variational autoencoder?](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/)
- ["Grokking Stable Diffusion" from Johno](https://colab.research.google.com/drive/1dlgggNa5Mz8sEAGU0wFCHhGLFooW_pf1?usp=sharing)
- [Grokking SD Part 2: Textual Inversion](https://colab.research.google.com/drive/1RTHDzE-otzmZOuy8w1WEOxmn9pNcEz3u?usp=sharing)
- [What are Diffusion Models? · Lilian Weng](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [Generative Modeling by Estimating Gradients of the Data Distribution](https://yang-song.net/blog/2021/score/) (Yang Song)
- [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion)
- [Understanding VQ-VAE (DALL-E Explained Pt. 1)](https://ml.berkeley.edu/blog/posts/vq-vae/)
- []()

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
