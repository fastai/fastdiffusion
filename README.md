# fastdiffusion

- Big resource list: [What's the score? Review of latest Score Based Generative Modeling papers.](https://scorebasedgenerativemodeling.github.io/)
- [labml.ai Annotated PyTorch Paper Implementations](https://nn.labml.ai/)

## Useful resources

- [Stable Diffusion with 🧨 Diffusers](https://huggingface.co/blog/stable_diffusion)
- [Huggingface noteboooks](https://github.com/huggingface/notebooks/tree/main/diffusers)
- [Simple diffusion from Johno](https://colab.research.google.com/drive/12xmTDBYssfFVMUs0XhGXWP42SN6mijtt?usp=sharing)
- [Introduction to Diffusion Models for Machine Learning](https://www.assemblyai.com/blog/diffusion-models-for-machine-learning-introduction/) - AssemblyAI
- [Tutorial - What is a variational autoencoder?](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/)
- ["Grokking Stable Diffusion" from Johno](https://colab.research.google.com/drive/1dlgggNa5Mz8sEAGU0wFCHhGLFooW_pf1?usp=sharing)
- [Grokking SD Part 2: Textual Inversion](https://colab.research.google.com/drive/1RTHDzE-otzmZOuy8w1WEOxmn9pNcEz3u?usp=sharing)
- [What are Diffusion Models? · Lilian Weng](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [Generative Modeling by Estimating Gradients of the Data Distribution](https://yang-song.net/blog/2021/score/) (Yang Song)
- [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion)
- [Understanding VQ-VAE (DALL-E Explained Pt. 1)](https://ml.berkeley.edu/blog/posts/vq-vae/)
- [Diffusers Interpret](https://github.com/JoaoLages/diffusers-interpret). Model explainability, could be adapted to show some nice instructive plots. 
- []()

### Additional papers

- [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233), Dhariwal & Nichol 2021.

  Proposes architecture improvements (as of the state of the art in 2021, i.e. DDPM and DDIM) that could give some insight when we write models from scratch. In addition, introduces _classifier guidance_ to improve conditional image synthesis. This was later replaced by classifier-free guidance, but using a classifier looks like the natural thing to do for conditional generation.

- [Fast Sampling of Diffusion Models with Exponential Integrator](https://arxiv.org/abs/2204.13902).

  DEIS Scheduler. Authors claim excellent sampling results with as few as 12 steps. I haven't read it yet.

#### Application-oriented papers

Some of these tricks could be effective / didactic.

- [An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion](https://arxiv.org/abs/2208.01618).

  "Text Inversion": create new text embeddings from a few sample images. This effectively introduces new terms in the vocabulary that can be used in phrases for text to image generation.

- [DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242).

  Similar goal as the text inversion paper, but different approach I think (I haven't read it yet).

- [Prompt-to-Prompt Image Editing with Cross Attention Control](https://arxiv.org/abs/2208.01626), Hertz et al. 2022.

  Manipulate the cross-attention layers to produce changes in the text-to-image generation by replacing words, introducing new terms or weighting the importance of existing terms.

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
