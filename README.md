# fastdiffusion

- Big resource list: [What's the score? Review of latest Score Based Generative Modeling papers.](https://scorebasedgenerativemodeling.github.io/)
- List of diffusion papers: [Diffusion Reading Group](https://github.com/tmabraham/diffusion_reading_group#list-of-papers-to-cover)
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
- [Denoising Diffusion Probabilistic Model in Flax](https://github.com/yiyixuxu/denoising-diffusion-flax) by YiYi Xu, includes P2 weighting, self-conditioning, and EMA
- [A Traveler’s Guide to the Latent Space](https://sweet-hall-e72.notion.site/A-Traveler-s-Guide-to-the-Latent-Space-85efba7e5e6a40e5bd3cae980f30235f)
- [Denoising diffusion probabilistic models](https://github.com/acids-ircam/diffusion_models) - math+code tutorials in 4 notebooks
- Two articles from Sander Dieleman
  - [Diffusion models are autoencoders](https://benanne.github.io/2022/01/31/diffusion.html#fnref:gyom4)
  - [Guidance: a cheat code for diffusion models](https://benanne.github.io/2022/05/26/guidance.htm )
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

- [VToonify: Controllable High-Resolution Portrait Video Style Transfer](https://arxiv.org/abs/2209.11224)
  
  High-quality and temporally-coherent artistic portrait videos with flexible style controls.

## Improvements on simple diffusion

- Better denoising autoencoder (diffusion model)
  - Unet
  - Attention
  - P2 weighting
  - EMA
  - Self-conditioning
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
- Blur noise

## Applications

- Style transfer
- Super-res
- Colorisation
- Remove jpeg noise
- Remove watermarks
- Deblur
- CycleGAN / Pixel2Pixel -> change subject/location/weather/etc

### Diffusion Applications and Demos

- Stable Diffusion fine-tuning (for specific styles or domains).

  * [Pokemon fine-tuning](https://github.com/justinpinkney/stable-diffusion#fine-tuning).
  
  * Japanese Stable Diffusion [code](https://github.com/rinnakk/japanese-stable-diffusion#why-japanese-stable-diffusion) [demo](https://huggingface.co/spaces/rinna/japanese-stable-diffusion). They had to fine-tune the text embeddings too because the tokenizer was different.

- Stable Diffusion morphing / videos. [Code](https://github.com/nateraw/stable-diffusion-videos) by @nateraw based on [a gist by @karpathy](https://gist.github.com/karpathy/00103b0037c5aaea32fe1da1af553355).

- Image Variations. [Demo, with links to code](https://huggingface.co/spaces/lambdalabs/stable-diffusion-image-variations). Use the CLIP _image_ embeddings as conditioning for the generation, instead of the text embeddings. This requires fine-tuning of the model because, as far as I understand it, the text and image embeddings are not aligned in the embedding space. CLOOB doesn't have this limitation, but I heard (source: Boris Dayma from a conversation with Katherine Crowson) that attempting to train a diffusion model with CLOOB conditioning instead of CLIP produced less variety of results.

- Image to image generation. [Demo sketch -> image](https://huggingface.co/spaces/huggingface/diffuse-the-rest).
  
## Style Transfer
- Vincent's work: https://github.com/VinceMarron/style_transfer/blob/master/vgg_styletrans.py
- Johno's implementation of that plus some different style loss variants: https://colab.research.google.com/drive/1nTcswqeDmiW67WjEaQ8lAZP9v_5gKjCB?usp=sharing
- Insporation for the Sliced OT version: https://www.youtube.com/watch?v=ZFYZFlY7lgI&t=10s (Aside: NCA are super cool, I want to research them more as soon as the course craziness subsides)
- ImStack (which I like over just optimizing raw pixels): https://johnowhitaker.github.io/imstack/
- Q: fast style transfer (where a network does one-shot stylization) what networks and tricks seem to work best?
- Q: Do augmentations help with Getys style style transfer? TODO Johno test
- Q: What layers give good results? Would a different network to VGG16 be better?


## Other model ideas

- Latent space models
  - Imagenet
  - CLIP
  - Noisy clip
