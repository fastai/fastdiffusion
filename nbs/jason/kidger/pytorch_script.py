import logging, torch, torchvision, torch.nn.functional as F, torchvision.transforms.functional as TF, matplotlib as mpl
from matplotlib import pyplot as plt
from functools import partial
from torch import tensor,nn,optim
from torch.utils.data import DataLoader,default_collate
from torchvision.utils import make_grid
from datasets import load_dataset,load_dataset_builder
from miniai.datasets import *
from miniai.learner import *
from fastprogress import progress_bar
from timm.optim.adabelief import AdaBelief
from fastai.callback.schedule import combined_cos
from fastai.layers import SequentialEx, MergeLayer
from fastai.losses import MSELossFlat
from fastcore.basics import store_attr
from fastai.torch_core import TensorImage
from fastai.optimizer import OptimWrapper
from einops import rearrange, repeat
from dataclasses import field
import PIL
import numpy as np

import typing
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union


import array
import functools as ft
import gzip
import os
import struct
import urllib.request
#import diffrax as dfx  # https://github.com/patrick-kidger/diffrax
import einops  # https://github.com/arogozhnikov/einops
import matplotlib.pyplot as plt
#import optax  # https://github.com/deepmind/optax
from functorch import vmap, jvp



def static_field(**kwargs):
    """Used for marking that a field should _not_ be treated as a leaf of the PyTree
    of a [`equinox.Module`][]. (And is instead treated as part of the structure, i.e.
    as extra metadata.)
    !!! example
        ```python
        class MyModule(equinox.Module):
            normal_field: int
            static_field: int = equinox.static_field()
        mymodule = MyModule("normal", "static")
        leaves, treedef = jtu.tree_flatten(mymodule)
        assert leaves == ["normal"]
        assert "static" in str(treedef)
        ```
    In practice this should rarely be used; it is usually preferential to just filter
    out each field with `eqx.filter` whenever you need to select only some fields.
    **Arguments:**
    - `**kwargs`: If any are passed then they are passed on to `datacalss.field`.
        (Recall that Equinox uses dataclasses for its modules.)
    """
    try:
        metadata = dict(kwargs["metadata"])
    except KeyError:
        metadata = kwargs["metadata"] = {}
    if "static" in metadata:
        raise ValueError("Cannot use metadata with `static` already set.")
    metadata["static"] = True
    return field(**kwargs)

def _identity(x):
    return x

class MLP(nn.Module):
    """Standard Multi-Layer Perceptron; also known as a feed-forward network."""

    layers: List[nn.Linear]
    activation: Callable
    final_activation: Callable
    in_size: int = static_field()
    out_size: int = static_field()
    width_size: int = static_field()
    depth: int = static_field()

    def __init__(
        self,
        in_size: int,
        out_size: int,
        width_size: int,
        depth: int,
        activation: Callable = nn.functional.relu,
        final_activation: Callable = _identity,
        **kwargs,
    ):
        """**Arguments**:
        - `in_size`: The size of the input layer.
        - `out_size`: The size of the output layer.
        - `width_size`: The size of each hidden layer.
        - `depth`: The number of hidden layers.
        - `activation`: The activation function after each hidden layer. Defaults to
            ReLU.
        - `final_activation`: The activation function after the output layer. Defaults
            to the identity.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """

        super().__init__(**kwargs)
        layers = []
        if depth == 0:
            layers.append(nn.Linear(in_size, out_size))
        else:
            layers.append(nn.Linear(in_size, width_size))
            for i in range(depth - 1):
                layers.append(nn.Linear(width_size, width_size))
            layers.append(nn.Linear(width_size, out_size))
        self.layers = layers
        self.in_size = in_size
        self.out_size = out_size
        self.width_size = width_size
        self.depth = depth
        self.activation = activation
        self.final_activation = final_activation

    def __call__(
        self, x, *, key: Optional["jax.random.PRNGKey"] = None
    ):
        """**Arguments:**
        - `x`: A JAX array with shape `(in_size,)`.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)
        **Returns:**
        A JAX array with shape `(out_size,)`.
        """
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        x = self.layers[-1](x)
        x = self.final_activation(x)
        return x



class MixerBlock(torch.Module):
    patch_mixer: MLP
    hidden_mixer: MLP
    norm1: torch.nn.LayerNorm
    norm2: torch.nn.LayerNorm

    def __init__(
        self, num_patches, hidden_size, mix_patch_size, mix_hidden_size, *, key
    ):
        self.patch_mixer = MLP(
            num_patches, num_patches, mix_patch_size, depth=1
        )
        self.hidden_mixer = MLP(
            hidden_size, hidden_size, mix_hidden_size, depth=1
        )
        self.norm1 = torch.nn.LayerNorm((hidden_size, num_patches))
        self.norm2 = torch.nn.LayerNorm((num_patches, hidden_size))

    def __call__(self, y):
        y = y + vmap(self.patch_mixer)(self.norm1(y))
        y = einops.rearrange(y, "c p -> p c")
        y = y + vmap(self.hidden_mixer)(self.norm2(y))
        y = einops.rearrange(y, "p c -> c p")
        return y


class Mixer2d(torch.Module):
    conv_in: torch.nn.Conv2d
    conv_out: torch.nn.ConvTranspose2d
    blocks: list
    norm: torch.nn.LayerNorm
    t1: float

    def __init__(
        self,
        img_size,
        patch_size,
        hidden_size,
        mix_patch_size,
        mix_hidden_size,
        num_blocks,
        t1,
        *,
        key,
    ):
        input_size, height, width = img_size
        assert (height % patch_size) == 0
        assert (width % patch_size) == 0
        num_patches = (height // patch_size) * (width // patch_size)

        self.conv_in = torch.nn.Conv2d(
            input_size + 1, hidden_size, patch_size, stride=patch_size
        )
        self.conv_out = torch.nn.ConvTranspose2d(
            hidden_size, input_size, patch_size, stride=patch_size
        )
        self.blocks = [
            MixerBlock(
                num_patches, hidden_size, mix_patch_size, mix_hidden_size
            )
            for i in range(2 + num_blocks)
        ]
        self.norm = torch.nn.LayerNorm((hidden_size, num_patches))
        self.t1 = t1

    def __call__(self, t, y):
        t = t / self.t1
        _, height, width = y.shape
        t = einops.repeat(t, "-> 1 h w", h=height, w=width)
        y = np.concatenate([y, t])
        y = self.conv_in(y)
        _, patch_height, patch_width = y.shape
        y = einops.rearrange(y, "c h w -> c (h w)")
        for block in self.blocks:
            y = block(y)
        y = self.norm(y)
        y = einops.rearrange(y, "c (h w) -> c h w", h=patch_height, w=patch_width)
        return self.conv_out(y)

def single_loss_fn(model, weight, int_beta, data, t, key):
    mean = data * np.exp(-0.5 * int_beta(t))
    var = np.maximum(1 - np.exp(-int_beta(t)), 1e-5)
    std = np.sqrt(var)
    noise = torch.randn_like(data)
    y = mean + std * noise
    pred = model(t, y)
    return weight(t) * np.mean((pred + noise / std) ** 2)

def batch_loss_fn(model, weight, int_beta, data, t1, key):
    batch_size = data.shape[0]
    # Low-discrepancy sampling over t to reduce variance
    minval=0
    maxval =  t1/batch_size
    t = (minval - maxval) * torch.rand((batch_size,))
    t = t + (t1 / batch_size) * np.arange(batch_size)
    loss_fn = ft.partial(single_loss_fn, model, weight, int_beta)
    loss_fn = vmap(loss_fn)
    return np.mean(loss_fn(data, t))


def single_sample_fn(model, int_beta, data_shape, dt0, t1, key):
    def drift(t, y, args):
        _, beta = jvp(int_beta, torch.tensor(t,), torch.tensor(np.ones_like(t),))
        return -0.5 * beta * (y + model(t, y))

    term = dfx.ODETerm(drift)
    solver = dfx.Tsit5()
    t0 = 0
    y1 = jr.normal(key, data_shape)
    # reverse time, solve from t1 to t0
    sol = dfx.diffeqsolve(term, solver, t1, t0, -dt0, y1, adjoint=dfx.NoAdjoint())
    return sol.ys[0]


def mnist():
    filename = "train-images-idx3-ubyte.gz"
    url_dir = "https://storage.googleapis.com/cvdf-datasets/mnist"
    target_dir = os.getcwd() + "/data/mnist"
    url = f"{url_dir}/{filename}"
    target = f"{target_dir}/{filename}"

    if not os.path.exists(target):
        os.makedirs(target_dir, exist_ok=True)
        urllib.request.urlretrieve(url, target)
        print(f"Downloaded {url} to {target}")

    with gzip.open(target, "rb") as fh:
        _, batch, rows, cols = struct.unpack(">IIII", fh.read(16))
        shape = (batch, 1, rows, cols)
        return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(shape)

def dataloader(data, batch_size, *, key):
    dataset_size = data.shape[0]
    indices = np.arange(dataset_size)
    while True:
        perm = jr.permutation(key, indices)
        (key,) = jr.split(key, 1)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield data[batch_perm]
            start = end
            end = start + batch_size

@torch.jit.script
def make_step(model, weight, int_beta, data, t1, key, opt_state, opt_update):
    loss_fn = torch.filter_value_and_grad(batch_loss_fn)
    loss, grads = loss_fn(model, weight, int_beta, data, t1, key)
    updates, opt_state = opt_update(grads, opt_state)
    model = torch.apply_updates(model, updates)
    key = jr.split(key, 1)[0]
    return loss, model, key, opt_state

def main(
    # Model hyperparameters
    patch_size=4,
    hidden_size=64,
    mix_patch_size=512,
    mix_hidden_size=512,
    num_blocks=4,
    t1=10.0,
    # Optimisation hyperparameters
    num_steps=1_000_000,
    lr=3e-4,
    batch_size=256,
    print_every=10_000,
    # Sampling hyperparameters
    dt0=0.1,
    sample_size=10,
    # Seed
    seed=5678,
):
    data = mnist()
    data_mean = np.mean(data)
    data_std = np.std(data)
    data_max = np.max(data)
    data_min = np.min(data)
    data_shape = data.shape[1:]
    data = (data - data_mean) / data_std

    model = Mixer2d(
        data_shape,
        patch_size,
        hidden_size,
        mix_patch_size,
        mix_hidden_size,
        num_blocks,
        t1
    )
    int_beta = lambda t: t  # Try experimenting with other options here!
    weight = lambda t: 1 - np.exp(
        -int_beta(t)
    )  # Just chosen to upweight the region near t=0.

    opt = AdaBelief(lr=lr)
    # Optax will update the floating-point JAX arrays in the model.
    opt_state = opt.init(torch.filter(model, torch.is_inexact_array))

    total_value = 0
    total_size = 0
    for step, data in zip(
        range(num_steps), dataloader(data, batch_size)
    ):
        value, model, train_key, opt_state = make_step(
            model, weight, int_beta, data, t1, train_key, opt_state, opt.update
        )
        total_value += value.item()
        total_size += 1
        if (step % print_every) == 0 or step == num_steps - 1:
            print(f"Step={step} Loss={total_value / total_size}")
            total_value = 0
            total_size = 0

    sample_key = jr.split(sample_key, sample_size**2)
    sample_fn = ft.partial(single_sample_fn, model, int_beta, data_shape, dt0, t1)
    sample = vmap(sample_fn)(sample_key)
    sample = data_mean + data_std * sample
    sample = np.clip(sample, data_min, data_max)
    sample = einops.rearrange(
        sample, "(n1 n2) 1 h w -> (n1 h) (n2 w)", n1=sample_size, n2=sample_size
    )
    plt.imshow(sample, cmap="Greys")
    plt.axis("off")
    plt.tight_layout()
    plt.show()