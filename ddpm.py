import os
import random
import math
from multiprocessing import cpu_count
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np

import mindspore
import mindspore.dataset.vision.c_transforms as transformer
import mindspore.dataset.vision.py_transforms as py_transformer
import mindspore.common.dtype as ms_type

from mindspore import ops, nn, ms_function, save_checkpoint, load_checkpoint,\
    load_param_into_net, set_auto_parallel_context, Tensor, Parameter, context, ms_class
from mindspore.ops import constexpr, stop_gradient, GradOperation
from mindspore.ops.operations.image_ops import ResizeBilinearV2, ResizeLinear1D
from mindspore.communication import init, get_rank
from mindspore.dataset import VisionBaseDataset, GeneratorDataset, MindDataset, ImageFolderDataset
from mindspore.common.initializer import initializer, HeUniform, Uniform, \
    Normal, _calculate_fan_in_and_fan_out
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore.parallel._utils import _get_device_num, _get_gradients_mean

from ema import EMA
from amp import DynamicLossScaler, StaticLossScaler, NoLossScaler, auto_mixed_precision, all_finite, BMM


def rsqrt(x):
    rsqrt_op = _get_cache_prim(ops.Rsqrt)()
    return rsqrt_op(x)


def randn_like(x, dtype=None):
    if dtype is None:
        dtype = x.dtype
    normal = _get_cache_prim(ops.StandardNormal)()
    return normal(x.shape).astype(dtype)


def rearrange(head, inputs):
    b, hc, x, y = inputs.shape
    c = hc // head
    return inputs.reshape((b, head, c, x*y))


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def save_images(all_images_list, path):
    if not os.path.exists(path):
        os.mkdir(path)
    for i, image in enumerate(all_images_list):
        image = image[0]
        image = image * 255 + 0.5
        image = np.clip(image, 0, 255).astype(np.uint8)
        image = image.transpose((1, 2, 0))
        im = Image.fromarray(image)
        save_path = os.path.join(path, f'{i}-img.png')
        im.save(save_path)


def calcu_output_size(input_tensor, scale_factor, mode):
    """calculate output size for up_sample"""
    input_size = input_tensor.shape
    if isinstance(scale_factor, tuple) and len(scale_factor) != len(input_size[2:]):
        raise ValueError(f"the number of 'scale_factor' must match to inputs.shape[2:], "
                         f"but get scale_factor={scale_factor}, inputs.shape[2:]={input_size[2:]}")

    ret = ()
    for i in range(len(input_size[2:])):
        if isinstance(scale_factor, float):
            out_i = int(scale_factor * input_size[i + 2])
        else:
            out_i = int(scale_factor[i] * input_size[i + 2])
        ret = ret + (out_i,)
    if mode == "nearest":
        return ret
    return Tensor(ret)


class Residual(nn.Cell):
    """残差块"""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def construct(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


@constexpr
def _check_scale_factor(shape, scale_factor):
    if isinstance(scale_factor, tuple) and len(scale_factor) != len(shape[2:]):
        raise ValueError(f"the number of 'scale_fator' must match to inputs.shape[2:], "
                         f"but get scale_factor={scale_factor}, inputs.shape[2:]={shape[2:]}")


def _interpolate_output_shape(shape, scales, sizes, mode):
    """calculate output shape"""
    if sizes is not None:
        if mode == "nearest":
            return sizes
        return Tensor(sizes)

    ret = ()
    for i in range(len(shape[2:])):
        if isinstance(scales, float):
            out_i = int(scales * shape[i+2])
        else:
            out_i = int(scales[i] * shape[i+2])
        ret = ret + (out_i,)
    if mode == "nearest":
        return ret
    return Tensor(ret)


class UpSample(nn.Cell):
    def __init__(self, size=None, scale_factor=None,
                 mode: str = 'nearest', align_corners=False):
        super().__init__()
        if mode not in ['nearest', 'linear', 'bilinear']:
            raise ValueError(f'do not support mode :{mode}.')
        if size and scale_factor:
            raise ValueError(f"can not set 'size' and 'scale_fator' at the same time.")
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners

    def construct(self, inputs):
        inputs_shape = inputs.shape
        _check_scale_factor(inputs_shape, self.scale_factor)
        sizes = _interpolate_output_shape(inputs_shape, self.scale_factor, self.size, self.mode)
        if self.mode == 'nearest':
            interpolate = _get_cache_prim(ops.ResizeNearestNeighbor)(sizes, self.align_corners)
            return interpolate(inputs)
        elif self.mode == 'linear':
            interpolate = _get_cache_prim(ResizeLinear1D)('align_corners' if self.align_corners else 'half_pixel')
            return interpolate(inputs, sizes)
        elif self.mode == 'bilinear':
            interpolate = _get_cache_prim(ResizeBilinearV2)(self.align_corners, True if self.align_corners==False else False)
            return interpolate(inputs, sizes)
        return inputs


class Conv2d(nn.Conv2d):
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 pad_mode='same',
                 padding=0,
                 dilation=1,
                 group=1,
                 has_bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         pad_mode, padding, dilation, group, has_bias,
                         weight_init='normal', bias_init='zeros')
        self.reset_parameters()

    def reset_parameters(self):
        # 同步pytorch, weight参数采用了He-uniform初始化策略
        self.weight.set_data(initializer(HeUniform(math.sqrt(5)), self.weight.shape))
        if self.has_bias:
            fan_in, _ = _calculate_fan_in_and_fan_out(self.weight.shape)
            bound = 1 / math.sqrt(fan_in)
            self.bias.set_data(initializer(Uniform(bound), [self.out_channels]))


class Dense(nn.Dense):
    def __init__(self, in_channels, out_channels, has_bias=True, activation=None):
        super().__init__(in_channels, out_channels, weight_init='normal', bias_init='zeros', has_bias=has_bias, activation=activation)
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.set_data(initializer(HeUniform(math.sqrt(5)), self.weight.shape))
        if self.has_bias:
            fan_in, _ = _calculate_fan_in_and_fan_out(self.weight.shape)
            bound = 1 / math.sqrt(fan_in)
            self.bias.set_data(initializer(Uniform(bound), [self.out_channels]))


def up_sample(dim, dim_out=None):
    return nn.SequentialCell(
        UpSample(scale_factor=2, mode='nearest'),
        Conv2d(dim, default(dim_out, dim), 3, padding=1, pad_mode='pad')
    )


def down_sample(dim, dim_out=None):
    return Conv2d(dim, default(dim_out, dim), 4, 2, 'pad', 1)


class WeightStandardizedConv2d(Conv2d):
    def construct(self, x):
        eps = 1e-5 if x.dtype == ms_type.float32 else 1e-3
        weight = self.weight
        mean = weight.mean((1, 2, 3), keep_dims=True)
        var = weight.var((1, 2, 3), keepdims=True)
        normalized_weight = (weight - mean) * rsqrt(var + eps)

        conv_ops = self.conv2d(x, normalized_weight.astype(x.dtype))
        if self.has_bias:
            conv_ops = self.bias_add(conv_ops, self.bias)
        return conv_ops


class LayerNorm(nn.Cell):
    def __init__(self, dim):
        super().__init__()
        self.params = Parameter(initializer('ones', (1, dim, 1, 1)), name='g')

    def construct(self, x):
        eps = 1e-5 if x.dtype == ms_type.float32 else 1e-3
        var = x.var(axis=1, keepdims=True)
        mean = x.mean(axis=1, keep_dims=True)
        return (x - mean) * rsqrt(var + eps) * self.params


class PreNorm(nn.Cell):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def construct(self, x):
        x = self.norm(x)
        return self.fn(x)


class SinusoidalPosEmb(nn.Cell):
    def __init__(self, dim):
        super().__init__()
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = np.exp(np.arange(half_dim) * - emb)
        self.emb = Tensor(emb, mindspore.float32)
        self.Concat = _get_cache_prim(ops.Concat)(-1)

    def construct(self, x):
        emb = x[:, None] * self.emb[None, :]
        emb = self.Concat((ops.sin(emb), ops.cos(emb)))
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Cell):
    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = Parameter(initializer(Normal(1.0), (half_dim,)), name='weights',
                                 requires_grad=not is_random)
        self.Concat = _get_cache_prim(ops.Concat)(-1)

    def construct(self, x):
        x = x.expand_dims(1)
        freqs = x * self.weights.expand_dims(0) * 2 * Tensor(math.pi, mindspore.float32)
        fouriered = self.Concat((ops.sin(freqs), ops.cos(freqs)))
        fouriered = self.Concat((x, fouriered))
        return fouriered


class Block(nn.Cell):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1, pad_mode='pad')
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def construct(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Cell):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.SequentialCell(
            nn.SiLU(),
            Dense(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = Conv2d(dim, dim_out, 1) if dim != dim_out else _get_cache_prim(ops.Identity)()
        self.split = _get_cache_prim(ops.Split)(axis=1, output_num=2)

    def construct(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = time_emb.expand_dims(-1).expand_dims(-1)
            scale_shift = self.split(time_emb)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class LinearAttention(nn.Cell):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = Conv2d(dim, hidden_dim * 3, 1, pad_mode='valid')

        self.to_out = nn.SequentialCell(
            Conv2d(hidden_dim, dim, 1, pad_mode='valid', has_bias=True),
            LayerNorm(dim)
        )

        self.map = _get_cache_prim(ops.Map)()
        self.partial = _get_cache_prim(ops.Partial)()
        self.bmm = _get_cache_prim(BMM)()
        self.split = _get_cache_prim(ops.Split)(axis=1, output_num=3)
        self.softmax1 = _get_cache_prim(ops.Softmax)(-1)
        self.softmax2 = _get_cache_prim(ops.Softmax)(-2)

    def construct(self, x):
        b, c, h, w = x.shape
        qkv = self.split(self.to_qkv(x))
        q, k, v = self.map(self.partial(rearrange, self.heads), qkv)

        q = self.softmax2(q)
        k = self.softmax1(k)

        q = q * self.scale
        v = v / (h * w)

        bmm_result = self.bmm(k, v.swapaxes(2, 3))
        out = self.bmm(bmm_result.swapaxes(2, 3), q)

        out = out.reshape((b, -1, h, w))
        return self.to_out(out)


class Attention(nn.Cell):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = _get_cache_prim(Conv2d)(dim, hidden_dim * 3, 1, pad_mode='valid', has_bias=False)
        self.to_out = _get_cache_prim(Conv2d)(hidden_dim, dim, 1, pad_mode='valid', has_bias=True)
        self.map = ops.Map()
        self.partial = ops.Partial()
        self.bmm = BMM()
        self.split = ops.Split(axis=1, output_num=3)
        self.softmax = ops.Softmax(-1)

    def construct(self, x):
        b, c, h, w = x.shape
        qkv = self.split(self.to_qkv(x))
        q, k, v = self.map(self.partial(rearrange, self.heads), qkv)
        q = q * self.scale
        sim = self.bmm(q.swapaxes(2, 3), k)
        attn = self.softmax(sim)
        out = self.bmm(attn, v.swapaxes(2, 3))
        out = out.swapaxes(-1, -2).reshape((b, -1, h, w))
        return self.to_out(out)


grad_func = GradOperation(True, False, False)
grad_cell = GradOperation(False, True, False)
gpu_target = (context.get_context("device_target") == "GPU")
context.set_context(mode=context.GRAPH_MODE)
ascend_target = (context.get_context("device_target") == "Ascend")
gpu_float_status = ops.FloatStatus()
npu_alloc_float_status = ops.NPUAllocFloatStatus()
npu_clear_float_status = ops.NPUClearFloatStatus()
npu_get_float_status = ops.NPUGetFloatStatus()

if ascend_target:
    status = npu_alloc_float_status()
    _ = npu_clear_float_status(status)
else:
    status = None

hyper_map = ops.HyperMap()
partial = ops.Partial()


def is_finite(inputs):
    """whether input tensor is finite."""
    if gpu_target:
        return gpu_float_status(inputs)[0] == 0
    state = ops.isfinite(inputs)
    return state.all()


def value_and_grad(fn, pos=None, params=None, has_aux=False):
    if params is None:
        grad_ = grad_func
    else:
        grad_ = grad_cell

    def fn_aux(*args):
        outputs = fn(*args)
        no_grad_outputs = (outputs[0],)
        for out in outputs[1:]:
            no_grad_outputs += (stop_gradient(out),)
        return no_grad_outputs

    if has_aux:
        fn_ = fn_aux
    else:
        fn_ = fn

    def value_and_grad_f(*args):
        values = fn_(*args)
        if params is None:
            grads = grad_(fn_)(*args)
        else:
            grads = grad_(fn_, params)(*args)
        return values, grads
    return value_and_grad_f


def has_int_square_root(num):
    return (math.sqrt(num) ** 2) == num


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def create_dataset(folder, image_size, extensions=None, augment_horizontal_flip=False,
                   batch_size=1, shuffle=True, num_workers=cpu_count()):
    extensions = ['.jpg', '.jpeg', '.png', '.tiff'] if not extensions else extensions
    dataset = ImageFolderDataset(folder, num_parallel_workers=num_workers, shuffle=False,
                                 extensions=extensions, decode=True)

    transformers = [
        transformer.Resize([image_size, image_size], transformer.Inter.BILINEAR),
        py_transformer.ToTensor()
    ]
    if augment_horizontal_flip:
        transformers.append(transformer.RandomHorizontalFlip())
    dataset = dataset.project('image')
    dataset = dataset.map(transformers, 'image')
    if shuffle:
        dataset = dataset.shuffle(dataset.get_dataset_size())
    dataset = dataset.batch(batch_size, drop_remainder=False)
    return dataset


@ms_class
class Accumulator:
    def __init__(self, optimizer, accumulate_step, total_step=None, clip_norm=1.0):
        self.optimizer = optimizer
        self.clip_norm = clip_norm
        self.inner_grads = optimizer.parameters.clone(prefix="accumulate_", init='zeros')
        self.zeros = optimizer.parameters.clone(prefix="zeros_", init='zeros')
        self.counter = Parameter(Tensor(1, mindspore.int32), 'counter_')
        assert accumulate_step > 0
        self.accumulate_step = accumulate_step
        if total_step is not None:
            assert total_step > accumulate_step and total_step > 0
        self.total_step = total_step
        self.map = ops.Map()
        self.partial = ops.Partial()

    def __call__(self, grads):
        success = self.map(self.partial(ops.assign_add), self.inner_grads, grads)
        if self.counter % self.accumulate_step == 0:
            clip_grads = ops.clip_by_global_norm(self.inner_grads, self.clip_norm)
            success = ops.depend(success, self.optimizer(clip_grads))
            success = ops.depend(success, self.map(self.partial(ops.assign), self.inner_grads, self.zeros))

        success = ops.depend(success, ops.assign_add(self.counter, Tensor(1, mindspore.int32)))

        return success


class Unet(nn.Cell):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        self_condition=False,
        resnet_block_groups=8,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16
    ):
        super().__init__()

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = Conv2d(input_channels,
                                init_dim,
                                7,
                                padding=3,
                                pad_mode='pad',
                                has_bias=True)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        import functools
        block_klass = functools.partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        time_dim = dim * 4
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sin_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim,
                                                          random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sin_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.SequentialCell(
            sin_pos_emb,
            Dense(fourier_dim, time_dim),
            nn.GELU(False),
            Dense(time_dim, time_dim)
        )

        # layers
        self.downs = nn.CellList([])
        self.ups = nn.CellList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.CellList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                down_sample(dim_in, dim_out) if not is_last else
                Conv2d(dim_in, dim_out, 3, padding=1, pad_mode='pad', has_bias=True)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.CellList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                up_sample(dim_out, dim_in) if not is_last else
                Conv2d(dim_out, dim_in, 3, padding=1, pad_mode='pad')
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_res_block.recompute()
        self.final_conv = Conv2d(dim, self.out_dim, 1, pad_mode='valid', has_bias=True)
        self.final_conv.recompute()
        # self.Concat = ops.concat(axis=1)
        # self.Concat.recompute()

    def construct(self, x, time, x_self_cond):
        if self.self_condition:
            if x_self_cond is None:
                x_self_cond = ops.zeros_like(x)
            x = ops.concat((x_self_cond, x), 1)
        x = self.init_conv(x)
        r = x.copy()
        t = self.time_mlp(time)
        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        len_h = len(h) - 1
        for block1, block2, attn, upsample in self.ups:
            x = ops.concat((x, h[len_h]), 1)
            len_h -= 1
            x = block1(x, t)

            x = ops.concat((x, h[len_h]), 1)
            len_h -= 1
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = ops.concat((x, r), 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)


# Gaussian diffusion class
def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


def extract(a, t, x_shape):
    return a[t, None, None, None]


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return np.linspace(beta_start, beta_end, timesteps).astype(np.float32)


def cosine_beta_schedule(time_steps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = time_steps + 1
    x = np.linspace(0, time_steps, steps).astype(np.float32)
    alphas_cumprod = np.cos(((x / time_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999)


def generate_noise(x, data_type=None):
    if data_type is None:
        data_type = x.dtype
    normal = _get_cache_prim(ops.StandardNormal)()
    return normal(x.shape).astype(data_type)


def generate_t_tensor(t, data_type=mindspore.float32):
    return Tensor(t, data_type)


class GaussianDiffusion(nn.Cell):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps=1000,
        sampling_timesteps=None,
        loss_type='l1',
        objective='pred_noise',
        beta_schedule='cosine',
        # p2 loss weight, - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_gamma=0.,
        p2_loss_weight_k=1,
        ddim_sampling_eta=1.,
        auto_normalize=True
    ):
        super().__init__()
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not model.random_or_learned_sinusoidal_cond

        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        self.image_size = image_size

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, \
            'objective must be either pred_noise (predict noise)' \
            ' or pred_x0 (predict image start) or pred_v '

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.pad(alphas_cumprod[:-1], (1, 0), constant_values=1)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters
        # default num sampling timesteps to number of timesteps at training
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32
        self.betas = betas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = Tensor(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = Tensor(np.sqrt(1. - alphas_cumprod))
        self.log_one_minus_alphas_cumprod = Tensor(np.log(1. - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod = Tensor(np.sqrt(1. / alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod = Tensor(np.sqrt(1. / alphas_cumprod - 1))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        self.posterior_variance = Tensor(posterior_variance)

        self.posterior_log_variance_clipped = Tensor(
            np.log(np.clip(posterior_variance, 1e-20, None)))
        self.posterior_mean_coef1 = Tensor(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.posterior_mean_coef2 = Tensor(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        p2_loss_weight = (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod))\
                          ** - p2_loss_weight_gamma
        self.p2_loss_weight = Tensor(p2_loss_weight)

        if self.loss_type == 'l1':
            self.loss_fn = nn.L1Loss('none')
        elif self.loss_type == 'l2':
            self.loss_fn = nn.MSELoss('none')
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped,
                                                 t,
                                                 x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    @ms_function
    def model_predictions(self, x, t, x_self_cond=None, clip_x_start=False):
        model_output = self.model(x, t, x_self_cond)

        def maybe_clip(x, clip):
            if clip:
                return x.clip(-1., 1.)
            return x

        if self.objective == 'pred_noise':
            predict_noise = model_output
            x_start = self.predict_start_from_noise(x, t, predict_noise)
            x_start = maybe_clip(x_start, clip_x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start, clip_x_start)
            predict_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start, clip_x_start)
            predict_noise = self.predict_noise_from_start(x, t, x_start)
        else:
            predict_noise = model_output
            x_start = model_output

        return predict_noise, x_start

    def p_mean_variance(self, x, t, x_self_cond=None, clip_denoised=True):
        _, x_start = self.model_predictions(x, t, x_self_cond)

        if clip_denoised:
            x_start.clip(-1., 1.)

        model_mean, post_variance, post_log_variance = self.q_posterior(
            x_start=x_start, x_t=x, t=t)
        return model_mean, post_variance, post_log_variance, x_start

    @ms_function
    def p_sample(self, x, t, x_self_cond=None, clip_denoise=True):
        batched_times = ops.ones((x.shape[0],), mindspore.int32) * t
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, t=batched_times, x_self_cond=x_self_cond, clip_denoised=clip_denoise)
        noise = randn_like(x) if t > 0 else ops.zeros_like(x)  # no noise if t == 0
        predict_img = model_mean + ops.exp(0.5 * model_log_variance) * noise
        return predict_img, x_start

    def p_sample_loop(self, shape):
        img = np.random.randn(*shape).astype(np.float32)
        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)),
                      desc='sampling loop time step',
                      total=self.num_timesteps):
            x_start = Tensor(x_start) if x_start is not None else x_start
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(Tensor(img), Tensor(t, mindspore.int32), self_cond)
            img, x_start = img.asnumpy(), x_start.asnumpy()

        img = self.unnormalize(img)
        return img

    def ddim_sample(self, shape, clip_denoise=True):
        batch = shape[0]
        total_timesteps, sampling_timesteps, = self.num_timesteps, self.sampling_timesteps
        eta, objective = self.ddim_sampling_eta, self.objective

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = np.linspace(-1, total_timesteps - 1, sampling_timesteps + 1).astype(np.int32)
        # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        times = list(reversed(times.tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        img = np.random.randn(*shape).astype(np.float32)
        x_start = None

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            # time_cond = ops.fill(mindspore.int32, (batch,), time)
            time_cond = np.full((batch,), time).astype(np.int32)
            x_start = Tensor(x_start) if x_start is not None else x_start
            self_cond = x_start if self.self_condition else None
            predict_noise, x_start, *_ = self.model_predictions(Tensor(img, mindspore.float32),
                                                                Tensor(time_cond),
                                                                self_cond,
                                                                clip_denoise)
            predict_noise, x_start = predict_noise.asnumpy(), x_start.asnumpy()
            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * np.sqrt(((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)))
            c = np.sqrt(1 - alpha_next - sigma ** 2)

            noise = np.random.randn(*img.shape)

            img = x_start * np.sqrt(alpha_next) + c * predict_noise + sigma * noise

        img = self.unnormalize(img)

        return img

    def sample(self, batch_size=16):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        shape = (batch_size, channels, image_size, image_size)
        return sample_fn(shape)

    def interpolate(self, x1, x2, t=None, lam=0.5):
        b = x1.shape[0]
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = ops.stack([mindspore.Tensor(t)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, ops.fill(mindspore.int32, (b,), i))

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: generate_noise(x_start))
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise, random_cond):
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        if self.self_condition:
            if random_cond:
                _, x_self_cond = self.model_predictions(x, t)
                x_self_cond = ops.stop_gradient(x_self_cond)
            else:
                x_self_cond = ops.zeros_like(x)
        else:
            x_self_cond = ops.zeros_like(x)

        # predict and take gradient step
        model_out = self.model(x, t, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            target = noise

        loss = self.loss_fn(model_out, target)
        loss = loss.reshape(loss.shape[0], -1)
        loss = loss * extract(self.p2_loss_weight, t, loss.shape)
        return loss.mean()

    def construct(self, img, t, noise, random_cond):
        img = self.normalize(img)
        return self.p_losses(img, t, noise, random_cond)


class Trainer:
    """
    training DDPM
    """
    def __init__(
        self,
        diffusion_model,
        folder_or_dataset,
        *,
        train_batch_size=16,
        gradient_accumulate_every=1,
        augment_horizontal_flip=True,
        train_lr=1e-4,
        train_num_steps=100000,
        ema_update_every=10,
        ema_decay=0.995,
        adam_betas=(0.9, 0.99),
        save_and_sample_every=500,
        num_samples=5,
        results_folder='./results',
        dynamic_loss_scale=False,
        self_condition=True,
        use_static=True,
        akg=True,
        distributed=False,
        jit=True,
        amp_level="O1"
    ):
        super().__init__()
        self.amp_level = amp_level
        self.jit = jit
        self.self_condition = self_condition
        device_id = int(os.getenv('DEVICE_ID', "0"))
        mindspore.set_context(device_id=device_id)
        backend = mindspore.get_context('device_target')
        if jit and akg and backend != 'Ascend':
            mindspore.set_context(enable_graph_kernel=True, graph_kernel_flags="--opt_level=1", mode=context.GRAPH_MODE)

        # distributed training
        self.distributed = distributed
        if distributed:
            init()
            rank_id = get_rank()
            set_auto_parallel_context(parallel_mode=mindspore.ParallelMode.DATA_PARALLEL,
                                      gradients_mean=True)
        else:
            rank_id = 0

        self.is_main_process = True if rank_id == 0 else False
        if self.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)

            self.results_folder = Path(results_folder)
            self.results_folder.mkdir(exist_ok=True)

        square_info = 'number of samples must have an integer square root'
        assert has_int_square_root(num_samples), square_info

        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        # dataset and dataloader
        if isinstance(folder_or_dataset, str):
            self.dataset = create_dataset(folder_or_dataset, self.image_size,
                                          augment_horizontal_flip=augment_horizontal_flip,
                                          batch_size=train_batch_size, shuffle=True)
        elif isinstance(folder_or_dataset, (VisionBaseDataset, GeneratorDataset, MindDataset)):
            self.dataset = folder_or_dataset
        else:
            raise ValueError(f"the value of 'folder_or_dataset' should be a str or Dataset,"
                             f" but get {folder_or_dataset}.")

        dataset_size = self.dataset.get_dataset_size()
        print("training dataset size:", dataset_size)
        self.dataset = self.dataset.repeat(
            int(train_num_steps * gradient_accumulate_every // dataset_size) + 1)

        # optimizer
        self.opt = nn.Adam(diffusion_model.trainable_params(),
                           train_lr,
                           adam_betas[0],
                           adam_betas[1])

        # accumulator
        self.gradient_accumulate_every = gradient_accumulate_every
        self.accumulator = Accumulator(self.opt, gradient_accumulate_every)

        self.step = 0
        self.results_folder = results_folder
        self.use_static = use_static
        self.model = diffusion_model
        self.dynamic_loss_scale = dynamic_loss_scale

    def save(self, milestone):
        append_info = dict()
        append_info['batch_size'] = self.batch_size
        append_info['step'] = self.step
        save_path = str(self.results_folder) + f'/model-{milestone}.ckpt'
        save_checkpoint(self.model,
                        save_path,
                        append_dict=append_info)

    def load(self, load_path):
        param_dict = load_checkpoint(load_path)
        load_param_into_net(self.model, param_dict)

    def save_images(self, all_images_list, milestone):
        image_folder = str(self.results_folder) + f'/image-{milestone}'
        save_images(all_images_list, image_folder)

    def inference(self):
        batches = num_to_groups(self.num_samples, self.batch_size)
        all_images_list = list(map(lambda n: self.ema.online_model.sample(batch_size=n), batches))
        return all_images_list

    def train(self):
        model = self.model
        accumulator = self.accumulator
        grad_acc = self.gradient_accumulate_every
        self_condition = model.self_condition
        num_timesteps = model.num_timesteps

        # auto mixed precision
        model = auto_mixed_precision(model, self.amp_level)
        if self.amp_level != 'O0':
            if self.dynamic_loss_scale:
                loss_scaler = DynamicLossScaler(65536, 2, 1000)
            else:
                loss_scaler = StaticLossScaler(65536)
        else:
            loss_scaler = NoLossScaler()

        if self.distributed:
            mean = _get_gradients_mean()
            degree = _get_device_num()
            grad_reducer = nn.DistributedGradReducer(self.opt.parameters, mean, degree)
        else:
            grad_reducer = ops.identity

        def forward_fn(data, t, noise, self_cond):
            loss = model(data, t, noise, self_cond)
            loss = loss / grad_acc
            loss = loss_scaler.scale(loss)
            return loss

        grad_fn = value_and_grad(forward_fn, None, self.opt.parameters)

        def train_step(data, t, noise, self_cond):
            loss, grads = grad_fn(data, t, noise, self_cond)
            grads = grad_reducer(grads)
            status = all_finite(grads)
            if status:
                loss = loss_scaler.unscale(loss)
                grads = loss_scaler.unscale(grads)
                loss = ops.depend(loss, accumulator(grads))
                # grads = ops.clip_by_global_norm(grads, 1.0)
                # loss = ops.depend(loss, optimizer(grads))
            loss = ops.depend(loss, loss_scaler.adjust(status))
            return loss

        if self.jit:
            train_step = ms_function(train_step)

        data_iterator = self.dataset.create_tuple_iterator()
        with tqdm(initial = self.step, total = self.train_num_steps, disable = not self.is_main_process) as pbar:
            total_loss = 0.
            for (data,) in data_iterator:
                model.set_train()

                self_cond = random.random() < 0.5 if self_condition else False
                b = data.shape[0]

                # 采样时间
                t = Tensor(np.random.randint(0, num_timesteps, (b,)).astype(np.int32))
                noise = Tensor(np.random.randn(*data.shape), mindspore.float32)
                loss = train_step(data, t, noise, self_cond)
                total_loss += float(loss.asnumpy())

                self.step += 1
                if self.step % self.gradient_accumulate_every == 0:
                    if self.is_main_process:
                        self.ema.update()
                    pbar.set_description(f'loss: {total_loss:.4f}')
                    pbar.update(1)
                    total_loss = 0.

                if self.is_main_process:
                    accumulate_step = self.step // self.gradient_accumulate_every
                    accumulate_remain_step = self.step % self.gradient_accumulate_every
                    if accumulate_step != 0 and \
                        accumulate_step % self.save_and_sample_every == 0 and \
                        accumulate_remain_step == (self.gradient_accumulate_every - 1):

                        self.ema.set_train(False)
                        # model -> swap, ema -> model
                        self.ema.synchronize()

                        batches = num_to_groups(self.num_samples, self.batch_size)
                        all_images_list = list(map(lambda n: self.ema.online_model.sample(batch_size=n),
                                                   batches))
                        self.save_images(all_images_list, accumulate_step)
                        self.save(accumulate_step)
                        self.ema.desynchronize()

                if self.step >= self.gradient_accumulate_every * self.train_num_steps:
                    break

        print('training complete')
