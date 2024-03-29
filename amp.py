"""
amp api.
"""

import mindspore
from mindspore import nn
from mindspore import ops
from mindspore import Tensor, Parameter, context, ms_class
import mindspore.common.dtype as ms_type


class BMM(nn.Cell):
    def __init__(self):
        super().__init__()
        self.bmm = ops.BatchMatMul()

    def construct(self, x, y):
        return self.bmm(x, y)


amp_white_list = (
    nn.Dense,
    BMM,
)

amp_black_list = (nn.BatchNorm1d, nn.BatchNorm2d)


class _OutputTo32(nn.Cell):
    """Wrap cell for amp. Cast network output back to float32"""

    def __init__(self, op):
        super().__init__(auto_prefix=False)
        self._op = op

    def construct(self, *x):
        return ops.cast(self._op(*x), ms_type.float32)


class _OutputTo16(nn.Cell):
    """Wrap cell for amp. Cast network output back to float32"""

    def __init__(self, op):
        super().__init__(auto_prefix=False)
        self._op = op

    def construct(self, *x):
        return ops.cast(self._op(*x), ms_type.float16)


def auto_mixed_precision(network, amp_level='O1'):
    """auto mixed precision cast."""
    if amp_level == 'O0':
        if mindspore.get_context('device_target') == 'Ascend':
            print(
                'Model on Ascend must use auto mixed precision, the "amp_level" will be set to "O1".'
            )
    elif amp_level == 'O1':
        auto_white_list(network)
    elif amp_level == 'O2':
        auto_black_list(network)
    elif amp_level == 'O3':
        network.to_float(ms_type.float16)
    else:
        raise ValueError(f"the amp_level '{amp_level}' is not supported.")
    return network


def auto_white_list(network, white_list=None):
    """auto cast based on white list"""
    if white_list is None:
        white_list = amp_white_list
    cells = network.name_cells()
    change = False
    for name in cells:
        subcell = cells[name]
        if subcell == network:
            continue
        if isinstance(subcell, white_list):
            network._cells[name] = _OutputTo32(subcell.to_float(
                ms_type.float16))
            change = True
        else:
            auto_white_list(subcell, white_list)

    if isinstance(network, nn.SequentialCell) and change:
        network.cell_list = list(network.cells())


def auto_black_list(network, black_list=None):
    """auto cast based on black list"""
    if black_list is None:
        black_list = amp_black_list
    network.to_float(ms_type.float16)
    cells = network.name_cells()
    change = False
    for name in cells:
        subcell = cells[name]
        if subcell == network:
            continue
        if isinstance(subcell, black_list):
            network._cells[name] = _OutputTo16(subcell.to_float(
                ms_type.float32))
            change = True
        else:
            auto_black_list(subcell, black_list)

    if isinstance(network, nn.SequentialCell) and change:
        network.cell_list = list(network.cells())


# For Loss Scaler
ascend_target = (context.get_context("device_target") == "Ascend")
gpu_target = (context.get_context("device_target") == "GPU")
reciprocal = ops.Reciprocal()

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


def grad_unscale(scale, grad):
    """grad unscale."""
    return grad * reciprocal(scale).astype(grad.dtype)


def grad_scale(scale, grad):
    """grad scale."""
    return grad * scale.astype(grad.dtype)


def is_finite(inputs):
    """whether input tensor is finite."""
    if gpu_target:
        return gpu_float_status(inputs)[0] == 0
    status = ops.isfinite(inputs)
    return status.all()


def all_finite(inputs):
    """whether all inputs tensor are finite."""
    if ascend_target:
        status = ops.depend(status, inputs)
        get_status = npu_get_float_status(status)
        status = ops.depend(status, get_status)
        status_finite = status.sum() == 0
        _ = npu_clear_float_status(status)
        return status_finite
    outputs = hyper_map(partial(is_finite), inputs)
    return ops.stack(outputs).all()


@ms_class
class LossScaler:
    """
    Basic LossScaler.
    """

    def __init__(self, scale_value):
        super().__init__()
        self.scale_value = Parameter(Tensor(scale_value, dtype=ms_type.float32),
                                     name="scale_value")
        self.counter = Parameter(Tensor(0, dtype=ms_type.int32), name="counter")

    def scale(self, inputs):
        """scale inputs tensor."""
        raise NotImplementedError

    def unscale(self, inputs):
        """unscale inputs tensor."""
        raise NotImplementedError

    def adjust(self, grads_finite):
        """adjust scale value."""
        raise NotImplementedError


class NoLossScaler(LossScaler):
    """
    No LossScaler
    """

    def __init__(self):
        super().__init__(1)

    def scale(self, inputs):
        return inputs

    def unscale(self, inputs):
        return inputs

    def adjust(self, grads_finite):
        return True


class StaticLossScaler(LossScaler):
    """
    Static LossScaler.
    """

    def scale(self, inputs):
        return hyper_map(partial(grad_scale, self.scale_value), inputs)

    def unscale(self, inputs):
        return hyper_map(partial(grad_unscale, self.scale_value), inputs)

    def adjust(self, grads_finite):
        return True


class DynamicLossScaler(LossScaler):
    """
    Dynamic LossScaler
    """

    def __init__(self, scale_value, scale_factor, scale_window):
        super().__init__(scale_value)
        self.scale_factor = scale_factor
        self.scale_window = scale_window

    def scale(self, inputs):
        return hyper_map(partial(grad_scale, self.scale_value), inputs)

    def unscale(self, inputs):
        return hyper_map(partial(grad_unscale, self.scale_value), inputs)

    def adjust(self, grads_finite):
        one = ops.ones((), self.scale_value.dtype)
        scale_mul_factor = self.scale_value * self.scale_factor
        scale_value = ops.select(
            grads_finite,
            # When grads are finite increase loss scale periodically.
            ops.select(
                self.counter == (self.scale_window - 1),
                ops.select(is_finite(scale_mul_factor), scale_mul_factor,
                           self.scale_value), self.scale_value),
            # If grads are non-finite reduce loss scale.
            ops.maximum(one, self.scale_value / self.scale_factor))
        ops.assign(self.scale_value, scale_value)

        counter = ((self.counter + 1) % self.scale_window) * grads_finite
        ops.assign(self.counter, counter)
        return True
