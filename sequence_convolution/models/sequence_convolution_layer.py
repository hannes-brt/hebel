import numpy as np
from pycuda import gpuarray
from pycuda.curandom import rand as curand
from .. import pycuda_ops
from neural_nets.models import HiddenLayer
from neural_nets.pycuda_ops.elementwise import sign

class SequenceConvolutionLayer(HiddenLayer):
    n_parameters = 2

    def __init__(self, n_in, filter_width, n_filters, activation_function='sigmoid',
                 weights_scale=.01, W=None, b=None,
                 l1_penalty_weight=0., l2_penalty_weight=0.,
                 dtype=np.float32):
        if W is None:
            self.W = weights_scale * \
              curand((n_filters, 4*filter_width), dtype=dtype) \
              -.5 * weights_scale
        else:
            self.W = W

        if b is None:
            self.b = gpuarray.zeros((n_filters,), dtype)
        else:
            self.b = b

        assert self.W.shape == (n_filters, 4*filter_width)
        assert self.b.shape == (n_filters,)

        self.n_in = n_in
        self.filter_width = filter_width
        self.n_filters = n_filters
        self.n_units = n_filters * n_in

        self._set_activation_fct(activation_function)
        self.l1_penalty_weight = l1_penalty_weight
        self.l2_penalty_weight = l2_penalty_weight

        self.lr_multiplier = [1., 1.]

    def feed_forward(self, input, prediction=False):
        activations = \
            pycuda_ops.convolve_sequence(input, self.W, self.b)

        self.f(activations)
        return (activations,)

    def backprop(self, input, df_output, cache=None):
        if cache is None:
            activations = self.feed_forward(input)[0]
        else:
            activations = cache[0]

        df_activations = self.df(activations)
        delta = df_activations * df_output
        df_b = pycuda_ops.sum_delta(delta, self.n_filters)
        df_W = pycuda_ops.convolve_sequence_gradient(
            input, delta,
            self.filter_width, self.n_filters)

        # L1 weight decay
        if self.l1_penalty_weight:
            df_W -= self.l1_penalty_weight * sign(self.W)

        # L2 weight decay
        if self.l2_penalty_weight:
            df_W -= self.l2_penalty_weight * self.W

        return (df_W, df_b), None
