import numpy as np
from pycuda import gpuarray
from pycuda.curandom import rand as curand
from pycuda import cumath
from math import sqrt
from scikits.cuda import linalg
from . import pycuda_ops
from neural_nets.models import HiddenLayer, NeuralNet
from neural_nets.pycuda_ops.elementwise import sigmoid_kernel, df_sigmoid, \
     tanh_kernel, df_tanh, relu_kernel, df_relu
from neural_nets.pycuda_ops.reductions import matrix_sum_out_axis

STRIDE = 4

class SequenceConvolutionLayer(HiddenLayer):
    def __init__(self, n_in, filter_width, n_filters, activation_function='sigmoid',
                 weights_scale=.01, W=None, b=None):
        if W is None:
            self.W = weights_scale * \
              curand((n_filters, filter_width), dtype=np.float32) \
              -.5 * weights_scale
        else:
            self.W = W

        if b is None:
            self.b = gpuarray.zeros((n_filters,), np.float32)
        else:
            self.b = b
            
        assert self.W.shape == (n_filters, filter_width)
        assert self.b.shape == (n_filters,)
        assert not n_in % STRIDE
        
        self.n_in = n_in
        self.filter_width = filter_width
        self.n_filters = n_filters

        self._set_activation_fct(activation_function)
        self.l1_penalty_weight = 0.
        self.l2_penalty_weight = 0.

    @property
    def l1_penalty(self):
        return 0.

    @property
    def l2_penalty(self):
        return 0.

    def feed_forward(self, input, prediction=False):
        activations = \
            pycuda_ops.convolve_sequence(input, self.W, self.b, stride=STRIDE)

        self.f(activations)
        return (activations,)

    def backprop(self, input, df_output, cache=None):
        if cache is None:
            activations = self.feed_forward(input)
        else:
            activations = cache[0]

        df_activations = self.df(activations)
        delta = df_activations * df_output
        df_b = matrix_sum_out_axis(
            delta.reshape((self.n_filters, delta.shape[1]*delta.shape[2])), 1)
        df_W = pycuda_ops.convolve_sequence_gradient(
            input, delta,
            self.filter_width, self.n_filters)

        return (df_W, df_b, gpuarray.zeros(1, np.float32))

class MaxPoolingLayer(HiddenLayer):
    def __init__(self, pool_size, n_filters):
        self.pool_size = pool_size
        self.n_filters = n_filters

        self.l1_penalty_weight = 0.
        self.l2_penalty_weight = 0.

        self.W = gpuarray.zeros(1, np.float32)
        self.b = gpuarray.zeros(1, np.float32)

    @property
    def l1_penalty(self):
        return 0.

    @property
    def l2_penalty(self):
        return 0.

    def feed_forward(self, input, prediction=False):
        activations, argmax = pycuda_ops.max_pool(input, self.pool_size)
        n, f, m = activations.shape
        return (activations.reshape((n, f*m)), argmax)

    def backprop(self, input, df_output, cache=None):
        if cache is None:
            activations, argmax = self.feed_forward(input)
        else:
            activations, argmax = cache

        n, fm = activations.shape
        activations = activations.reshape((n, self.n_filters, 
                                           fm / self.n_filters))
        df_input = pycuda_ops.max_pool_gradient(input, argmax,
                                                df_output,
                                                self.pool_size)
        return (gpuarray.zeros(1, np.float32), gpuarray.zeros(1, np.float32), df_input)

class SequenceConvolutionNet(NeuralNet):
    def __init__(self, n_in, n_out, filter_width, n_filters, 
                 pool_size, layers, activation_function='sigmoid',
                 dropout=False, l1_penalty_weight=0., l2_penalty_weight=0.,
                 **kwargs):

        n_in_nn = n_filters * n_in / STRIDE / pool_size
        super(SequenceConvolutionNet, self)\
          .__init__(n_in_nn, n_out, layers, activation_function,
                    dropout, l1_penalty_weight, l2_penalty_weight, **kwargs)

        self.n_layers = len(layers) + 2
        self.n_in = n_in
        self.n_in_nn = n_in_nn
        self.filter_width = filter_width
        self.n_filters = n_filters
        self.pool_size = pool_size
        
        self.conv_layer = SequenceConvolutionLayer(n_in, filter_width, n_filters, 
                                                   activation_function=activation_function)
        self.max_pool_layer = MaxPoolingLayer(pool_size, n_filters)
        self.fully_connected_layers = self.hidden_layers
        self.hidden_layers = [self.conv_layer, self.max_pool_layer] + self.fully_connected_layers

        self.lr_multiplier = [np.array(.1)] + self.lr_multiplier
