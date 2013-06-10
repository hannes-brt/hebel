import numpy as np
from pycuda import gpuarray
from pycuda.curandom import rand as curand
from pycuda import cumath
from math import sqrt
from scikits.cuda import linalg
from . import pycuda_ops
from neural_nets.models import HiddenLayer, NeuralNet
from neural_nets.pycuda_ops.reductions import matrix_sum_out_axis

STRIDE = 4

class SequenceConvolutionLayer(HiddenLayer):
    n_parameters = 2
    
    def __init__(self, n_in, filter_width, n_filters, activation_function='sigmoid',
                 weights_scale=.01, W=None, b=None, dtype=np.float32):
        if W is None:
            self.W = weights_scale * \
              curand((n_filters, filter_width), dtype=dtype) \
              -.5 * weights_scale
        else:
            self.W = W

        if b is None:
            self.b = gpuarray.zeros((n_filters,), dtype)
        else:
            self.b = b
            
        assert self.W.shape == (n_filters, filter_width)
        assert self.b.shape == (n_filters,)
        assert not n_in % STRIDE
        
        self.n_in = n_in
        self.filter_width = filter_width
        self.n_filters = n_filters
        self.n_units = n_filters * n_in / STRIDE

        self._set_activation_fct(activation_function)
        self.l1_penalty_weight = 0.
        self.l2_penalty_weight = 0.

        self.lr_multiplier = [1., 1.]

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
            activations = self.feed_forward(input)[0]
        else:
            activations = cache[0]

        df_activations = self.df(activations)
        delta = df_activations * df_output
        df_b = matrix_sum_out_axis(
            delta.reshape((self.n_filters, delta.shape[1]*delta.shape[2])), 1)
        df_W = pycuda_ops.convolve_sequence_gradient(
            input, delta,
            self.filter_width, self.n_filters)

        return (df_W, df_b), None

class MaxPoolingLayer(HiddenLayer):
    n_parameters = 0
    lr_multiplier = []
    
    def __init__(self, n_in, pool_size, n_filters):
        self.n_in = n_in
        self.pool_size = pool_size
        self.n_filters = n_filters

        self.l1_penalty_weight = 0.
        self.l2_penalty_weight = 0.

        self.n_units = n_in / pool_size

    @property
    def parameters(self):
        return []

    @parameters.setter
    def parameters(self, value):
        pass
        
    def update_parameters(self, values, stream=None):
        pass
        
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
        return tuple(), df_input

class SequenceConvolutionNet(NeuralNet):
    def __init__(self, n_in, n_out, filter_width, n_filters, 
                 pool_size, layers, activation_function='sigmoid',
                 dropout=False, l1_penalty_weight=0., l2_penalty_weight=0.,
                 **kwargs):

        n_in_nn = n_filters * n_in / STRIDE / pool_size
        conv_layer = SequenceConvolutionLayer(n_in, filter_width, n_filters, 
                                                   activation_function=activation_function)
        max_pool_layer = MaxPoolingLayer(conv_layer.n_units, 
                                         pool_size, n_filters)
        hidden_layers = [conv_layer, max_pool_layer] + layers
        
        super(SequenceConvolutionNet, self)\
          .__init__(n_in, n_out, hidden_layers, activation_function,
                    dropout, l1_penalty_weight, l2_penalty_weight, **kwargs)

        self.n_layers = len(layers) + 2
        self.n_in = n_in
        self.n_in_nn = n_in_nn
        self.filter_width = filter_width
        self.n_filters = n_filters
        self.pool_size = pool_size
        
        # self.fully_connected_layers = self.hidden_layers
        # self.hidden_layers = [self.conv_layer, self.max_pool_layer] + self.fully_connected_layers
        
