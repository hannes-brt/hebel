import numpy as np
from pycuda import gpuarray
from pycuda.curandom import rand as curand
from pycuda import cumath
from pycuda.driver import Stream
from math import sqrt
from itertools import izip
from scikits.cuda import linalg
from . import pycuda_ops
from neural_nets.models import HiddenLayer, NeuralNet
from neural_nets.pycuda_ops.reductions import matrix_sum_out_axis
from neural_nets.pycuda_ops.elementwise import sign, sample_dropout_mask, \
     apply_dropout_mask, sigmoid, df_sigmoid, tanh, df_tanh, relu, df_relu

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

class MaxPoolingLayer(HiddenLayer):
    n_parameters = 0
    lr_multiplier = []
    
    def __init__(self, n_in, pool_size, n_filters, dropout=False,
                 l1_penalty_weight=0., l2_penalty_weight=0.):
        self.n_in = n_in
        self.pool_size = pool_size
        self.n_filters = n_filters

        self.l1_penalty_weight = 0.
        self.l2_penalty_weight = 0.

        self.dropout = dropout

        self.n_units = int(np.ceil(n_in / float(pool_size))) * n_filters

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
        activations, argmax = pycuda_ops.max_pool(input, self.pool_size, self.n_filters)

        if self.dropout and prediction:
            activations *= .5

        if self.dropout and not prediction:
            dropout_mask = sample_dropout_mask(activations)
            return activations, argmax, dropout_mask

        return activations, argmax

    def backprop(self, input, df_output, cache=None):
        if cache is None:
            cache = self.feed_forward(input)

        if len(cache) == 2:
            activations, argmax = cache
        elif len(cache) == 3:
            activations, argmax, dropout_mask = cache
        else:
            raise ValueError

        if self.dropout and dropout_mask is not None:
            apply_dropout_mask(df_output, dropout_mask)

        n, fm = activations.shape
        activations = activations.reshape((n, self.n_filters, 
                                           fm / self.n_filters))
        df_input = pycuda_ops.max_pool_gradient(input, argmax,
                                                df_output,
                                                self.pool_size,
                                                self.n_filters)
        return tuple(), df_input

class MultiSequenceConvolutionLayer(HiddenLayer):
    def __init__(self, subregion_layers, dtype=np.float32,
                 weight_scale=.01):
        self.subregion_layers = subregion_layers
        self.dtype = dtype

        self.W = []
        self.b = []

        output_offset = 0
        param_idx = 0
        for layer in subregion_layers:
            n_in = layer['n_in']

            if not layer.has_key('weight_share'):
                layer['layer_type'] = 'master'
                _weight_scale = layer.get('weight_scale', weight_scale)
                if layer.get('W') is None:
                    W = _weight_scale * \
                      curand((layer['n_filters'], 4*layer['filter_width']),
                             dtype) - .5 * _weight_scale
                else:
                    W = layer['W']

                assert W.shape == (layer['n_filters'], 4*layer['filter_width'])
                self.W.append(W)

                if layer.get('b') is None:
                    b = gpuarray.zeros((layer['n_filters'],), dtype)
                else:
                    b = layer['b']

                assert b.shape == (layer['n_filters'],)
                self.b.append(b)

                layer['param_idx'] = param_idx
                param_idx += 1

                if layer['activation_function'] == 'sigmoid':
                    layer['f'] = sigmoid
                    layer['df'] = df_sigmoid
                elif layer['activation_function'] == 'tanh':
                    layer['f'] = tanh
                    layer['df'] = df_tanh
                elif layer['activation_function'] == 'relu':
                    layer['f'] = relu
                    layer['df'] = df_relu
                else:
                    raise ValueError
            else:
                layer['layer_type'] = 'slave' 
                master_layer = subregion_layers[layer['weight_share']]                
                layer['n_filters'] = master_layer['n_filters']
                layer['filter_width'] = master_layer['filter_width']
                layer['param_idx'] = master_layer['param_idx']
                layer['activation_function'] = master_layer['activation_function']
                layer['f'] = master_layer['f']
                layer['df'] = master_layer['df']

            layer['n_units'] = int(np.ceil(layer['n_in'] / 
                float(layer['pool_size']))) * layer['n_filters']

            layer['output_offset'] = output_offset
            output_offset += layer['n_units']

        self.n_units = sum((layer['n_units'] for layer in subregion_layers))

        self.l1_penalty_weight = 0.
        self.l2_penalty_weight = 0.

    @property
    def n_parameters(self):
        return len(self.W) + len(self.b)

    @property
    def n_in(self):
        return sum((l['n_in'] for l in self.subregion_layers))

    @property
    def lr_multiplier(self):
        return self.n_parameters * [1.]

    @property
    def parameters(self):
        return self.W + self.b

    @parameters.setter
    def parameters(self, value):
        assert len(value) == self.n_parameters
        
        self.W = value[:len(self.W)]
        self.b = value[len(self.W):]

    def update_parameters(self, values, stream=None):
        assert len(values) == self.n_parameters

        for (param, (gparam, mult)) \
          in izip(self.W + self.b, values):
          param._axpbyz(1., gparam, mult, param, stream=stream)

    @property
    def l1_penalty(self):
        return 0.

    @property
    def l2_penalty(self):
        return 0.

    def feed_forward(self, input, prediction=False):
        assert all((input[0].shape[0] == i.shape[0] for i in input[1:]))

        N = input[0].shape[0]
        activations_pooled = gpuarray.empty((N, self.n_units), 
                                            self.dtype)
        argmax = gpuarray.empty(activations_pooled.shape,
                                np.uint32)

        filtermaps = []

        for input_region, layer \
            in izip(input, self.subregion_layers):
            W = self.W[layer['param_idx']]
            b = self.b[layer['param_idx']]
            act_fct = layer['f']
            
            filtermap = pycuda_ops.convolve_sequence(input_region, W, b)
            act_fct(filtermap)
            filtermaps.append(filtermap)
            pycuda_ops.max_pool(filtermap, layer['pool_size'], layer['n_filters'],
                                pooled_offset=layer['output_offset'],
                                target=activations_pooled, argmax=argmax)

        return activations_pooled, argmax, filtermaps

    def backprop(self, input, df_output, cache=None):
        if cache is None:
            activations_pooled, argmax, filtermaps = self.feed_forward(input)
        else:
            activations_pooled, argmax, filtermaps = cache

        df_W = []
        df_b = []
        df_filtermaps = []
            
        for input_region, filtermap, layer \
          in izip(input, filtermaps, self.subregion_layers):
            W = self.W[layer['param_idx']]
            b = self.b[layer['param_idx']]
            act_fct = layer['f']
            act_df = layer['df']

            df_filtermap = pycuda_ops.max_pool_gradient(
                filtermap, argmax, df_output, layer['pool_size'],
                layer['n_filters'],
                width_pooled=layer['n_units']/layer['n_filters'],
                pooled_offset=layer['output_offset'])

            df_filtermaps.append(df_filtermap)
            
            df_conv = act_df(filtermap)
            delta = df_conv * df_filtermap
            df_b_layer = pycuda_ops.sum_delta(delta, layer['n_filters'])
            df_W_layer = pycuda_ops.convolve_sequence_gradient(
                input_region, delta, layer['filter_width'], layer['n_filters'])

            if layer['layer_type'] == 'master':
                df_W.append(df_W_layer)
                df_b.append(df_b_layer)
            else:
                df_W[layer['param_idx']] += df_W_layer
                df_b[layer['param_idx']] += df_b_layer

        return df_W + df_b, df_filtermaps

class SequenceConvolutionNet(NeuralNet):
    def __init__(self, n_in, n_out, filter_width, n_filters, 
                 pool_size, layers, activation_function='sigmoid',
                 dropout=False, l1_penalty_weight=0., l2_penalty_weight=0.,
                 **kwargs):

        if np.isscalar(l1_penalty_weight):
            l1_conv = l1_penalty_weight
            l1_nn = l1_penalty_weight
        else:
            l1_conv = l1_penalty_weight[0]
            l1_nn = l1_penalty_weight[1:]

        if np.isscalar(l2_penalty_weight):
            l2_conv = l2_penalty_weight
            l2_nn = l2_penalty_weight
        else:
            l2_conv = l2_penalty_weight[0]
            l2_nn = l2_penalty_weight[1:]

        n_in_nn = n_filters * n_in / pool_size
        conv_layer = SequenceConvolutionLayer(n_in, filter_width, n_filters, 
                                              activation_function=activation_function,
                                              l1_penalty_weight=l1_conv, 
                                              l2_penalty_weight=l2_conv)
        
        max_pool_layer = MaxPoolingLayer(conv_layer.n_units, 
                                         pool_size, n_filters, 
                                         dropout=dropout)
        
        hidden_layers = [conv_layer, max_pool_layer] + layers
        
        super(SequenceConvolutionNet, self)\
          .__init__(layers=hidden_layers, 
                    activation_function=activation_function,
                    dropout=dropout, 
                    l1_penalty_weight=l1_nn, 
                    l2_penalty_weight=l2_nn, 
                    n_in=n_in, n_out=n_out, **kwargs)

        self.n_layers = len(layers) + 2
        self.n_in = n_in
        self.n_in_nn = n_in_nn
        self.filter_width = filter_width
        self.n_filters = n_filters
        self.pool_size = pool_size
        
        # self.fully_connected_layers = self.hidden_layers
        # self.hidden_layers = [self.conv_layer, self.max_pool_layer] + self.fully_connected_layers
        
