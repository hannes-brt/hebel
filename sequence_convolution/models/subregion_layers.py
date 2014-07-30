import numpy as np
from pycuda import gpuarray
from itertools import izip
from .. import pycuda_ops
from . import MaxPoolingLayer
from hebel import sampler, memory_pool
from hebel.layers import HiddenLayer
from hebel.pycuda_ops.elementwise import sign, sample_dropout_mask, \
     apply_dropout_mask, mult_matrix
from hebel.pycuda_ops.matrix import extract_columns, insert_columns
from hebel.pycuda_ops.reductions import matrix_sum_out_axis


class SubregionLayer(HiddenLayer):
    is_master_layer = True
    n_parameters = 2
    
    def __init__(self, n_in, n_filters, filter_width, pool_size,
                 activation_function, l1_penalty_weight=0.,
                 l2_penalty_weight=0., lr_multiplier=1.,
                 parameters=None, weights_scale=.01, param_idx=None,
                 output_offset=0):

        self.n_in = n_in
        self.n_filters = n_filters
        self.filter_width = filter_width
        self.pool_size = pool_size
        self.activation_function = activation_function
        self.l1_penalty_weight = l1_penalty_weight
        self.l2_penalty_weight = l2_penalty_weight
        self.lr_multiplier = lr_multiplier
        self.param_idx = param_idx
        self.output_offset = output_offset

        if n_in % pool_size:
            raise ValueError("Pool size must be an even divider of n_in")

        if parameters is None:
            self.W = gpuarray.empty((n_filters, 4*filter_width), dtype=np.float32,
                                    allocator=memory_pool.allocate)
            sampler.fill_uniform(self.W)
            self.W = weights_scale * (self.W - .5)
            self.b = gpuarray.zeros((n_filters,), np.float32,
                                    allocator=memory_pool.allocate)
        else:
            self.W, self.b = parameters

        assert self.W.shape == (n_filters, 4 * filter_width)
        assert self.b.shape == (n_filters,)

        self.f, self.df = self._resolve_activation_fct(self.activation_function)
        self.n_units = MaxPoolingLayer._compute_n_units(self.n_in,
            self.pool_size, self.n_filters)

    def feed_forward(self, input_data, prediction=False,
                     target_activations=None, target_argmax=None):

        if target_activations is None:
            target_activations = gpuarray.empty((input_data.shape[0], self.n_units),
                                                input_data.dtype, allocator=memory_pool.allocate)
            target_argmax = gpuarray.empty(target_activations.shape, np.uint32,
                                           allocator=memory_pool.allocate)
            target_offset = 0
            
        filtermap = pycuda_ops.convolve_sequence(input_data, self.W, self.b)
        self.f(filtermap)
        if self.pool_size is None or self.pool_size == 1:
            insert_columns(filtermap, target_activations, self.output_offset)
        else:
            target_activations, target_argmax = \
                pycuda_ops.max_pool(filtermap, self.pool_size,
                                    width=self.n_in*self.n_filters,
                                    pooled_offset=self.output_offset,
                                    target=target_activations, argmax=target_argmax)
        return filtermap

    def backprop(self, input_data, df_output, filtermap, argmax):
        df_filtermap = pycuda_ops.max_pool_gradient(filtermap, argmax, df_output,
                                                    self.pool_size,
                                                    width_pooled=self.n_units,
                                                    pooled_offset=self.output_offset)
        df_conv = self.df(filtermap)
        delta = mult_matrix(df_conv, df_filtermap)
        df_b = pycuda_ops.sum_delta(delta, self.n_filters)
        df_W = pycuda_ops.convolve_sequence_gradient(input_data, delta,
                                                     self.filter_width, self.n_filters)

        if self.l1_penalty_weight:
            W_sign = self.get_temp_object('W_sign', self.W.shape, self.W.dtype)
            sign(self.W, W_sign)
            df_W._axpbyz(1., W_sign, self.l1_penalty_weight, df_W)

        if self.l2_penalty_weight:
            df_W._axpbyz(1., self.W, self.l2_penalty_weight, df_W)
        
        return (df_W, df_b), df_filtermap

class SlavedSubregionLayer(SubregionLayer):
    is_master_layer = False
    
    def __init__(self, master_layer, output_offset=0):

        self.n_in = master_layer.n_in
        self.n_filters = master_layer.n_filters
        self.filter_width = master_layer.filter_width
        self.pool_size = master_layer.pool_size
        self.activation_function = master_layer.activation_function
        self.l1_penalty_weight = master_layer.l1_penalty_weight
        self.l2_penalty_weight = master_layer.l2_penalty_weight
        self.lr_multiplier = master_layer.lr_multiplier
        self.master_layer = master_layer
        self.param_idx = master_layer.param_idx
        self.output_offset = output_offset

        self.W = master_layer.W
        self.b = master_layer.b

        self.f = master_layer.f
        self.df = master_layer.df
        self.n_units = master_layer.n_units
