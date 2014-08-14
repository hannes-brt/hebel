# Copyright (C) 2013  Hannes Bretschneider

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.


import numpy as np
from pycuda import gpuarray
from .. import pycuda_ops
from hebel import sampler, memory_pool
from hebel.layers import HiddenLayer
from hebel.pycuda_ops.elementwise import sign
from hebel.pycuda_ops.matrix import pad_array, rand_array

class SequenceConvolutionLayer(HiddenLayer):
    n_parameters = 2
    is_master_layer = True

    def __init__(self, n_in, filter_width, n_filters, activation_function='relu',
                 weights_scale=.01, W=None, b=None,
                 l1_penalty_weight=0., l2_penalty_weight=0.,
                 padding=(True, True)):

        dtype = np.float32

        if W is None:
            self.W = weights_scale * (rand_array((n_filters, filter_width, 4)) - .5)
        else:
            self.W = W

        if b is None:
            self.b = gpuarray.zeros((n_filters,), dtype, allocator=memory_pool.allocate)
        else:
            self.b = b

        assert self.W.shape == (n_filters, filter_width, 4)
        assert self.b.shape == (n_filters,)

        self.n_in = n_in
        halo = filter_width - 1
        self.n_in_padded = self.n_in + sum(padding) * halo
        self.filter_width = filter_width
        self.n_filters = n_filters
        self.n_units_per_filter = self.n_in_padded - halo
        self.n_units = self.n_units_per_filter * self.n_filters

        self._set_activation_fct(activation_function)
        self.l1_penalty_weight = l1_penalty_weight
        self.l2_penalty_weight = l2_penalty_weight

        self.lr_multiplier = [1., 1.]

        self.padding = padding
        self.halo = filter_width - 1

    def feed_forward(self, input_data, prediction=False):
        if any(self.padding):
            input_padded = pad_array(input_data,
                                     left=self.halo if self.padding[0] else 0,
                                     right=self.halo if self.padding[1] else 0,
                                     val='N')
        else:
            input_padded = input_data
            
        filtermap = pycuda_ops.convolve_sequence(input_padded, self.W, self.b)

        self.f(filtermap)
        return filtermap, input_padded

    def backprop(self, input_data, df_output, cache=None):
        if cache is None:
            filtermap, input_padded = self.feed_forward(input_data)
        else:
            filtermap, input_padded = cache

        h, w, f = filtermap.shape
        filtermap = filtermap.reshape(h, w * f)
        df_output = df_output.reshape(h, w * f)
        df_filtermap = self.df(filtermap)
        delta = df_filtermap * df_output
        df_b = pycuda_ops.sum_delta(delta, self.n_filters)
        delta = delta.reshape(h, w, f)
        df_W = pycuda_ops.convolve_sequence_gradient(
            input_padded, delta,
            self.filter_width, self.n_filters)

        # L1 weight decay
        if self.l1_penalty_weight:
            df_W += self.l1_penalty_weight * sign(self.W)

        # L2 weight decay
        if self.l2_penalty_weight:
            df_W += self.l2_penalty_weight * self.W

        return (df_W, df_b), None


class SlavedSequenceConvolutionLayer(SequenceConvolutionLayer):
    is_master_layer = False

    def __init__(self, master_layer, n_in=None, padding=None):

        self.n_in = master_layer.n_in if n_in is None else n_in
        self.padding = master_layer.padding if padding is None else padding
        self.n_filters = master_layer.n_filters
        self.filter_width = master_layer.filter_width
        self.activation_function = master_layer.activation_function
        self.l1_penalty_weight = master_layer.l1_penalty_weight
        self.l2_penalty_weight = master_layer.l2_penalty_weight
        self.lr_multiplier = master_layer.lr_multiplier
        self.master_layer = master_layer

        self.W = master_layer.W
        self.b = master_layer.b

        self.f = master_layer.f
        self.df = master_layer.df

        self.halo = self.filter_width - 1
        self.n_in_padded = self.n_in + sum(self.padding) * self.halo
        self.n_units_per_filter = self.n_in_padded - self.halo
        self.n_units = self.n_units_per_filter * self.n_filters

