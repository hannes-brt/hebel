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
from hebel.pycuda_ops.reductions import matrix_sum_out_axis
from hebel.pycuda_ops.matrix import pad_array, rand_array, extract_columns

class Convolution1DLayer(HiddenLayer):
    n_parameters = 2
    is_master_layer = True
    
    def __init__(self, n_in, filter_width, n_filters_in, n_filters_out,
                 activation_function='relu',
                 weights_scale=.01,
                 W=None, b=None,
                 l1_penalty_weight=0., l2_penalty_weight=0.,
                 padding=(True, True)):

        if W is None:
            self.W = weights_scale * \
                (rand_array((n_filters_out, filter_width, n_filters_in)) - .5)
        else:
            self.W = W

        if b is None:
            self.b = gpuarray.zeros((n_filters_out,),
                                    np.float32,
                                    allocator=memory_pool.allocate)
        else:
            self.b = b

        assert self.W.shape == (n_filters_out, filter_width, n_filters_in)
        assert self.b.shape == (n_filters_out,)

        self.n_in = n_in
        halo = filter_width - 1
        self.n_in_padded = self.n_in + sum(padding) * halo
        self.filter_width = filter_width
        self.n_filters_in = n_filters_in
        self.n_filters_out = n_filters_out
        self.n_units_per_filter = self.n_in_padded - halo
        self.n_units = self.n_units_per_filter * self.n_filters_out

        self._set_activation_fct(activation_function)
        self.l1_penalty_weight = l1_penalty_weight
        self.l2_penalty_weight = l2_penalty_weight

        self.lr_multiplier = [1., 1.]

        self.padding = padding
        self.halo = filter_width - 1

    def feed_forward(self, input_data, prediction=False):
        if any(self.padding):
            new_shape = (input_data.shape[0], self.n_in_padded, input_data.shape[2])
            input_padded = pad_array(input_data,
                                     left=self.n_filters_in * self.halo
                                     if self.padding[0] else 0,
                                     right=self.n_filters_in * self.halo
                                     if self.padding[1] else 0,
                                     val=0.,
                                     new_shape=new_shape)
        else:
            input_padded = input_data
            
        filtermap = pycuda_ops.convolve_1d(input_padded, self.W, self.b)

        self.f(filtermap)
        return filtermap, input_padded

    def backprop(self, input_data, df_output, cache=None):
        if cache is None:
            filtermap, input_padded = self.feed_forward(input_data, False)
        else:
            filtermap, input_padded = cache

        if len(filtermap.shape) == 2:
            h, w = filtermap.shape
            filtermap = filtermap.reshape((h, self.n_units_per_filter, self.n_filters_out))

        h, w, f = filtermap.shape
        filtermap = filtermap.reshape(h, w * f)
        df_output = df_output.reshape(h, w * f)
        df_filtermap = self.df(filtermap)
        delta = df_filtermap * df_output
        df_b = pycuda_ops.sum_delta(delta, self.n_filters_out)
        delta = delta.reshape(h, w, f)
        df_W = pycuda_ops.convolve_1d_gradient_filters(
            input_padded, delta, self.filter_width
        )
        df_input = pycuda_ops.convolve_1d_gradient_input(delta, self.W)

        if any(self.padding):
            column_start = 0 + self.padding[0] * self.halo
            column_end = column_start + self.n_in
            df_input = extract_columns(df_input, column_start, column_end)

        # L1 weight decay
        if self.l1_penalty_weight:
            df_W += self.l1_penalty_weight * np.sign(self.W)

        # L2 weight decay
        if self.l2_penalty_weight:
            df_W += self.l2_penalty_weight * self.W

        return (df_W, df_b), df_input


class SlavedConvolution1DLayer(Convolution1DLayer):
    is_master_layer = False
    n_parameters = 0

    def __init__(self, master_layer, n_in=None, padding=None):

        self.n_in = master_layer.n_in if n_in is None else n_in
        self.padding = master_layer.padding if padding is None else padding
        self.n_filters_in = master_layer.n_filters_in
        self.n_filters_out = master_layer.n_filters_out
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
        self.n_units = self.n_units_per_filter * self.n_filters_out

