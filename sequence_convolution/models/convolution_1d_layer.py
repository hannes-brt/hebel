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
from hebel import memory_pool
from hebel.layers import HiddenLayer
from hebel.pycuda_ops.matrix import rand_array
from hebel.pycuda_ops.elementwise import sign
from hebel.pycuda_ops import cudnn

class Convolution1DLayer(HiddenLayer):
    n_parameters = 2
    is_master_layer = True
    
    def __init__(self, n_in, filter_width, n_filters_in, n_filters,
                 activation_function='relu',
                 weights_scale=.01,
                 W=None, b=None,
                 l1_penalty_weight=0., l2_penalty_weight=0.,
                 padding=True):

        if W is None:
            self.W = weights_scale * \
                (rand_array((n_filters, n_filters_in, 1, filter_width)) - .5)
        else:
            self.W = W

        if b is None:
            self.b = gpuarray.zeros((1, n_filters, 1, 1),
                                    np.float32,
                                    allocator=memory_pool.allocate)
        else:
            self.b = b

        assert self.W.shape == (n_filters, n_filters_in, 1, filter_width)
        assert self.b.shape == (1, n_filters, 1, 1)

        self.n_in = n_in
        halo = filter_width - 1
        self.n_in_padded = self.n_in + ((2 * halo) if padding else 0)
        self.filter_width = filter_width
        self.n_filters_in = n_filters_in
        self.n_filters = n_filters
        self.n_units_per_filter = self.n_in_padded - halo
        self.n_units = self.n_units_per_filter * self.n_filters

        self.activation_function = activation_function
        self.l1_penalty_weight = l1_penalty_weight
        self.l2_penalty_weight = l2_penalty_weight

        self.lr_multiplier = [1., 1.]

        self.padding = padding
        self.halo = filter_width - 1

    def feed_forward(self, input_data, prediction=False):
        input_desc = cudnn.Tensor4dDesc(input_data.shape[0], self.n_filters_in, 1, self.n_in)
        filter_desc = cudnn.FilterDesc(self.n_filters, self.n_filters_in, 1, self.filter_width)

        padding = self.halo if self.padding else 0
        conv_desc = cudnn.ConvolutionDesc(input_desc, filter_desc, pad_w=padding)

        filtermap = cudnn.convolution_forward(input_data, self.W, self.b, conv_desc)

        filtermap_act = cudnn.activation_forward(filtermap, self.activation_function)
        return filtermap_act, (filtermap, conv_desc)

    def backprop(self, input_data, df_output, cache=None):
        if cache is None:
            filtermap_act, (filtermap, conv_desc) = self.feed_forward(input_data, False)
        else:
            filtermap_act, (filtermap, conv_desc) = cache

        df_filtermap = cudnn.activation_backward(filtermap, filtermap_act,
                                                 df_output, self.activation_function)
        df_b, df_W, df_input = cudnn.convolution_backward(input_data, self.W, df_filtermap, conv_desc)

        # L1 weight decay
        if self.l1_penalty_weight:
            df_W += self.l1_penalty_weight * sign(self.W)

        # L2 weight decay
        if self.l2_penalty_weight:
            df_W += self.l2_penalty_weight * self.W

        return (df_W, df_b), df_input


class SlavedConvolution1DLayer(Convolution1DLayer):
    is_master_layer = False
    n_parameters = 0
    lr_multiplier = []

    def __init__(self, master_layer, n_in=None, padding=None,
                 l1_penalty_weight=0., l2_penalty_weight=0.):

        self.n_in = master_layer.n_in if n_in is None else n_in
        self.padding = master_layer.padding if padding is None else padding
        self.n_filters_in = master_layer.n_filters_in
        self.master_layer = master_layer

        self.activation_function = master_layer.activation_function

        self.halo = self.filter_width - 1
        self.n_in_padded = self.n_in + sum(self.padding) * self.halo
        self.n_units_per_filter = self.n_in_padded - self.halo
        self.n_units = self.n_units_per_filter * self.n_filters

        self.l1_penalty_weight = master_layer.l1_penalty_weight \
            if l1_penalty_weight is None else l1_penalty_weight

        self.l2_penalty_weight = master_layer.l2_penalty_weight \
            if l2_penalty_weight is None else l2_penalty_weight

    @property
    def parameters(self):
        return tuple()

    @parameters.setter
    def parameters(self, value):
        pass

    @property
    def W(self):
        return self.master_layer.W

    @W.setter
    def W(self, value):
        raise AttributeError('Setting the weights is not allowed in a slaved layer.')

    @property
    def b(self):
        return self.master_layer.b

    @b.setter
    def b(self):
        raise AttributeError('Setting the biases is not allowed in a slaved layer.')

    @property
    def n_filters(self):
        return self.master_layer.n_filters

    @property
    def filter_width(self):
        return self.master_layer.filter_width

    @property
    def activation_function(self):
        return self.master_layer.activation_function

