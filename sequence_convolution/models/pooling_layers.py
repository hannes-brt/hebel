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
from .. import pycuda_ops
from hebel.layers import HiddenLayer
from hebel.pycuda_ops.elementwise import sample_dropout_mask, \
     apply_dropout_mask
from hebel.pycuda_ops.matrix import pad_array
from hebel.utils.math import ceil_div, div_up


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

        self.n_units_per_filter = self._compute_n_units(n_in, pool_size, n_filters)
        self.n_units = self.n_units_per_filter * self.n_filters

    @staticmethod
    def _compute_n_units(n_in, pool_size, n_filters):
        """ Compute the number of output units """
        if pool_size is None:
            pool_size = 1
        return ceil_div(n_in, pool_size)

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

    def feed_forward(self, input_data, prediction=False):
        padded_width = div_up(self.n_in, self.pool_size)
        padding_width = padded_width - self.n_in
        new_shape = (input_data.shape[0], padded_width, input_data.shape[2])
        input_padded = pad_array(input_data, right=padding_width * input_data.shape[2],
                                 new_shape=new_shape, val=np.finfo(np.float32).min)

        pooled, argmax = pycuda_ops.max_pool(input_padded, self.pool_size)

        if self.dropout and prediction:
            pooled *= .5

        if self.dropout and not prediction:
            dropout_mask = sample_dropout_mask(pooled)
            return pooled, argmax, dropout_mask, input_padded

        return pooled, argmax, input_padded

    def backprop(self, input, df_output, cache=None):
        if cache is None:
            cache = self.feed_forward(input)

        if len(cache) == 3:
            activations, argmax, input_padded = cache
        elif len(cache) == 4:
            activations, argmax, dropout_mask, input_padded = cache
        else:
            raise ValueError

        if len(df_output.shape) == 2:
            h, w = df_output.shape
            df_output = df_output.reshape((h, self.n_units_per_filter, self.n_filters))

        if self.dropout and dropout_mask is not None:
            apply_dropout_mask(df_output, dropout_mask)

        df_input = pycuda_ops.max_pool_gradient(
            input, argmax, df_output)
        return tuple(), df_input

class SumPoolingLayer(MaxPoolingLayer):
    def feed_forward(self, input_data, prediction=False):
        padded_width = div_up(self.n_in, self.pool_size)
        padding_width = padded_width - self.n_in
        new_shape = (input_data.shape[0], padded_width, input_data.shape[2])
        input_padded = pad_array(input_data, right=padding_width * input_data.shape[2],
                                 new_shape=new_shape, val=0.)

        pooled = pycuda_ops.sum_pool(input_padded, self.pool_size)

        if self.dropout and prediction:
            pooled *= .5

        if self.dropout and not prediction:
            dropout_mask = sample_dropout_mask(pooled)
            return pooled, dropout_mask, input_padded

        return pooled, input_padded

    def backprop(self, input_data, df_output, cache=None):
        if cache is None:
            cache = self.feed_forward(input_data)

        if len(cache) == 2:
            activations, input_padded = cache
        elif len(cache) == 3:
            activations, dropout_mask, input_padded = cache
        else:
            raise ValueError

        if self.dropout and dropout_mask is not None:
            apply_dropout_mask(df_output, dropout_mask)

        df_input = pycuda_ops.sum_pool_gradient(input_padded, df_output)
        return tuple(), df_input

class AveragePoolingLayer(SumPoolingLayer):
    def feed_forward(self, input_data, prediction=False):
        ff = super(AveragePoolingLayer, self)\
            .feed_forward(input_data, prediction)
        pooled = ff[0]
        pooled /= self.pool_size
        ff = (pooled,) + ff[1:]
        return ff

    def backprop(self, input_data, df_output, cache=None):
        _, df_input = super(AveragePoolingLayer, self)\
                      .backprop(input_data, df_output, cache)
        df_input /= self.pool_size
        return tuple(), df_input
