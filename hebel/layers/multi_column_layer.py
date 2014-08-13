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

from .. import memory_pool
from . import HiddenLayer, Column
from pycuda import gpuarray
import numpy as np
from ..pycuda_ops.matrix import insert_columns, extract_columns
from itertools import chain

class MultiColumnLayer(HiddenLayer):
    l1_penalty_weight = True
    l2_penalty_weight = True

    def __init__(self, columns, input_as_list=False):
        assert all([isinstance(c, (Column, HiddenLayer)) for c in columns])
        self.columns = columns
        self.input_as_list = input_as_list

    @property
    def n_in(self):
        return sum(c.n_in for c in self.columns)

    @property
    def n_units(self):
        return sum(c.n_units for c in self.columns)

    @property
    def lr_multiplier(self):
        return tuple(chain.from_iterable((c.lr_multiplier for c in self.columns)))

    @property
    def n_parameters(self):
        return sum(c.n_parameters for c in self.columns)

    @property
    def parameters(self):
        return tuple(chain.from_iterable((c.parameters for c in self.columns)))

    @parameters.setter
    def parameters(self, value):
        assert len(value) == self.n_parameters

        i = 0
        for c in self.columns:
            c.parameters = value[i:i+c.n_parameters]
            i += c.n_parameters

    def update_parameters(self, values, stream=None):
        assert len(values) == self.n_parameters

        i = 0
        for c in self.columns:
            c.update_parameters(values[i:i+c.n_parameters])
            i += c.n_parameters

    @property
    def l1_penalty(self):
        return sum(c.l1_penalty for c in self.columns if c.l1_penalty_weight)

    @property
    def l2_penalty(self):
        return sum(c.l2_penalty for c in self.columns if c.l2_penalty_weight)

    def feed_forward(self, input_data, prediction=False):
        if self.input_as_list:
            return self._feed_forward_list(input_data, prediction)
        else:
            return self._feed_forward_array(input_data, prediction)

    def _feed_forward_list(self, input_data, prediction=False):
        output = gpuarray.empty((input_data[0].shape[0], self.n_units), np.float32,
                                allocator=memory_pool.allocate)
        cache = []
        i_out = 0
        for column, input_column in zip(self.columns, input_data):
            c = column.feed_forward(input_column, prediction)
            cache.append((input_column, c))
            insert_columns(c[0], output, i_out)
            i_out += column.n_units

        return output, cache

    def _feed_forward_array(self, input_data, prediction=False):
        output = gpuarray.empty((input_data.shape[0], self.n_units), np.float32,
                                allocator=memory_pool.allocate)
        cache = []
        i_in = 0
        i_out = 0
        for column in self.columns:
            input_column = extract_columns(input_data, i_in, i_in + column.n_in)
            c = column.feed_forward(input_column, prediction)
            cache.append((input_column, c))
            insert_columns(c[0], output, i_out)
            i_in += column.n_in
            i_out += column.n_units

        return output, cache

    def backprop(self, input_data, df_output, cache=None):
        if cache is None:
            _, cache = self.feed_forward(input_data, False)
        else:
            cache = cache[1]

        df_params = []
        df_input = []
        i = 0
        for column, cache_column in zip(self.columns, cache):
            df_output_column = extract_columns(df_output, i, i + column.n_units)
            df_params_column, df_input_column = column.backprop(cache_column[0], df_output_column, cache_column[1])
            df_params.extend(df_params_column)
            df_input.append(df_input_column)

        if not self.input_as_list:
            df_input_list = df_input
            df_input = gpuarray.empty(input_data.shape, np.float32,
                                      allocator=memory_pool.allocate)

            i = 0
            for dfi, column in zip(df_input_list, self.columns):
                insert_columns(dfi, df_input, i)
                i += column.n_in

        return df_params, df_input
