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

from . import SequenceConvolutionLayer, Convolution1DLayer, AveragePoolingLayer, \
    SumPoolingLayer, MaxPoolingLayer, \
    SlavedSequenceConvolutionLayer, SlavedConvolution1DLayer
from hebel.layers import HiddenLayer


class Convolution1DAndPoolLayer(HiddenLayer):
    is_master_layer = True

    conv_layer_attr = (
        'n_in', 'filter_width', 'n_filters_in', 'n_filters',
        'activation_function', 'weights_scale',
        'l1_penalty_weight', 'l2_penalty_weight', 'padding',
        'n_parameters', 'n_in_padded',
        'lr_multiplier', 'halo', 'l1_penalty', 'l2_penalty')
    pooling_layer_attr = ('n_units', 'n_units_per_filter',
                          'pool_size', 'dropout')

    def __init__(self, n_in, filter_width,
                 n_filters_in, n_filters,
                 pool_size,
                 activation_function='relu',
                 pooling_op='max', dropout=True,
                 weights_scale=.01,
                 W=None, b=None,
                 l1_penalty_weight=0., l2_penalty_weight=0.,
                 padding=(True, True)):
        self.conv_layer = Convolution1DLayer(n_in, filter_width, n_filters_in, n_filters,
                                             activation_function, weights_scale,
                                             W, b, l1_penalty_weight, l2_penalty_weight,
                                             padding)

        if pooling_op == 'max':
            self.pooling_layer = MaxPoolingLayer(self.conv_layer.n_units_per_filter, pool_size, n_filters,
                                                 dropout)
        elif pooling_op == 'sum':
            self.pooling_layer = SumPoolingLayer(self.conv_layer.n_units_per_filter, pool_size, n_filters,
                                                 dropout)
        elif pooling_op == 'avg':
            self.pooling_layer = AveragePoolingLayer(self.conv_layer.n_units_per_filter, pool_size, n_filters,
                                                     dropout)
        else:
            raise ValueError("Unknown pooling op '%s'" % pooling_op)
        self.pooling_op = pooling_op

    @property
    def l1_penalty(self):
        return self.conv_layer.l1_penalty

    @property
    def l2_penalty(self):
        return self.conv_layer.l2_penalty

    @property
    def parameters(self):
        return self.conv_layer.parameters

    @parameters.setter
    def parameters(self, value):
        self.conv_layer.parameters = value

    def update_parameters(self, values, stream=None):
        self.conv_layer.update_parameters(values, stream)

    @property
    def lr_multiplier(self):
        return self.conv_layer.lr_multiplier

    @lr_multiplier.setter
    def lr_multiplier(self, value):
        self.conv_layer.lr_multiplier = value

    def __getattr__(self, name):
        if name in self.conv_layer_attr:
            return self.conv_layer.__getattribute__(name)
        elif name in self.pooling_layer_attr:
            return self.pooling_layer.__getattribute__(name)
        else:
            return self.__getattribute__(name)

    def __setattr__(self, key, value):
        if key in self.conv_layer_attr:
            self.conv_layer.__setattr__(key, value)
        elif key in self.pooling_layer_attr:
            self.pooling_layer.__setattr__(key, value)
        else:
            self.__dict__[key] = value

    @property
    def W(self):
        return self.conv_layer.W

    @W.setter
    def W(self, value):
        self.conv_layer.W = value

    @property
    def b(self):
        return self.conv_layer.b

    @b.setter
    def b(self, value):
        self.conv_layer.b = value

    def feed_forward(self, input_data, prediction=False):
        filtermap, input_padded_conv = self.conv_layer.feed_forward(input_data, prediction)
        pooling_cache = self.pooling_layer.feed_forward(filtermap, prediction)
        pooled = pooling_cache[0]

        return pooled, ((filtermap, input_padded_conv), pooling_cache)

    def backprop(self, input_data, df_output, cache=None):
        if cache is None:
            _, cache = self.feed_forward(input_data, False)
        _, (conv_cache, pooling_cache) = cache
        filtermap = conv_cache[0]
        _, df_pooled = self.pooling_layer.backprop(filtermap, df_output, pooling_cache)
        (df_W, df_b), df_input = self.conv_layer.backprop(input_data, df_pooled, conv_cache)
        return (df_W, df_b), df_input


class SlavedConvolution1DAndPoolLayer(Convolution1DAndPoolLayer):
    is_master_layer = False

    def __init__(self, master_layer, n_in, padding):
        self.conv_layer = SlavedConvolution1DLayer(master_layer.conv_layer, n_in, padding)
        self.pooling_layer = master_layer.pooling_layer
        self.pooling_op = master_layer.pooling_op

    @property
    def master_layer(self):
        return self.conv_layer.master_layer


class SequenceConvolutionAndPoolLayer(Convolution1DAndPoolLayer):
    conv_layer_attr = (
        'n_in', 'filter_width', 'n_filters',
        'activation_function', 'weights_scale', 'W', 'b',
        'l1_penalty_weight', 'l2_penalty_weight', 'padding',
        'n_parameters', 'n_in_padded',
        'lr_multiplier', 'halo', 'l1_penalty', 'l2_penalty')
    pooling_layer_attr = ('n_units', 'n_units_per_filter',
                          'pool_size', 'dropout')

    def __init__(self, n_in, filter_width,
                 n_filters, pool_size,
                 activation_function='relu',
                 pooling_op='max', dropout=True,
                 weights_scale=.01,
                 W=None, b=None,
                 l1_penalty_weight=0., l2_penalty_weight=0.,
                 padding=(True, True)):
        self.conv_layer = SequenceConvolutionLayer(n_in, filter_width, n_filters,
                                                   activation_function, weights_scale,
                                                   W, b, l1_penalty_weight, l2_penalty_weight,
                                                   padding)

        if pooling_op == 'max':
            self.pooling_layer = MaxPoolingLayer(self.conv_layer.n_units_per_filter, pool_size, n_filters,
                                                 dropout)
        elif pooling_op == 'sum':
            self.pooling_layer = SumPoolingLayer(self.conv_layer.n_units_per_filter, pool_size, n_filters,
                                                 dropout)
        elif pooling_op == 'avg':
            self.pooling_layer = AveragePoolingLayer(self.conv_layer.n_units_per_filter, pool_size, n_filters,
                                                     dropout)
        else:
            raise ValueError("Unknown pooling op '%s'" % pooling_op)
        self.pooling_op = pooling_op


class SlavedSequenceConvolutionAndPoolLayer(SequenceConvolutionAndPoolLayer):
    is_master_layer = False
    n_parameters = 0

    def __init__(self, master_layer, n_in=None, padding=None):
        self.master_layer = master_layer
        self.conv_layer = SlavedSequenceConvolutionLayer(master_layer.conv_layer, n_in, padding)
        self.pooling_layer = master_layer.pooling_layer
        self.pooling_op = master_layer.pooling_op
