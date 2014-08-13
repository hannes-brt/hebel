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
from .pooling_layers import MaxPoolingLayer
from .sequence_convolution_layer import SequenceConvolutionLayer
from hebel.models import NeuralNet


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

        max_pool_layer = MaxPoolingLayer(conv_layer.n_units / n_filters,
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
