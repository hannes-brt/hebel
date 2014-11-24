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

from .parameterfree_layer import ParameterfreeLayer

class FlatteningLayer(ParameterfreeLayer):
    def __init__(self, height, width, n_filters,
                 l1_penalty_weight=0., l2_penalty_weight=0.):
        self.height = height
        self.width = width
        self.n_filters = n_filters
        self.n_units = height * width * n_filters

        self.l1_penalty_weight = 0.
        self.l2_penalty_weight = 0.

    def feed_forward(self, input_data, prediction=False):
        N = input_data.shape[0]
        return input_data.reshape((N, self.n_units)), None

    def backprop(self, input_data, df_output, cache=None):
        N = input_data.shape[0]
        return tuple(), df_output.reshape((N, self.n_filters,
                                           self.height, self.width))
