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
from . import HiddenLayer

class FlatteningLayer(HiddenLayer):
    n_parameters = 0
    lr_multiplier = []

    def __init__(self, n_in, n_filters,
                 l1_penalty_weight=0., l2_penalty_weight=0.):
        self.n_in = n_in
        self.n_filters = n_filters
        self.n_units = n_in * n_filters

        self.l1_penalty_weight = 0.
        self.l2_penalty_weight = 0.

    def feed_forward(self, input_data, prediction=False):
        N = input_data.shape[0]
        return input_data.reshape((N, self.n_units)), None

    def backprop(self, input_data, df_output, cache=None):
        N = input_data.shape[0]
        return tuple(), df_output.reshape((N, self.n_in, self.n_filters))

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
