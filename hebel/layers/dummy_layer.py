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

from .hidden_layer import HiddenLayer


class DummyLayer(HiddenLayer):
    """ This class has no hidden units and simply passes through its
    input
    """

    lr_multiplier = []
    n_parameters = 0
    l1_penalty_weight = 0.
    l2_penalty_weight = 0.
    dropout = False

    def __init__(self, n_in):
        self.n_in = n_in
        self.n_units = n_in

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
        assert input_data.shape[1] == self.n_in
        return (input_data,)

    def backprop(self, input_data, df_output, cache=None):
        return tuple(), df_output
