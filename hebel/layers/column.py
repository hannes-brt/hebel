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

from . import HiddenLayer
from itertools import chain

class Column(object):
    l1_penalty_weight = True
    l2_penalty_weight = True

    def __init__(self, hidden_layers):
        assert all([isinstance(hl, HiddenLayer) for hl in hidden_layers])
        self.hidden_layers = hidden_layers

    @property
    def n_parameters(self):
        return sum(hl.n_parameters for hl in self.hidden_layers)

    @property
    def n_units(self):
        return self.hidden_layers[-1].n_units

    @property
    def n_in(self):
        return self.hidden_layers[0].n_in

    @property
    def parameters(self):
        return list(chain.from_iterable(hl.parameters for hl in self.hidden_layers))

    @parameters.setter
    def parameters(self, new_parameters):
        for hl in self.hidden_layers:
            hl.parameters = new_parameters[:hl.n_parameters]
            new_parameters = new_parameters[hl.n_parameters:]

    def update_parameters(self, values, stream=None):
        assert len(values) == self.n_parameters

        for hl in self.hidden_layers:
            hl.update_parameters(values[:hl.n_parameters])
            values = values[hl.n_parameters:]

    @property
    def l1_penalty(self):
        return sum(hl.l1_penalty for hl in self.hidden_layers)

    @property
    def l2_penalty(self):
        return sum(hl.l2_penalty for hl in self.hidden_layers)

    @property
    def lr_multiplier(self):
        return tuple(chain.from_iterable((hl.lr_multiplier for hl in self.hidden_layers)))

    @lr_multiplier.setter
    def lr_multiplier(self, value):
        assert self.n_parameters == len(value)
        i = 0
        for hl in self.hidden_layers:
            hl.lr_multiplier = value[i:i+hl.n_parameters]
            i += hl.n_parameters

    def feed_forward(self, input_data, prediction=False):
        cache = []
        activations = [input_data]
        a = input_data
        for hl in self.hidden_layers:
            c = hl.feed_forward(a, prediction)
            a = c[0]
            activations.append(c[0])
            cache.append(c)

        del activations[-1]
        return a, (activations, cache)

    def backprop(self, input_data, df_output, cache=None):
        if cache is None:
            _, (activations, cache) = self.feed_forward(input_data, False)
        else:
            _, (activations, cache) = cache

        df_param = []
        df_input = df_output
        for hl, a, c in zip(self.hidden_layers[::-1], activations[::-1], cache[::-1]):
            df_p, df_input = hl.backprop(a, df_input, c)
            df_param.append(df_p)

        df_param.reverse()
        df_param = list(chain.from_iterable(df_param))

        return df_param, df_input