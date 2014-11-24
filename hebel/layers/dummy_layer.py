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


class DummyLayer(ParameterfreeLayer):
    """ This class has no hidden units and simply passes through its
    input
    """

    dropout = False

    def __init__(self, n_in):
        self.n_in = n_in
        self.n_units = n_in

    def feed_forward(self, input_data, prediction=False):
        assert input_data.shape[1] == self.n_in
        return (input_data,)

    def backprop(self, input_data, df_output, cache=None):
        return tuple(), df_output
