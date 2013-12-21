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
import cPickle
from pycuda import gpuarray
from .dummy_layer import DummyLayer
from ..pycuda_ops.elementwise import sample_dropout_mask, \
    apply_dropout_mask
from ..pycuda_ops.matrix import add_vec_to_mat
from ..pycuda_ops.reductions import matrix_sum_out_axis

class InputDropout(DummyLayer):
    r"""This layer performs dropout on the input data.

    **Parameters:**

    n_in : integer
        Number of input units.

    dropout_probability : float in ``[0, 1]``
        Probability of dropping out each unit.

    compute_input_gradients : Bool
        Whether to compute the gradients with respect to the input
        data. This only necessary if you're training a model where the
        input itself is learned.'

    """

    def __init__(self, n_in, dropout_probability=.2,
                 compute_input_gradients=False):
        self.n_in = n_in
        self.n_units = n_in
        
        assert dropout_probability >= 0. and \
            dropout_probability <= 1.
        self.dropout_probability = dropout_probability
        self.compute_input_gradients = compute_input_gradients

    def feed_forward(self, input_data, prediction=False):
        """Propagate forward through the layer

        **Parameters:**

        input_data : ``GPUArray``
            Inpute data to compute activations for.

        prediction : bool, optional
            Whether to use prediction model. Only relevant when using
            dropout. If true, then weights are halved if the layers
            uses dropout.

        **Returns:**
        
        activations : ``GPUArray``
            The activations of the hidden units.
        """

        assert input_data.shape[1] == self.n_in

        if not prediction:
            dropout_input = gpuarray.empty_like(input_data)
            dropout_mask = sample_dropout_mask(input_data,
                                               self.dropout_probability,
                                               target=dropout_input)
            return dropout_input, dropout_mask
        else:
            return (input_data * (1 - self.dropout_probability),)

    def backprop(self, input_data, df_output, cache=None):
        """ Backpropagate through the hidden layer

        **Parameters:**

        input_data : ``GPUArray``
            Inpute data to compute activations for.

        df_output : ``GPUArray``
            Gradients with respect to the activations of this layer
            (received from the layer above).

        cache : list of ``GPUArray``
            Cache obtained from forward pass. If the cache is
            provided, then the activations are not recalculated.

        **Returns:**

        gradients : tuple of ``GPUArray``
            Gradients with respect to the weights and biases in the
            form ``(df_weights, df_biases)``.

        df_input : ``GPUArray``
            Gradients with respect to the input.
        """

        if self.compute_input_gradients:            
            apply_dropout_mask(df_output, dropout_mask)

        return tuple(), df_output