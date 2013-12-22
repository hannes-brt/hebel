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

    It does not have any learnable parameters of its own. It should be
    used as the first layer and will perform dropout with any dropout
    probability on the incoming data.

    **Parameters:**

    n_in : integer
        Number of input units.

    dropout_probability : float in [0, 1]
        Probability of dropping out each unit.

    compute_input_gradients : Bool
        Whether to compute the gradients with respect to the input
        data. This only necessary if you're training a model where the
        input itself is learned.

    """

    def __init__(self, n_in, dropout_probability=.2,
                 compute_input_gradients=False):
        self.n_in = n_in
        self.n_units = n_in
        
        assert dropout_probability >= 0. and \
            dropout_probability <= 1.
        self.dropout_probability = dropout_probability
        self.compute_input_gradients = compute_input_gradients

        self.persistent_temp_objects_config = (
            ('dropout_input', ('batch_size', self.n_units), np.float32),
            ('dropout_prob_array', ('batch_size', self.n_units), np.float32),
            ('dropout_mask', ('batch_size', self.n_units), np.int8)
        )

    def feed_forward(self, input_data, prediction=False):
        """Propagate forward through the layer

        **Parameters:**

        input_data : ``GPUArray``
            Inpute data to perform dropout on.

        prediction : bool, optional
            Whether to use prediction model. If true, then the data is
            scaled by ``1 - dropout_probability`` uses dropout.

        **Returns:**
        
        dropout_data : ``GPUArray``
            The data after performing dropout.
        """

        assert input_data.shape[1] == self.n_in

        if not prediction:
            dropout_input = self.get_temp_object('dropout_input',
                input_data.shape, input_data.dtype)
            dropout_prob_array = self.get_temp_object('dropout_prob_array',
                input_data.shape, input_data.dtype)
            dropout_mask = self.get_temp_object('dropout_mask',
                input_data.shape, np.int8)
            sample_dropout_mask(input_data,
                self.dropout_probability, target=dropout_input,
                dropout_prob_array=dropout_prob_array, dropout_mask=dropout_mask)
            return dropout_input, dropout_mask
        else:
            return (input_data * (1 - self.dropout_probability),)

    def backprop(self, input_data, df_output, cache=None):
        """ Backpropagate through the hidden layer

        **Parameters:**

        input_data : ``GPUArray``
            Inpute data to perform dropout on.

        df_output : ``GPUArray``
            Gradients with respect to the output of this layer
            (received from the layer above).

        cache : list of ``GPUArray``
            Cache obtained from forward pass. If the cache is
            provided, then the activations are not recalculated.

        **Returns:**

        gradients : empty tuple
            Gradients are empty since this layer has no parameters.

        df_input : ``GPUArray``
            Gradients with respect to the input.
        """

        if self.compute_input_gradients:            
            apply_dropout_mask(df_output, dropout_mask)

        return tuple(), df_output
