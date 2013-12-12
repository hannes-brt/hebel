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
from pycuda import gpuarray, cumath
from math import sqrt
from scikits.cuda import linalg
from .. import sampler
from .logistic_layer import LogisticLayer
from ..pycuda_ops.elementwise import sign, nan_to_zeros
from ..pycuda_ops.reductions import matrix_sum_out_axis
from ..pycuda_ops.matrix import add_vec_to_mat


class LinearRegressionLayer(LogisticLayer):
    n_parameters = 2
    
    def __init__(self, n_in, n_out,
                 parameters=None,
                 weights_scale=None,
                 l1_penalty_weight=0.,
                 l2_penalty_weight=0.,
                 lr_multiplier=None):

        # Initialize weight using Bengio's rule
        self.weights_scale = 4 * sqrt(6. / (n_in + n_out)) \
                             if weights_scale is None \
                                else weights_scale

        if parameters is not None:
            self.W, self.b = parameters
        else:
            self.W = self.weights_scale * \
                     sampler.gen_uniform((n_in, n_out), dtype=np.float32) \
                     - .5 * self.weights_scale

            self.b = gpuarray.zeros((n_out,), dtype=np.float32)

        self.n_in = n_in
        self.n_out = n_out

        self.l1_penalty_weight = l1_penalty_weight
        self.l2_penalty_weight = l2_penalty_weight

        self.lr_multiplier = 2 * [1. / np.sqrt(n_in, dtype=np.float32)] \
          if lr_multiplier is None else lr_multiplier

    def feed_forward(self, input_data, prediction=False):
        """Propagate forward through the layer.

        **Parameters:**

        input_data : ``GPUArray``
            Inpute data to compute activations for.

        prediction : bool, optional
            Whether to use prediction model. Only relevant when using
            dropout. If true, then weights are halved if the layers
            uses dropout.

        **Returns:**
        
        activations : ``GPUArray``
            The activations of the output units.
        """
        activations = linalg.dot(input_data, self.W)
        activations = add_vec_to_mat(activations, self.b, inplace=True)

        return activations

    def backprop(self, input_data, targets,
                 cache=None):
        """ Backpropagate through the logistic layer.

        **Parameters:**

        input_data : ``GPUArray``
            Inpute data to compute activations for.

        targets : ``GPUArray``
            The target values of the units.

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

        if cache is not None:
            activations = cache
        else:
            activations = self.feed_forward(input_data, prediction=False)

        delta = activations - targets
        nan_to_zeros(delta, delta)
            
        # Gradient wrt weights
        df_W = linalg.dot(input_data, delta, transa='T')
        # Gradient wrt bias
        df_b = matrix_sum_out_axis(delta, 0)

        # Gradient wrt input
        df_input = linalg.dot(delta, self.W, transb='T')

        # L1 penalty
        if self.l1_penalty_weight:
            df_W -= self.l1_penalty_weight * sign(self.W)

        # L2 penalty
        if self.l2_penalty_weight:
            df_W -= self.l2_penalty_weight * self.W

        return (df_W, df_b), df_input

    def test_error(self, input_data, targets, average=True,
                   cache=None, prediction=True):
        """Compute the test error function given some data and targets.

        Uses the error function defined in
        :class:`LogisticLayer.test_error_fct`, which may be different
        from the cross-entropy error function used for
        training'. Alternatively, the other test error functions may
        be called directly.

        **Parameters:**

        input_data : ``GPUArray``
            Inpute data to compute the test error function for.

        targets : ``GPUArray``
            The target values of the units.

        average : bool
            Whether to divide the value of the error function by the
            number of data points given.

        cache : list of ``GPUArray``
            Cache obtained from forward pass. If the cache is
            provided, then the activations are not recalculated.

        prediction : bool, optional
            Whether to use prediction model. Only relevant when using
            dropout. If true, then weights are halved if the layers
            uses dropout.

        **Returns:**
        test_error : float
        """

        return self.squared_loss(input_data, targets, average,
                                 cache, prediction)

    def squared_loss(self, input_data, targets, average=True,
                     cache=None, prediction=False):
        if cache is not None:
            activations = cache
        else:
            activations  = \
                self.feed_forward(input_data, prediction=prediction)

        loss = gpuarray.sum(
            matrix_sum_out_axis((targets - activations) ** 2, 1))

        if average: loss = loss.mean()
        return float(loss.get())
    train_error = squared_loss
