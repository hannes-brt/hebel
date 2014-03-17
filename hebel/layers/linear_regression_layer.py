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
from .. import sampler
from .softmax_layer import SoftmaxLayer
from ..pycuda_ops.elementwise import sign, nan_to_zeros
from ..pycuda_ops.reductions import matrix_sum_out_axis
from ..pycuda_ops.matrix import add_vec_to_mat
from ..pycuda_ops import linalg


class LinearRegressionLayer(SoftmaxLayer):
    r"""Linear regression layer with linear outputs and squared loss error function.

        **Parameters:**
    
    n_in : integer
        Number of input units.

    n_out : integer
        Number of output units (classes).

    parameters : array_like of ``GPUArray``
        Parameters used to initialize the layer. If this is omitted,
        then the weights are initalized randomly using *Bengio's rule*
        (uniform distribution with scale :math:`4 \cdot \sqrt{6 /
        (\mathtt{n\_in} + \mathtt{n\_out})}`) and the biases are
        initialized to zero. If ``parameters`` is given, then is must
        be in the form ``[weights, biases]``, where the shape of
        weights is ``(n_in, n_out)`` and the shape of ``biases`` is
        ``(n_out,)``. Both weights and biases must be ``GPUArray``.
    
    weights_scale : float, optional
        If ``parameters`` is omitted, then this factor is used as
        scale for initializing the weights instead of *Bengio's rule*.

    l1_penalty_weight : float, optional
        Weight used for L1 regularization of the weights.

    l2_penalty_weight : float, optional
       Weight used for L2 regularization of the weights.

    lr_multiplier : float, optional
        If this parameter is omitted, then the learning rate for the
        layer is scaled by :math:`2 / \sqrt{\mathtt{n\_in}}`. You may
        specify a different factor here.

    test_error_fct : {``class_error``, ``kl_error``, ``cross_entropy_error``}, optional
        Which error function to use on the test set. Default is
        ``class_error`` for classification error. Other choices are
        ``kl_error``, the Kullback-Leibler divergence, or
        ``cross_entropy_error``.

    **See also:**

    :class:`hebel.models.NeuralNetRegression`,
    :class:`hebel.models.NeuralNet`,
    :class:`hebel.layers.LogisticLayer`

    """

    
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

        self.persistent_temp_objects_config = (
            ('activations', ('batch_size', self.n_out), np.float32),
            ('df_W', self.W.shape, self.W.dtype),
            ('df_b', self.b.shape, self.b.dtype),
            ('df_input', ('batch_size', self.n_in), np.float32),
            ('delta', ('batch_size', self.n_out), np.float32)
        )

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

        activations = self.get_temp_object('activations',
            (input_data.shape[0], self.n_out), input_data.dtype)
        
        linalg.dot(input_data, self.W, target=activations)
        activations = add_vec_to_mat(activations, self.b, inplace=True)

        return activations

    def test_error(self, input_data, targets, average=True,
                   cache=None, prediction=True):
        """Compute the test error function given some data and targets.

        Uses the error function defined in
        :class:`SoftmaxLayer.test_error_fct`, which may be different
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
