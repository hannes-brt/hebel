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
from pycuda import cumath
from math import sqrt
from .. import sampler
from .top_layer import TopLayer
from ..pycuda_ops import eps, linalg
from ..pycuda_ops.elementwise import sign, nan_to_zeros, substract_matrix, sigmoid
from ..pycuda_ops.reductions import matrix_sum_out_axis
from ..pycuda_ops.matrix import add_vec_to_mat
from ..pycuda_ops.softmax import cross_entropy_logistic


class LogisticLayer(TopLayer):
    r""" A logistic classification layer for two classes, using
    cross-entropy loss function and sigmoid activations.

    **Parameters:**
    
    n_in : integer
        Number of input units.

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

    :class:`hebel.layers.SoftmaxLayer`,
    :class:`hebel.models.NeuralNet`,
    :class:`hebel.models.NeuralNetRegression`,
    :class:`hebel.layers.LinearRegressionLayer`

    **Examples**::

        # Use the simple initializer and initialize with random weights
        logistic_layer = LogisticLayer(1000)

        # Sample weights yourself, specify an L1 penalty, and don't
        # use learning rate scaling
        import numpy as np
        from pycuda import gpuarray

        n_in = 1000
        weights = gpuarray.to_gpu(.01 * np.random.randn(n_in, 1))
        biases = gpuarray.to_gpu(np.zeros((1,)))
        softmax_layer = SoftmaxLayer(n_in,
                                     parameters=(weights, biases),
                                     l1_penalty_weight=.1,
                                     lr_multiplier=1.)
    """

    n_parameters = 2
    n_out = 1

    def __init__(self, n_in,
                 parameters=None,
                 weights_scale=None,
                 l1_penalty_weight=0., l2_penalty_weight=0.,
                 lr_multiplier=None,
                 test_error_fct='class_error'):

        # Initialize weight using Bengio's rule
        self.weights_scale = 4 * sqrt(6. / (n_in + 1)) \
                             if weights_scale is None \
                                else weights_scale

        if parameters is not None:
            self.W, self.b = parameters
        else:
            self.W = self.weights_scale * \
                     sampler.gen_uniform((n_in, 1), dtype=np.float32) \
                     - .5 * self.weights_scale

            self.b = gpuarray.zeros((1,), dtype=np.float32)

        self.n_in = n_in

        self.test_error_fct = test_error_fct

        self.l1_penalty_weight = l1_penalty_weight
        self.l2_penalty_weight = l2_penalty_weight

        self.lr_multiplier = 2 * [1. / np.sqrt(n_in, dtype=np.float32)] \
          if lr_multiplier is None else lr_multiplier

        self.persistent_temp_objects_config = (
            ('activations', ('batch_size', 1), np.float32),
            ('df_W', self.W.shape, np.float32),
            ('df_b', self.b.shape, np.float32),
            ('df_input', ('batch_size', self.n_in), np.float32),
            ('delta', ('batch_size', 1), np.float32)
        )

    @property
    def architecture(self):
        return {'class': self.__class__,
                'n_in': self.n_in,
                'n_out': 1}

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
            (input_data.shape[0], 1), input_data.dtype)
        linalg.dot(input_data, self.W, target=activations)
        activations = add_vec_to_mat(activations, self.b, inplace=True)

        sigmoid(activations)

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

        # Get temporary objects
        df_W = self.get_temp_object('df_W', self.W.shape, self.W.dtype)
        df_b = self.get_temp_object('df_b', self.b.shape, self.b.dtype)
        df_input = self.get_temp_object('df_input',
                input_data.shape, input_data.dtype)
        delta = self.get_temp_object('delta',
                activations.shape, activations.dtype)

        substract_matrix(activations, targets, delta)
        nan_to_zeros(delta, delta)

        # Gradient wrt weights
        linalg.dot(input_data, delta, transa='T', target=df_W)
        # Gradient wrt bias
        matrix_sum_out_axis(delta, 0, target=df_b)

        # Gradient wrt input
        linalg.dot(delta, self.W, transb='T', target=df_input)

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
        if self.test_error_fct == 'class_error':
            test_error = self.class_error
        elif self.test_error_fct == 'cross_entropy_error':
            test_error = self.cross_entropy_error
        else:
            raise ValueError('unknown test error function "%s"'
                             % self.test_error_fct)

        return test_error(input_data, targets, average,
                          cache, prediction)

    def cross_entropy_error(self, input_data, targets, average=True,
                            cache=None, prediction=False):
        """ Return the cross entropy error
        """

        if cache is not None:
            activations = cache
        else:
            activations = \
              self.feed_forward(input_data, prediction=prediction)

        loss = cross_entropy_logistic(activations, targets)

        if average: loss /= targets.shape[0]
        return loss
        
    train_error = cross_entropy_error

    def class_error(self, input_data, targets, average=True,
                    cache=None, prediction=False):
        """ Return the classification error rate
        """

        if cache is not None:
            activations = cache
        else:
            activations = \
              self.feed_forward(input_data, prediction=prediction)

        targets = targets.get()
        class_error = np.sum((activations.get() >= .5) != (targets >= .5))

        if average: class_error = float(class_error) / targets.shape[0]
        return class_error
