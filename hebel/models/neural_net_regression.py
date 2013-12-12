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

from .neural_net import NeuralNet
from ..layers import LinearRegressionLayer

class NeuralNetRegression(NeuralNet):
    """A neural network for regression using the squared error loss
    function.

    This class exists for convenience. The same results can be
    achieved by creating a :class:`hebel.models.NeuralNet` instance
    and passing a :class:`hebel.layers.LinearRegressionLayer` instance
    as the ``top_layer`` argument.

    **Parameters:**

    layers : array_like
        An array of either integers or instances of
        :class:`hebel.models.HiddenLayer` objects. If integers are
        given, they represent the number of hidden units in each layer
        and new ``HiddenLayer`` objects will be created. If
        ``HiddenLayer`` instances are given, the user must make sure
        that each ``HiddenLayer`` has ``n_in`` set to the preceding
        layer's ``n_units``. If ``HiddenLayer`` instances are passed,
        then ``activation_function``, ``dropout``, ``n_in``,
        ``l1_penalty_weight``, and ``l2_penalty_weight`` are ignored.

    top_layer : :class:`hebel.models.TopLayer` instance, optional
        If ``top_layer`` is given, then it is used for the output
        layer, otherwise, a ``LinearRegressionLayer`` instance is created.

    activation_function : {'sigmoid', 'tanh', 'relu', or 'linear'}, optional
        The activation function to be used in the hidden layers.

    dropout : bool, optional
        Whether to use dropout regularization

    n_in : integer, optional
        The dimensionality of the input. Must be given, if the first
        hidden layer is not passed as a
        :class:`hebel.models.HiddenLayer` instance.

    n_out : integer, optional
        The number of classes to predict from. Must be given, if a
        :class:`hebel.models.HiddenLayer` instance is not given in
        ``top_layer``.

    l1_penalty_weight : float, optional
        Weight for L1 regularization

    l2_penalty_weight : float, optional
        Weight for L2 regularization

    kwargs : optional
        Any additional arguments are passed on to ``top_layer``

    **See also:**
    
    :class:`hebel.models.NeuralNet`,
    :class:`hebel.models.MultitaskNeuralNet`,
    :class:`hebel.layers.LinearRegressionLayer`

    """
    TopLayerClass = LinearRegressionLayer
