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
from hashlib import md5
from ..layers import HiddenLayer, TopLayer, SoftmaxLayer, LogisticLayer, InputDropout
from .model import Model


class NeuralNet(Model):
    """ A neural network for classification using the cross-entropy
    loss function.

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
        layer, otherwise, a ``LogisticLayer`` instance is created.

    activation_function : {'sigmoid', 'tanh', 'relu', or 'linear'}, optional
        The activation function to be used in the hidden layers.

    dropout : bool, optional
        Whether to use dropout regularization

    input_dropout : float, in ``[0, 1]``
        Dropout probability for the input (default 0.0).

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
    
    :class:`hebel.models.LogisticRegression`,
    :class:`hebel.models.NeuralNetRegression`,
    :class:`hebel.models.MultitaskNeuralNet`

    **Examples**::

        # Simple form
        model = NeuralNet(layers=[1000, 1000],
                          activation_function='relu',
                          dropout=True,
                          n_in=784, n_out=10,
                          l1_penalty_weight=.1)

        # Extended form, initializing with ``HiddenLayer`` and ``TopLayer`` objects
        hidden_layers = [HiddenLayer(784, 1000, 'relu', dropout=True,
                                     l1_penalty_weight=.2),
                         HiddenLayer(1000, 1000, 'relu', dropout=True,
                                     l1_penalty_weight=.1)]
        softmax_layer = LogisticLayer(1000, 10, l1_penalty_weight=.1)

        model = NeuralNet(hidden_layers, softmax_layer)
    """

    TopLayerClass = SoftmaxLayer

    def __init__(self, layers, top_layer=None, activation_function='sigmoid',
                 dropout=False, input_dropout=0., n_in=None, n_out=None,
                 l1_penalty_weight=0., l2_penalty_weight=0.,
                 **kwargs):
        self.n_layers = len(layers)
        if n_out == 1 and self.TopLayerClass == SoftmaxLayer:
            self.TopLayerClass = LogisticLayer

        if l1_penalty_weight is not None and \
           not np.isscalar(l1_penalty_weight) and \
           len(l1_penalty_weight) != (self.n_layers + 1):
            raise ValueError("l1_penalty_weight must be a scalar "
                             "or have length %d",
                             self.n_layers + 1)

        if l2_penalty_weight is not None and \
           not np.isscalar(l2_penalty_weight) and \
           len(l2_penalty_weight) != (self.n_layers + 1):
            raise ValueError("l2_penalty_weight must be a scalar "
                             "or have length %d",
                             self.n_layers + 1)

        if np.isscalar(l1_penalty_weight):
            self.l1_penalty_weight_hidden = self.n_layers * [l1_penalty_weight]
            self.l1_penalty_weight_output = l1_penalty_weight
        else:
            self.l1_penalty_weight_hidden = l1_penalty_weight[:-1]
            self.l1_penalty_weight_output = l1_penalty_weight[-1]

        if np.isscalar(l2_penalty_weight):
            self.l2_penalty_weight_hidden = self.n_layers * [l2_penalty_weight]
            self.l2_penalty_weight_output = l2_penalty_weight
        else:
            self.l2_penalty_weight_hidden = l2_penalty_weight[:-1]
            self.l2_penalty_weight_output = l2_penalty_weight[-1]

        if type(dropout) is not list:
            if self.n_layers:
                dropout = self.n_layers * [dropout]
            else:
                dropout = [False]

        self.hidden_layers = []

        self.input_dropout = input_dropout
        if input_dropout:
            self.hidden_layers.append(InputDropout(n_in, input_dropout))

        for i, hidden_layer in enumerate(layers):
            if isinstance(hidden_layer, HiddenLayer):
                self.hidden_layers.append(hidden_layer)
            elif isinstance(hidden_layer, int):
                n_in_hidden = self.hidden_layers[-1].n_units if self.hidden_layers else n_in
                self.hidden_layers.append(
                    HiddenLayer(
                        n_in_hidden, hidden_layer,
                        activation_function,
                        dropout=dropout[i],
                        l1_penalty_weight=self.l1_penalty_weight_hidden[i],
                        l2_penalty_weight=self.l2_penalty_weight_hidden[i]))

        self.n_units_hidden = [hl.n_units for hl in self.hidden_layers]

        if top_layer is None:
            assert issubclass(self.TopLayerClass, TopLayer)
            n_in_top_layer = self.n_units_hidden[-1] \
                             if self.n_units_hidden else n_in
            self.top_layer = self.TopLayerClass(
                n_in_top_layer, n_out,
                l1_penalty_weight=self.l1_penalty_weight_output,
                l2_penalty_weight=self.l2_penalty_weight_output,
                **kwargs)
        else:
            self.top_layer = top_layer

        self.n_in = self.hidden_layers[0].n_in if self.hidden_layers else n_in
        self.n_out = self.top_layer.n_out 

        self.n_parameters = sum(hl.n_parameters
                                for hl in self.hidden_layers) + \
                                    self.top_layer.n_parameters

        self.lr_multiplier = [lr for hl in
                              self.hidden_layers + [self.top_layer]
                              for lr in hl.lr_multiplier]
    
    def preallocate_temp_objects(self, data_provider):
        for hl in self.hidden_layers:
            if hasattr(hl, 'preallocate_temp_objects'):
                hl.preallocate_temp_objects(data_provider)
        if hasattr(self.top_layer, 'preallocate_temp_objects'):
            self.top_layer.preallocate_temp_objects(data_provider)
    
    @property
    def parameters(self):
        """ A property that returns all of the model's parameters. """
        parameters = []
        for hl in self.hidden_layers:
            parameters.extend(hl.parameters)
        parameters.extend(self.top_layer.parameters)
        return parameters

    @parameters.setter
    def parameters(self, value):
        """ Used to set all of the model's parameters to new values.

        **Parameters:**

        value : array_like
            New values for the model parameters. Must be of length
            ``self.n_parameters``.
        """
    
        if len(value) != self.n_parameters:
            raise ValueError("Incorrect length of parameter vector. "
                             "Model has %d parameters, but got %d" %
                             (self.n_parameters, len(value)))

        i = 0
        for hl in self.hidden_layers:
            hl.parameters = value[i:i + hl.n_parameters]
            i += hl.n_parameters

        self.top_layer.parameters = value[-self.top_layer.n_parameters:]

    def update_parameters(self, value):
        assert len(value) == self.n_parameters

        i = 0
        for hl in self.hidden_layers:
            hl.update_parameters(value[i:i + hl.n_parameters])
            i += hl.n_parameters

        self.top_layer.update_parameters(value[-self.top_layer.n_parameters:])

    def checksum(self):
        """ Returns an MD5 digest of the model.

        This can be used to easily identify whether two models have the
        same architecture.
        """
        
        m = md5()
        for hl in self.hidden_layers:
            m.update(str(hl.architecture))
        m.update(str(self.top_layer.architecture))
        return m.hexdigest()

    def evaluate(self, input_data, targets,
                 return_cache=False, prediction=True):
        """ Evaluate the loss function without computing gradients.

        **Parameters:**

        input_data : GPUArray
            Data to evaluate

        targets: GPUArray
            Targets

        return_cache : bool, optional
            Whether to return intermediary variables from the
            computation and the hidden activations.

        prediction : bool, optional
            Whether to use prediction model. Only relevant when using
            dropout. If true, then weights are halved in layers that
            use dropout.

        **Returns:**

        loss : float
            The value of the loss function.

        hidden_cache : list, only returned if ``return_cache == True``
            Cache as returned by :meth:`hebel.models.NeuralNet.feed_forward`.

        activations : list, only returned if ``return_cache == True``
            Hidden activations as returned by
            :meth:`hebel.models.NeuralNet.feed_forward`.
        """

        # Forward pass
        activations, hidden_cache = self.feed_forward(
            input_data, return_cache=True, prediction=prediction)

        loss = self.top_layer.train_error(None,
            targets, average=False, cache=activations,
            prediction=prediction)

        for hl in self.hidden_layers:
            if hl.l1_penalty_weight: loss += hl.l1_penalty
            if hl.l2_penalty_weight: loss += hl.l2_penalty

        if self.top_layer.l1_penalty_weight: loss += self.top_layer.l1_penalty
        if self.top_layer.l2_penalty_weight: loss += self.top_layer.l2_penalty

        if not return_cache:
            return loss
        else:
            return loss, hidden_cache, activations

    def training_pass(self, input_data, targets):
        """ Perform a full forward and backward pass through the model.

        **Parameters:**

        input_data : GPUArray
            Data to train the model with.

        targets : GPUArray
            Training targets.

        **Returns:**

        loss : float
            Value of loss function as evaluated on the data and targets.

        gradients : list of GPUArray
            Gradients obtained from backpropagation in the backward pass.
        """

        # Forward pass
        loss, hidden_cache, logistic_cache = self.evaluate(
            input_data, targets, return_cache=True, prediction=False)

        # Backpropagation
        if self.hidden_layers:
            hidden_activations = hidden_cache[-1][0]
        else:
            hidden_activations = input_data

        df_top_layer = \
          self.top_layer.backprop(hidden_activations, targets,
                                  cache=logistic_cache)
        gradients = list(df_top_layer[0][::-1])
        df_hidden = df_top_layer[1]

        if self.hidden_layers:
            hidden_inputs = [input_data] + [c[0] for c in hidden_cache[:-1]]            
            for hl, hc, hi in \
                zip(self.hidden_layers[::-1], hidden_cache[::-1],
                    hidden_inputs[::-1]):
                g, df_hidden = hl.backprop(hi, df_hidden, cache=hc)
                gradients.extend(g[::-1])

        gradients.reverse()

        return loss, gradients

    def test_error(self, test_data, average=True):
        """ Evaulate performance on a test set.

        **Parameters:**

        test_data : :class:``hebel.data_provider.DataProvider``
            A ``DataProvider`` instance to evaluate on the model.

        average : bool, optional
            Whether to divide the loss function by the number of
            examples in the test data set.

        **Returns:**

        test_error : float
        """

        test_error = 0.
        for batch_data, batch_targets in test_data:
            _, hidden_cache, logistic_cache = \
              self.evaluate(batch_data, batch_targets,
                            return_cache=True,
                            prediction=True)

            if self.hidden_layers:
                hidden_activations = hidden_cache[-1]
            else:
                hidden_activations = batch_data

            test_error += self.top_layer.test_error(hidden_activations,
                                                    batch_targets, average=False,
                                                    cache=logistic_cache,
                                                    prediction=True)

        if average: test_error /= float(test_data.N)

        return test_error

    def feed_forward(self, input_data, return_cache=False, prediction=True):
        """ Run data forward through the model.

        **Parameters:**

        input_data : GPUArray
            Data to run through the model.

        return_cache : bool, optional
            Whether to return the intermediary results.

        prediction : bool, optional
            Whether to run in prediction mode. Only relevant when
            using dropout. If true, weights are halved. If false, then
            half of hidden units are randomly dropped and the dropout
            mask is returned in case ``return_cache==True``.

        **Returns:**
        
        prediction : GPUArray
            Predictions from the model.

        cache : list of GPUArray, only returned if ``return_cache == True``
            Results of intermediary computations.    
        """

        hidden_cache = None     # Create variable in case there are no hidden layers
        if self.hidden_layers:
            # Forward pass
            hidden_cache = []
            for i in range(len(self.hidden_layers)):
                hidden_activations = hidden_cache[i - 1][0] if i else input_data
                # Use dropout predict if previous layer has dropout
                hidden_cache.append(self.hidden_layers[i]
                                    .feed_forward(hidden_activations,
                                                  prediction=prediction))

            hidden_activations = hidden_cache[-1][0]

        else:
            hidden_activations = input_data

        # Use dropout_predict if last hidden layer has dropout
        activations = \
          self.top_layer.feed_forward(hidden_activations,
                                      prediction=False)

        if return_cache:
            return activations, hidden_cache
        return activations
