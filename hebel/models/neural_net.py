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
from .hidden_layer import HiddenLayer
from .top_layer import TopLayer
from .logistic_layer import LogisticLayer
from .model import Model


class NeuralNet(Model):
    """ A Neural Network Object
    """

    TopLayerClass = LogisticLayer

    def __init__(self, layers, top_layer=None, activation_function='sigmoid',
                 dropout=False, n_in=None, n_out=None,
                 l1_penalty_weight=0., l2_penalty_weight=0.,
                 **kwargs):
        self.n_layers = len(layers)

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
        for i, hidden_layer in enumerate(layers):
            if isinstance(hidden_layer, HiddenLayer):
                self.hidden_layers.append(hidden_layer)
            elif isinstance(hidden_layer, int):
                n_in_hidden = self.hidden_layers[-1].n_units if i > 0 else n_in
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

        self.n_in = self.hidden_layers[0].n_in
        self.n_out = self.top_layer.n_out

        self.n_parameters = sum(hl.n_parameters
                                for hl in self.hidden_layers) + \
                                    self.top_layer.n_parameters

        self.lr_multiplier = [lr for hl in
                              self.hidden_layers + [self.top_layer]
                              for lr in hl.lr_multiplier]

    @property
    def parameters(self):
        # Gather the parameters
        parameters = []
        for hl in self.hidden_layers:
            parameters.extend(hl.parameters)
        parameters.extend(self.top_layer.parameters)
        return parameters

    @parameters.setter
    def parameters(self, value):
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

    @property
    def checksum(self):
        m = md5()
        for hl in self.hidden_layers:
            m.update(str(hl.architecture))
        m.update(str(self.top_layer.architecture))
        return m.hexdigest()

    def evaluate(self, input_data, targets,
                 return_cache=False, prediction=True):
        """ Evaluate the loss function without computing gradients
        """

        # Forward pass
        activations, hidden_cache = self.feed_forward(
            input_data, return_cache=True, prediction=prediction)

        loss = self.top_layer.cross_entropy_error(None,
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
        """ Perform a full forward and backward pass through the model
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

        hidden_inputs = [input_data] + [c[0] for c in hidden_cache[:-1]]
        for hl, hc, hi in \
            zip(self.hidden_layers[::-1], hidden_cache[::-1],
                hidden_inputs[::-1]):
            g, df_hidden = hl.backprop(hi, df_hidden, cache=hc)
            gradients.extend(g[::-1])

        gradients.reverse()

        return loss, gradients

    def test_error(self, test_data, average=True):
        """ Evaulate performance on a test set

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
        """ Get predictions from the model
        """

        if self.hidden_layers:
            # Forward pass
            hidden_cache = []
            # Input layer never has dropout
            hidden_cache.append(self.hidden_layers[0].feed_forward(input_data,
                                                                   prediction))

            for i in range(1, self.n_layers):
                hidden_activations = hidden_cache[i - 1][0]
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
                                      prediction=prediction)

        if return_cache:
            return activations, hidden_cache
        return activations
