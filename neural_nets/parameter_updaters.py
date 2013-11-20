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

""" Implements different variants of updating the parameters in SGD,
such as momentum and Nesterov momentum.

"""

from pycuda import gpuarray
from itertools import izip


class ParameterUpdater(object):
    def __init__(self, model):
        self.model = model

    def pre_gradient_update(self, stream=None):
        pass

    def post_gradient_update(self, gradients, stream=None):
        pass


class SimpleSGDUpdate(ParameterUpdater):
    def post_gradient_update(self, gradients, batch_size,
                             learning_parameters,
                             stream=None):
        learning_rate = learning_parameters[0]

        multiplier = [-lr_mult * learning_rate / batch_size for lr_mult in
                      self.model.lr_multiplier]
        update = zip(gradients, multiplier)
        self.model.update_parameters(update)


class MomentumUpdate(ParameterUpdater):
    def __init__(self, model):
        self.model = model
        self.velocity = [gpuarray.zeros_like(p)
                         for p in self.model.parameters]

    def post_gradient_update(self, gradients, batch_size,
                             learning_parameters, stream=None):
        learning_rate, momentum = learning_parameters

        updates = []
        for gparam, vparam, lr_multiplier in \
            izip(gradients, self.velocity, self.model.lr_multiplier):
            vparam._axpbyz(momentum,
                           gparam, -learning_rate * lr_multiplier / batch_size,
                           vparam, stream=stream)
            updates.append((vparam, 1.))
        self.model.update_parameters(updates)


class NesterovMomentumUpdate(MomentumUpdate):
    def pre_gradient_update(self):
        """ First step of Nesterov momentum method:
        take step in direction of accumulated gradient
        """

        updates = zip(self.velocity, self.model.n_parameters * [1.])
        self.model.update_parameters(updates)

    def post_gradient_update(self, gradients, batch_size,
                             learning_parameters, stream=None):
        """ Second step of Nesterov momentum method:
        take step in direction of new gradient and update velocity
        """

        learning_rate, momentum = learning_parameters

        updates = []
        for param, gparam, vparam, lr_multiplier in \
          izip(self.model.parameters, gradients,
              self.velocity, self.model.lr_multiplier):

            updates.append(
                (gparam, -learning_rate * lr_multiplier / batch_size))
            # param -= learning_rate*lr_multiplier/batch_size*gparam
            # param._axpbyz(1., gparam, -learning_rate*lr_multiplier/batch_size,
            #               param, stream=stream)
            # vparam = momentum*vparam \
            #    - learning_rate*lr_multiplier/batch_size*gparam
            vparam._axpbyz(momentum, gparam, -learning_rate*lr_multiplier/batch_size,
                           vparam, stream=stream)
        self.model.update_parameters(updates)
