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
from itertools import izip
from pycuda import gpuarray
from .top_layer import TopLayer
from .logistic_layer import LogisticLayer


class MultitaskTopLayer(TopLayer):

    def __init__(self, n_in=None, n_out=None, test_error_fct='class_error',
                 l1_penalty_weight=0., l2_penalty_weight=0.,
                 tasks=None, task_weights=None, n_tasks=None,
                 lr_multiplier=None):
        """ Inputs:
        n_in: number of input units (size of last hidden layer)
        n_out: sequence of output sizes for the targets
        test_error_fct: name of test error function
        l1_penalty_weight: scalar or sequence of l1 penalty weights
        l2_penalty_weight: scalar or sequence of l2 penalty weights
        tasks: sequence of TopLayer objects; overrides all_other parameters
        """

        if tasks is None and (n_in is None or n_out is None):
            raise ValueError('Either `tasks` or `n_in` and `n_out` ' +
                             'must be provided')

        if not tasks:
            self.n_in = n_in
            self.n_out = n_out if n_tasks is None else n_tasks * [n_out]
            # Number of output tasks
            self.n_tasks = n_tasks if n_tasks is not None else len(n_out)
            self.tasks = []

            if not isinstance(test_error_fct, (list, tuple)):
                test_error_fct = self.n_tasks * [test_error_fct]
            if not isinstance(l1_penalty_weight, (list, tuple)):
                l1_penalty_weight = self.n_tasks * [l1_penalty_weight]
            if not isinstance(l2_penalty_weight, (list, tuple)):
                l2_penalty_weight = self.n_tasks * [l2_penalty_weight]

            for (n_out_task, test_error_task, l1_task, l2_task) in \
              zip(self.n_out, test_error_fct,
                  l1_penalty_weight, l2_penalty_weight):
                self.tasks.append(LogisticLayer(n_in=n_in,
                                                n_out=n_out_task,
                                                l1_penalty_weight=l1_task,
                                                l2_penalty_weight=l2_task,
                                                test_error_fct=test_error_task,
                                                lr_multiplier=lr_multiplier))

        else:
            assert all([self.tasks[0].n_in == t.n_in for t in tasks])
            self.tasks = tasks

            self.n_in = self.tasks[0].n_in
            self.n_out = [t.n_out for t in self.tasks]

        if task_weights is not None:
            self.task_weights = task_weights
        else:
            self.task_weights = self.n_tasks * [1.]

        self.l1_penalty_weight = l1_penalty_weight
        self.l2_penalty_weight = l2_penalty_weight

        self.n_parameters = sum(task.n_parameters for task in self.tasks)
        self.lr_multiplier = [lr for task in self.tasks
                              for lr in task.lr_multiplier]

    @property
    def parameters(self):
        parameters = []
        for task in self.tasks:
            parameters.extend(task.parameters)
        return parameters

    @parameters.setter
    def parameters(self, value):
        assert len(value) == self.n_parameters

        i = 0
        for task in self.tasks:
            task.parameters = value[i:i + task.n_parameters]
            i += task.n_parameters

    def update_parameters(self, value):
        assert len(value) == self.n_parameters
        i = 0
        for task in self.tasks:
            task.update_parameters(value[i:i + task.n_parameters])
            i += task.n_parameters

    @property
    def architecture(self):
        return [task.architecture for task in self.tasks]

    @property
    def l1_penalty(self):
        return sum([task.l1_penalty for task in self.tasks])

    @property
    def l2_penalty(self):
        return sum([task.l2_penalty for task in self.tasks])

    def feed_forward(self, input_data, prediction=False):
        activations = []

        for task in self.tasks:
            activations_task = task.feed_forward(input_data, prediction)
            activations.append(activations_task)

        return activations

    def backprop(self, input_data, targets, cache=None):
        df_input = gpuarray.zeros_like(input_data)

        if cache is None: cache = self.n_tasks * [None]

        gradients = []
        for targets_task, cache_task, task, task_weight  in \
          izip(targets, cache, self.tasks, self.task_weights):
            gradients_task, df_input_task = \
              task.backprop(input_data, targets_task,
                            cache_task)

            df_input.mul_add(1., df_input_task, task_weight)

            gradients.extend(gradients_task)

        return gradients, df_input

    def test_error(self, input_data, targets, average=True,
                   cache=None, prediction=False,
                   sum_errors=True):

        test_error = []
        if cache is None:
            cache = self.n_tasks * [None]
        for targets_task, cache_task, task in \
            izip(targets, cache, self.tasks):
            test_error.append(task.test_error(input_data, targets_task,
                                              average, cache_task,
                                              prediction))

        if sum_errors:
            return sum(test_error)
        else:
            return np.array(test_error)

    def cross_entropy_error(self, input_data, targets, average=True,
                            cache=None, prediction=False,
                            sum_errors=True):
        """ Return the cross entropy error
        """

        loss = []
        if cache is None:
            cache = self.n_tasks * [None]

        for targets_task, cache_task, task in \
            izip(targets, cache, self.tasks):
            loss.append(task.cross_entropy_error(
                input_data, targets_task, average=average,
                cache=cache_task,
                prediction=prediction))

        if sum_errors:
            return sum(loss)
        else:
            return loss
