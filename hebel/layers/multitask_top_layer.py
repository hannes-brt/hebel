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
from .softmax_layer import SoftmaxLayer


class MultitaskTopLayer(TopLayer):
    """Top layer for performing multi-task training.

    This is a top layer that enables multi-task training, which
    can be thought of as training multiple models on the same data
    and sharing weights in all but the final layer. A
    ``MultitaskTopLayer`` has multiple layers as children that are
    subclasses of :class:`hebel.layers.TopLayer`. During the
    forward pass, the input from the previous layer is passed on
    to all tasks and during backpropagation, the gradients are
    added together from the different tasks (with different
    weights if necessary).

    There are two ways of initializing ``MultitaskTopLayer``:

    1. By supplying ``n_in``, ``n_out``, and optionally
       ``n_tasks``, which will initialize all tasks with
       :class:`hebel.layers.LogisticLayer`. If ``n_tasks`` is
       given, ``n_out`` must be an integer and ``n_tasks`` identical
       tasks will be created. If ``n_out`` is an ``array_like``, then
       as many tasks will be created as there are elements in
       ``n_out`` and ``n_tasks`` will be ignored.

    2. If ``tasks`` is supplied, then it must be an ``array_like``
       of objects derived from :class:`hebel.layers.TopLayer`, one
       object for each class. In this case ``n_in``, ``n_out``, and
       ``n_tasks`` will be ignored. The user must make sure that all
       tasks have their ``n_in`` member variable set to the same
       value.

    **Parameters:**

    n_in : integer, optional
        Number of input units. Is ignored, when ``tasks`` is supplied.

    n_out : integer or array_like, optional
        Number of output units. May be an integer (all tasks get
        the same number of units; ``n_tasks`` must be given), or
        ``array_like`` (create as many tasks as elements in
        ``n_out`` with different sizes; ``n_tasks is ignored). Is
        always ignored when ``tasks`` is supplied.

    test_error_fct : string, optional
        See :class:`hebel.layers.LogisticLayer` for
        options. Ignored when ``tasks`` is supplied.
        
    l1_penalty_weight : float or list/tuple of floats, optional
        Weight(s) for L1 regularization. Ignored when ``tasks`` is
        supplied. 
        
    l2_penalty_weight : float or list/tuple of floats, optional
        Weight(s)for L2 regularization. Ignored when ``tasks`` is
        supplied. 

    tasks : list/tuple of :class:`hebel.layers.TopLayer` objects, optional
        Tasks for multitask learning. Overrides ``n_in``,
        ``n_out``, ``test_error_fct``, ``l1_penalty_weight``,
        ``l2_penalty_weight``, ``n_tasks``, and ``lr_multiplier``.

    task_weights : list/tuple of floats, optional
        Weights to use when adding the gradients from the
        different tasks. Default is ``1./self.n_tasks``. The
        weights don't need to necessarily add up to one.

    n_tasks : integer, optional
        Number of tasks. Ignored if ``n_out`` is a list, or
        ``tasks`` is supplied.

    lr_multiplier : float or list/tuple of floats
        A task dependant multiplier for the learning rate. If this
        is ignored, then the tasks default is used. It is ignored
        when ``tasks`` is supplied.

    **See also:**
    :class:`hebel.layers.TopLayer`,
    :class:`hebel.layers.LogisticLayer`

    **Examples**::

        # Simple form of the constructor
        # Creating five tasks with same number of classes
        multitask_layer = MultitaskTopLayer(n_in=1000, n_out=10, n_tasks=5)

        # Extended form of the constructor
        # Initializing every task independently

        n_in = 1000              # n_in must be the same for all tasks
        tasks = (
            SoftmaxLayer(n_in, 10, l1_penalty_weight=.1),
            SoftmaxLayer(n_in, 15, l2_penalty_weight=.2),
            SoftmaxLayer(n_in, 10),
            SoftmaxLayer(n_in, 10),
            SoftmaxLayer(n_in, 20)
        )
        task_weights = [1./5, 1./10, 1./10, 2./5, 1./5]
        multitask_layer = MultitaskTopLayer(tasks=tasks,
                                            task_weights=task_weights)
    """

    def __init__(self, n_in=None, n_out=None,
                 test_error_fct='class_error',
                 l1_penalty_weight=0., l2_penalty_weight=0.,
                 tasks=None, task_weights=None, n_tasks=None,
                 lr_multiplier=None):

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
                self.tasks.append(SoftmaxLayer(n_in=n_in,
                                                n_out=n_out_task,
                                                l1_penalty_weight=l1_task,
                                                l2_penalty_weight=l2_task,
                                                test_error_fct=test_error_task,
                                                lr_multiplier=lr_multiplier))

        else:
            self.tasks = tasks
            assert all([self.tasks[0].n_in == t.n_in for t in tasks])

            self.n_in = self.tasks[0].n_in
            self.n_out = [t.n_out for t in self.tasks]
            self.n_tasks = len(self.tasks)

        if task_weights is not None:
            self.task_weights = task_weights
        else:
            self.task_weights = self.n_tasks * [1. / self.n_tasks]

        self.l1_penalty_weight = l1_penalty_weight
        self.l2_penalty_weight = l2_penalty_weight

        self.n_parameters = sum(task.n_parameters for task in self.tasks)
        self.lr_multiplier = [lr for task in self.tasks
                              for lr in task.lr_multiplier]

    def preallocate_temp_objects(self, batch_size):
        for task in self.tasks:
            if hasattr(task, 'preallocate_temp_objects'):
                task.preallocate_temp_objects(batch_size)

    @property
    def parameters(self):
        """Return a list where each element contains the parameters for a task.
        """
        parameters = []
        for task in self.tasks:
            parameters.extend(task.parameters)
        return parameters

    @parameters.setter
    def parameters(self, value):
        """Update the parameters.

        ``value`` must be a list/tuple of length
        ``MultitaskTopLayer.n_tasks``, each element of which must have
        the correct number of parameters for the task.
        """

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
        """Returns a dictionary describing the architecture of the layer."""
        return [task.architecture for task in self.tasks]

    @property
    def l1_penalty(self):
        """Compute the L1 penalty for all tasks."""
        return sum([task.l1_penalty for task in self.tasks])

    @property
    def l2_penalty(self):
        """Compute the L2 penalty for all tasks."""
        return sum([task.l2_penalty for task in self.tasks])

    def feed_forward(self, input_data, prediction=False):
        """Call ``feed_forward`` for each task and combine the activations.

        Passes ``input_data`` to all tasks and returns the activations
        as a list.
    
        **Parameters:**

        input_data : ``GPUArray``
            Inpute data to compute activations for.

        prediction : bool, optional
            Whether to use prediction model. Only relevant when using
            dropout. If true, then weights are halved if the task
            uses dropout.

        **Returns:**
        
        activations : list of ``GPUArray``
            The activations of the output units, one element for each task.
        """

        activations = []

        for task in self.tasks:
            activations_task = task.feed_forward(input_data, prediction)
            activations.append(activations_task)

        return activations

    def backprop(self, input_data, targets, cache=None):
        """Compute gradients for each task and combine the results.

        **Parameters:**

        input_data : ``GPUArray``
            Inpute data to compute activations for.

        targets : ``GPUArray``
            The target values of the units.

        cache : list of ``GPUArray``
            Cache obtained from forward pass. If the cache is
            provided, then the activations are not recalculated.

        **Returns:**

        gradients : list
            Gradients with respect to the weights and biases for each task

        df_input : ``GPUArray``
            Gradients with respect to the input, obtained by adding
            the gradients with respect to the inputs from each task,
            weighted by ``MultitaskTopLayer.task_weights``.
        """

        df_input = gpuarray.zeros_like(input_data)

        if cache is None: cache = self.n_tasks * [None]

        gradients = []
        for targets_task, cache_task, task, task_weight  in \
          izip(targets, cache, self.tasks, self.task_weights):
            gradients_task, df_input_task = \
              task.backprop(input_data, targets_task,
                            cache_task)

            df_input = df_input.mul_add(1., df_input_task, task_weight)

            gradients.extend(gradients_task)

        return gradients, df_input

    def test_error(self, input_data, targets, average=True,
                   cache=None, prediction=False,
                   sum_errors=True):
        """Compute the error function on a test data set.

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

        sum_errors : bool, optional
            Whether to add up the errors from the different tasks. If
            this option is chosen, the user must make sure that all
            tasks use the same test error function.

        **Returns:**
        
        test_error : float or list
            Returns a float when ``sum_errors == True`` and a list
            with the individual errors otherwise.
        """

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
        """ Computes the cross-entropy error for all tasks.
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

    train_error = cross_entropy_error
