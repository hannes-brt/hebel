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

""" Implements optimization algorithms to train the models. The single
algorithm we have in online stochastic gradient descent (SGD).

"""

import numpy as np
import time, cPickle, os, inspect
from .pycuda_ops.matrix import vector_normalize
from .schedulers import constant_scheduler
from .monitors import SimpleProgressMonitor


class EarlyStoppingModule(object):
    def __init__(self, model):
        self.model = model
        self.best_validation_loss = np.inf

    def update(self, epoch, validation_loss):
        if validation_loss < self.best_validation_loss:
            print '* ',
            self.best_validation_loss = validation_loss
            self.best_params = [p.copy() for p in self.model.parameters]
            assert self.best_params[0] is not self.model.parameters[0]
            self.best_epoch = epoch
            return True
        return False

    def finish(self):
        self.model.parameters = self.best_params
        print "Optimization complete. " \
            "Best validation error of %.5g obtained in self.epoch %d" % \
            (self.best_validation_loss, self.best_epoch)


class SGD(object):
    @property
    def best_validation_loss(self):
        return self.early_stopping_module.best_validation_loss

    def __init__(self,
                 model, parameter_updater,
                 train_data,
                 validation_data=None,
                 progress_monitor=None,
                 learning_rate_schedule=constant_scheduler(.1),
                 momentum_schedule=None,
                 early_stopping=True):

        """ Stochastic gradient descent
        """

        ### Initialization

        self.model = model

        ### Training data
        self.train_data = train_data

        ### Validation data
        self.validation_data = validation_data

        ### Data size
        self.N_train = self.train_data.N

        if validation_data is not None:
            self.N_validation = self.validation_data.N

        ### Learning rate schedule
        self.learning_parameter_iterators = [learning_rate_schedule]

        ### Momentum, rmsprop, etc

        self.parameter_updater = parameter_updater(self.model)

        if momentum_schedule is not None:
            self.learning_parameter_iterators.append(momentum_schedule)

        if progress_monitor is None:
            self.progress_monitor = SimpleProgressMonitor(model=self.model)
        else:
            self.progress_monitor = progress_monitor

        if self.progress_monitor.model is None:
            self.progress_monitor.model = self.model

        self.early_stopping_module = EarlyStoppingModule(self.model) \
                                     if early_stopping else None

        self.model.preallocate_temp_objects(self.train_data)

    def run(self, iterations=200, validation_interval=5,
            yaml_config=None,
            task_id=None):
        # Initialize variables
        self.epoch = 0
        done_looping = False

        self.progress_monitor.start_training()

        self.progress_monitor.task_id = task_id
        self.progress_monitor.yaml_config = yaml_config

        # Main loop
        for self.epoch in range(self.epoch, self.epoch + iterations):
            learning_parameters = map(lambda lp: lp.next(),
                                      self.learning_parameter_iterators)
            if done_looping: break

            try:
                t = time.time()

                # Train on mini-batches
                train_loss = 0.

                for batch_idx, (batch_data, batch_targets) in \
                  enumerate(self.train_data):
                    batch_size = self.train_data.batch_size

                    self.parameter_updater.pre_gradient_update()

                    batch_loss, gradients = \
                        self.model.training_pass(batch_data, batch_targets)
                    train_loss += batch_loss
                    self.parameter_updater\
                      .post_gradient_update(gradients, batch_size,
                                            learning_parameters)

                # Evaluate on validation data
                if self.validation_data is not None and \
                   not self.epoch % validation_interval:
                    validation_loss_rate = self.model.test_error(
                        self.validation_data)
                    # validation_loss = 0.
                    # for batch_idx, (batch_data, batch_targets) in \
                    #   enumerate(self.validation_data):

                    #     validation_loss += self.model.test_error(batch_data,
                    #                                              batch_targets,
                    #                                              average=False)

                    # validation_loss_rate = \
                    #     validation_loss / float(self.N_validation)

                    new_best = self.early_stopping_module.update(
                        self.epoch, validation_loss_rate) \
                        if self.early_stopping_module is not None else None

                    epoch_t = time.time() - t

                    self.progress_monitor.report(self.epoch, train_loss,
                                                 validation_loss_rate,
                                                 new_best,
                                                 epoch_t=epoch_t)
                else:
                    epoch_t = time.time() - t
                    self.progress_monitor.report(self.epoch, train_loss,
                                                 epoch_t=epoch_t)

            except KeyboardInterrupt:
                print "Keyboard interrupt. Stopping training and cleaning up."
                done_looping = True

        if self.early_stopping_module is not None:
            self.early_stopping_module.finish()

        self.progress_monitor.finish_training()

    def norm_v_norm(self):
        if self.max_vec_norm:
            for w in self.model.parameters:
                if len(w.shape) == 2:
                    vector_normalize(w, self.max_vec_norm)
