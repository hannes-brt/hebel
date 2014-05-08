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

""" Implements monitors that report on the progress of training, such
as error rates and parameters. Currently, we just have
SimpleProgressMonitor, which simply prints the current error to the
shell.

"""

import numpy as np
import time, cPickle, os, sys
from datetime import datetime

class ProgressMonitor(object):
    def __init__(self, experiment_name=None, save_model_path=None,
                 save_interval=None, output_to_log=False, 
                 model=None):

        self.experiment_name = experiment_name
        self.save_model_path = save_model_path
        self.save_interval = save_interval
        self.output_to_log = output_to_log
        self.model = model

        self.train_error = []
        self.validation_error = []
        self.avg_epoch_t = None
        self._time = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')

        self.epochs = 0

        self.makedir()

    def print_(self, obj):
        if self.log is not None:
            self.log.write(str(obj) + '\n')
        print obj

    @property
    def yaml_config(self):
        return self._yaml_config

    @yaml_config.setter
    def yaml_config(self, yaml_config):
        if yaml_config is not None:
            self._yaml_config = yaml_config
            yaml_path = os.path.join(self.save_path, 'yaml_config.yml')
            f = open(yaml_path, 'w')
            f.write(self._yaml_config)
            self._yaml_config = yaml_config

    @property
    def test_error(self):
        return self._test_error

    @test_error.setter
    def test_error(self, test_error):
        self._test_error = test_error
        self.print_("Test error: %.4f" % test_error)
        f = open(os.path.join(self.save_path, "test_error"), 'w')
        f.write('%.5f\n' % test_error)

    def makedir(self):
        experiment_dir_name = '_'.join((
            self.experiment_name,
            datetime.now().strftime('%Y-%m-%dT%H-%M-%S')))

        path = os.path.join(self.save_model_path,
                            experiment_dir_name)
        if not os.path.exists(path):
            os.makedirs(path)
        self.save_path = path

        if self.output_to_log:
            self.log = open(os.path.join(self.save_path, 'output.log'), 'w', 1)
            # sys.stdout = self.log
            # sys.stderr = self.log

    def start_training(self):
        self.start_time = datetime.now()

    def report(self, epoch, train_error, validation_error=None,
               new_best=None, epoch_t=None):
        # Print logs
        self.train_error.append((epoch, train_error))
        if validation_error is not None:
            self.validation_error.append((epoch, validation_error))
        self.print_error(epoch, train_error, validation_error, new_best)

        if epoch_t is not None:
            self.avg_epoch_t = ((epoch - 1) * \
                                self.avg_epoch_t + epoch_t) / epoch \
                                if self.avg_epoch_t is not None else epoch_t

        # Pickle model
        if self.save_interval is not None:
            if not epoch % self.save_interval:
                filename = 'model_%s_epoch%04d.pkl' % (
                  self.experiment_name,
                  epoch)
                path = os.path.join(self.save_path, filename)
                cPickle.dump(self.model, open(path, 'wb'))
        elif new_best is not None and new_best:
            filename = 'model_%s_current_best.pkl' % self.experiment_name
            path = os.path.join(self.save_path, filename)
            cPickle.dump(self.model, open(path, 'wb'))

    def print_error(self, epoch, train_error, validation_error=None, new_best=None):
        if validation_error is not None:
            report_str = 'Epoch %d, Validation error: %.5g, Train Loss: %.3f' % \
              (epoch, validation_error, train_error)
            if new_best is not None and new_best:
                report_str += ' (*)'
        else:
            report_str = 'Epoch %d, Train Loss: %.3f' % \
              (epoch, train_error)
        self.print_(report_str)

    def avg_weight(self):
        self.print_("\nAvg weights:")

        i = 0
        for param in self.model.parameters:
            if len(param.shape) != 2: continue
            param_cpu = np.abs(param.get())
            mean_weight = param_cpu.mean()
            std_weight = param_cpu.std()
            self.print_('Layer %d: %.4f [%.4f]' % (i, mean_weight, std_weight))
            i += 1

    def finish_training(self):
        # Print logs
        end_time = datetime.now()
        self.train_time = end_time - self.start_time
        self.print_("Runtime: %dm %ds" % (self.train_time.total_seconds() // 60,
                                    self.train_time.total_seconds() % 60))
        self.print_("Avg. time per epoch %.2fs" % self.avg_epoch_t)

        # Pickle model
        filename = 'model_%s_final.pkl' % self.experiment_name
        path = os.path.join(self.save_path, filename)
        self.print_("Saving model to %s" % path)
        cPickle.dump(self.model, open(path, 'wb'))
        if self.save_interval is None:
            os.remove(os.path.join(
                self.save_path, 'model_%s_current_best.pkl' % self.experiment_name))

    def __del__(self):
        if self.output_to_log:
            self.log.close()


class SimpleProgressMonitor(object):
    def __init__(self, model=None):
        self.model = model

        self.train_error = []
        self.validation_error = []
        self.avg_epoch_t = None
        self._time = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')

    def start_training(self):
        self.start_time = datetime.now()

    def report(self, epoch, train_error, validation_error=None,
               new_best=None, epoch_t=None):
        self.train_error.append((epoch, train_error))
        if validation_error is not None:
            self.validation_error.append((epoch, validation_error))

        # Print logs
        self.print_error(epoch, train_error, validation_error)

        if epoch_t is not None:
            self.avg_epoch_t = ((epoch - 1) * \
                                self.avg_epoch_t + epoch_t) / epoch \
                                if self.avg_epoch_t is not None else epoch_t

    def print_error(self, epoch, train_error, validation_error=None):
        if validation_error is not None:
            print 'Epoch %d, Validation error: %.5g, Train Loss: %.3f' % \
              (epoch, validation_error, train_error)
        else:
            print 'Epoch %d, Train Loss: %.3f' % \
              (epoch, train_error)

    def avg_weight(self):
        print "\nAvg weights:"

        i = 0
        for param in self.model.parameters:
            if len(param.shape) != 2: continue
            param_cpu = np.abs(param.get())
            mean_weight = param_cpu.mean()
            std_weight = param_cpu.std()
            print 'Layer %d: %.4f [%.4f]' % (i, mean_weight, std_weight)
            i += 1

    def finish_training(self):
        # Print logs
        end_time = datetime.now()
        self.train_time = end_time - self.start_time
        print "Runtime: %dm %ds" % (self.train_time.total_seconds() // 60,
                                    self.train_time.total_seconds() % 60)
        print "Avg. time per epoch %.2fs" % self.avg_epoch_t
