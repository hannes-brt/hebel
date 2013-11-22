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


class SimpleProgressMonitor(object):
    def __init__(self):
        self.model = None

        self.train_error = []
        self.validation_error = []
        self.avg_epoch_t = None
        self._time = datetime.isoformat(datetime.now())

    def start_training(self):
        self.start_time = datetime.now()

    def report(self, epoch, train_error, validation_error=None, epoch_t=None):
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
