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
import matplotlib.pyplot as plt
from neural_nets.schema import Experiment
from time import sleep

def draw_plot_loop(experiment, loop_delay,
                   max_epochs=None, log_scale=False):
    plt.ion()
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(111)
    if log_scale:
        ax1.set_yscale('log')
    ax2 = ax1.twinx()
    if log_scale:
        ax2.set_yscale('log')

    epochs = experiment.epochs
    train_epochs = np.array(experiment.train_error.epoch)
    train_values = np.array(experiment.train_error.value)

    if max_epochs is not None:
        train_epochs = train_epochs[train_epochs >= epochs - max_epochs]
        train_values = train_values[train_epochs >= epochs - max_epochs]

    train_plot, = ax1.plot(train_epochs,
                           train_values, 'b')

    validation_epochs = np.array(experiment.validation_error.epoch)
    validation_values = np.array(experiment.validation_error.value)

    if max_epochs is not None:
        validation_epochs = \
            validation_epochs[validation_epochs >= epochs - max_epochs]
        validation_values = \
            validation_values[validation_epochs >= epochs - max_epochs]

    validation_plot, = ax2.plot(validation_epochs,
                                validation_values, 'r')

    min_train = train_values.min()
    min_train_epoch = train_epochs[train_values == min_train][0]

    min_validation = validation_values.min()
    min_validation_epoch = \
        validation_epochs[validation_values == min_validation][0]

    annot = plt.suptitle('Epoch: %d, Train error: %.3f (min: %.3f @ %d), '
                         'validation error: %.3f (min: %.3f @ %d)' %
                         (epochs, train_values[-1], min_train, min_train_epoch,
                          validation_values[-1], min_validation,
                          min_validation_epoch))

    fig.canvas.draw()

    while True:
        try:
            experiment.reload()
            epochs = experiment.epochs

            train_epochs = np.array(experiment.train_error.epoch)
            train_values = np.array(experiment.train_error.value)

            if max_epochs is not None:
                train_epochs = \
                    train_epochs[train_epochs >= epochs - max_epochs]
                train_values = \
                    train_values[train_epochs >= epochs - max_epochs]

            train_plot.set_data(train_epochs, train_values)

            validation_epochs = np.array(experiment.validation_error.epoch)
            validation_values = np.array(experiment.validation_error.value)

            if max_epochs is not None:
                validation_epochs = \
                    validation_epochs[validation_epochs >= epochs - max_epochs]
                validation_values = \
                    validation_values[validation_epochs >= epochs - max_epochs]

            validation_plot.set_data(validation_epochs, validation_values)

            min_train = train_values.min()
            min_train_epoch = train_epochs[train_values == min_train][0]

            min_validation = validation_values.min()
            min_validation_epoch = \
                validation_epochs[validation_values == min_validation][0]

            annot.set_text('Epoch: %d, Train error: %.3f (min: %.3f @ %d), '
                           'validation error: %.5f (min: %.5f @ %d)' %
                           (epochs, train_values[-1],
                            min_train, min_train_epoch,
                            validation_values[-1], min_validation,
                            min_validation_epoch))

            ax1.relim()
            ax2.relim()
            ax1.autoscale_view()
            ax2.autoscale_view()

            fig.canvas.draw()
            sleep(loop_delay)
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Monitor a running experiment.")
    parser.add_argument('task_id', type=str)
    parser.add_argument('-l', '--loop', dest='loop_delay',
                        type=float, default=2)
    parser.add_argument('-m', '--max-epochs', dest='max_epochs', type=int)
    parser.add_argument('--log-scale', action='store_true')
    args = parser.parse_args()

    experiment = Experiment.objects(task_id=args.task_id).get()
    draw_plot_loop(experiment, args.loop_delay,
                   args.max_epochs, args.log_scale)
