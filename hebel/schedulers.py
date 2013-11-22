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

""" A bunch of different schedulers to scale learning
parameters. These are used e.g. to slowly reduce the learning rate
during training or scale momentum up and down during the early and
late phases of training.
"""


def constant_scheduler(value):
    while True:
        yield value


def exponential_scheduler(init_value, decay):
    """ Decreases exponentially """

    value = init_value
    while True:
        yield value
        value *= decay


def linear_scheduler_up(init_value, target_value, duration):
    """ Increases linearly and then stays flat """

    value = init_value
    t = 0
    while True:
        yield value
        t += 1
        if t < duration:
            value = init_value + t * (target_value - init_value) / duration
        else:
            value = target_value


def linear_scheduler_up_down(init_value, target_value, final_value,
                             duration_up, t_decrease, duration_down):
    """ Increases linearly to target_value, stays at target_value until
    t_decrease and then decreases linearly
    """

    value = init_value
    t = 0

    while True:
        yield value
        t += 1
        if t < duration_up:
            value = init_value + t * (target_value - init_value) / \
                    float(duration_up)
        elif t > t_decrease:
            value = target_value - (t - t_decrease) * \
                    (target_value - final_value) / \
                    float(duration_down)
        else:
            value = target_value
