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

import os
import numpy as np
from jinja2 import Template
from .config import load
from .remote import run_experiment


class ParameterSampler(object):
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __call__(self, n):
        return np.random.uniform(self.low, self.high, n)


class IntParameterSampler(ParameterSampler):
    def __call__(self, n):
        return np.random.randint(self.low, self.high, n)


class LogParameterSampler(ParameterSampler):
    def __call__(self, n):
        return 10. ** np.random.uniform(np.log10(self.low),
                                      np.log10(self.high),
                                      n)


def hyperparameter_search(model_config, hyperparameter_config, n_trials):
    hyperparameter_obj = load(hyperparameter_config)

    hyperparameters = {h: value(n_trials) if
                       isinstance(value, ParameterSampler)
                       else sampler
                       for h, value in hyperparameter_obj.iteritems()}
    template = Template(model_config)

    for i in range(n_trials):
        hyperparameters_trial = {h: v[i] for h, v
                                 in hyperparameters.iteritems()}
        config_trial = template.render(hyperparameters_trial)
        run_experiment.delay(config_trial)
