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

from neural_nets.config import run_from_config
from neural_nets.remote import run_experiment
from neural_nets.random import hyperparameter_search

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('-r', '--remote', dest='remote', action='store_true')
    parser.add_argument('-t', '--hyper-template')
    parser.add_argument('-n', '--n-trials', type=int, default=10)
    args = parser.parse_args()

    yaml_src = ''.join(open(args.config_file).readlines())

    if args.hyper_template is not None:
        hyperparameter_config = open(args.hyper_template).read()
        hyperparameter_search(yaml_src, hyperparameter_config, args.n_trials)
    elif args.remote:
        task = run_experiment.delay(yaml_src)
        print task.task_id
    else:
        import pycuda.autoinit
        from scikits.cuda import linalg
        linalg.init()
        run_from_config(yaml_src)
