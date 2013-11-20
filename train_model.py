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

description = """ Run this script with a yaml configuration file as input.
E.g.:

python train_model.py examples/mnist_deep.yaml

"""

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('config_file')
    args = parser.parse_args()

    yaml_src = ''.join(open(args.config_file).readlines())

    import pycuda.autoinit
    from scikits.cuda import linalg
    linalg.init()
    run_from_config(yaml_src)
