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

from hebel.pycuda_ops import linalg
linalg.init()
from pycuda import gpuarray
import numpy as np

import os as _os
neural_nets_root = _os.path.split(
    _os.path.abspath(_os.path.dirname(__file__)))[0]

# This defers the import of curandom until the first time sampler is accessed
class _Sampler(object):
    _sampler = None

    def __getattribute__(self, name):
        if name in ('seed', 'set_seed'):
            return object.__getattribute__(self, name)
    
        sampler = object.__getattribute__(self, '_sampler')
        if sampler is None:
            from pycuda import curandom
            seed = _os.environ.get('RANDOM_SEED')
            seed_func = curandom.seed_getter_uniform if not seed \
              else lambda N: gpuarray.to_gpu(
                      np.array(N * [seed], dtype=np.int32))
            sampler = curandom.XORWOWRandomNumberGenerator(seed_func)
            self._sampler = sampler
        return sampler.__getattribute__(name)

    def set_seed(self, seed):
        self.seed = seed

sampler = _Sampler()
