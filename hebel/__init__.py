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

"""Before Hebel can be used, it must be initialized using the function
:func:`hebel.init`.

"""

import numpy as np

import os as _os
neural_nets_root = _os.path.split(
    _os.path.abspath(_os.path.dirname(__file__)))[0]

is_initialized = False

class _Sampler(object):
    _sampler = None

    def __getattribute__(self, name):
        if name in ('seed', 'set_seed'):
            return object.__getattribute__(self, name)
    
        sampler = object.__getattribute__(self, '_sampler')
        if sampler is None:
            from pycuda import curandom, gpuarray
            seed_func = curandom.seed_getter_uniform if self.seed is None \
              else lambda N: gpuarray.to_gpu(
                      np.array(N * [self.seed], dtype=np.int32))
            sampler = curandom.XORWOWRandomNumberGenerator(seed_func)
            self._sampler = sampler
        return sampler.__getattribute__(name)

    def set_seed(self, seed):
        self.seed = seed
        self._sampler = None
sampler = _Sampler()

class _Context(object):
    _context = None

    def init_context(self, device_id=None):
        if device_id is None:
            from pycuda.autoinit import context
            self._context = context
        else:
            self._context = driver.Device(device_id).make_context()
            self._context.push()

    def __getattribute__(self, name):
        if name == 'init_context':
            return object.__getattribute__(self, name)
        
        if object.__getattribute__(self, '_context') is None:
            raise RuntimeError("Context hasn't been initialized yet")
        
        return object.__getattribute__(self, '_context').__getattribute__(name)

context = _Context()

def init(device_id=None, random_seed=None):
    """Initialize Hebel.

    This function creates a CUDA context, CUBLAS context and
    initializes and seeds the pseudo-random number generator.

    **Parameters:**
    
    device_id : integer, optional
        The ID of the GPU device to use. If this is omitted, PyCUDA's
        default context is used, which by default uses the fastest
        available device on the system. Alternatively, you can put the
        device id in the environment variable ``CUDA_DEVICE`` or into
        the file ``.cuda-device`` in the user's home directory.

    random_seed : integer, optional
        The seed to use for the pseudo-random number generator. If
        this is omitted, the seed is taken from the environment
        variable ``RANDOM_SEED`` and if that is not defined, a random
        integer is used as a seed.
    """

    if device_id is None:
        random_seed = _os.environ.get('CUDA_DEVICE')
    
    if random_seed is None:
        random_seed = _os.environ.get('RANDOM_SEED')

    global is_initialized
    if not is_initialized:
        is_initialized = True

    global context
    context.init_context(device_id)

    from pycuda import gpuarray, driver, curandom

    # Initialize PRG
    sampler.set_seed(random_seed)
        
    # Initialize pycuda_ops
    from hebel import pycuda_ops
    pycuda_ops.init()
