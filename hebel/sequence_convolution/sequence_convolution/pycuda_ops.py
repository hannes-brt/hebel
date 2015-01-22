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

from pycuda import gpuarray
from pycuda.compiler import SourceModule
from pycuda import driver
import numpy as np
import os, ctypes
from jinja2 import Template
from . import sequence_conv_root
from hebel import memory_pool
from hebel.utils.math import div_up

from hebel import context
MAX_THREADS_PER_BLOCK = context.get_device()\
    .get_attribute(driver.device_attribute.MAX_THREADS_PER_BLOCK)
MAX_SHARED_MEMORY_PER_BLOCK = context.get_device()\
    .get_attribute(driver.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK)
MULTIPROCESSOR_COUNT = context.get_device()\
    .get_attribute(driver.device_attribute.MULTIPROCESSOR_COUNT)
N_LETTERS = 4

_src_dir = os.path.join(sequence_conv_root, 'src')
_code = Template(open(os.path.join(_src_dir, 'convolution_kernels.cu')).read())

_source_modules = {dtype: SourceModule(_code.render(dtype=dtype),
                                       include_dirs=[_src_dir], no_extern_c=True)
                   for dtype in ('float', 'double', 'unsigned long')}

_kernels = {dtype: {f_name: sm.get_function(f_name)
                    for f_name in ('sequenceToFloatKernel',)}
                    for dtype, sm in _source_modules.iteritems()}

_dtype_name = {np.dtype(np.float32): 'float', np.dtype(np.float64): 'double'}

# Tell PyCUDA about the types of kernel arguments
_kernels['float']['sequenceToFloatKernel'].prepare('IIPP')
_kernels['double']['sequenceToFloatKernel'].prepare('IIPP')

def sequence_to_float(seq, output=None, dtype=np.float32, stream=None):
    n, w = seq.shape
    if output is None:
        output = gpuarray.empty((n, N_LETTERS, 1, w), dtype,
                                allocator=memory_pool.allocate)

    block = (128, 1, 1)
    grid = (div_up(n * w, block[0]), 1, 1)

    dtype_name = 'float' if dtype == np.float32 else 'double'
    _kernels[dtype_name]['sequenceToFloatKernel'].prepared_async_call(
        grid, block, stream,
        np.uint32(n), np.uint32(w), seq.gpudata, output.gpudata
    )

    return output
