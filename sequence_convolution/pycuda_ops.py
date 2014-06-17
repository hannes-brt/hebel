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
from hebel.pycuda_ops.reductions import matrix_sum_out_axis
from hebel.utils.math import ceil_div, div_up
from hebel import sampler

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


_source_modules = {dtype: SourceModule(_code.render(dtype=dtype, dtype_idx=dtype_idx),
                                       include_dirs=[_src_dir], no_extern_c=True)
                   for dtype, dtype_idx in (('float', 'unsigned int'),
                                            # ('double', 'unsigned long'))}
                                            )}

_kernels = {dtype: {f_name + '_kernel': sm.get_function(f_name)
                    for f_name in ('convolve_dna_sequence',
                                   'convolve_dna_sequence_gradient',
                                   'gradient_reduce',
                                   'max_pool',
                                   'max_pool_gradient',
                                   'sum_pool',
                                   'sum_pool_gradient')}
                    for dtype, sm in _source_modules.iteritems()}

_dtype_name = {np.dtype(np.float32): 'float', np.dtype(np.float64): 'double'}


def convolve_sequence(input_seq, conv_filter, bias,
                      width=None,
                      target=None, stream=None):

    assert input_seq.flags.c_contiguous
    assert conv_filter.flags.c_contiguous
    assert bias.flags.c_contiguous

    dtype = conv_filter.dtype
    assert dtype in (np.float32, ) # np.float64)

    if dtype == np.float32:
        dtype_idx = np.uint32
    else:
        dtype_idx = np.uint64
    
    assert bias.shape[0] == conv_filter.shape[0]

    height, width = input_seq.shape

    n_filters = conv_filter.shape[0]
    filter_width = conv_filter.shape[1] // N_LETTERS

    halo_width = filter_width - 1
    output_width = width - halo_width

    block = (4, 8, 1)
    grid = (ceil_div(n_filters, block[0]), 128, 1)
    n_input_elements = ceil_div(block[1], output_width) * width + halo_width
    shared = block[0] * filter_width * N_LETTERS  * np.dtype(dtype).itemsize + \
             block[0] * np.dtype(dtype).itemsize + \
             div_up(n_input_elements, 4) * np.dtype('|S1').itemsize
        
    assert np.product(block) < MAX_THREADS_PER_BLOCK
    assert shared < MAX_SHARED_MEMORY_PER_BLOCK

    if target is None:
        target = gpuarray.empty((height, output_width * n_filters),
                                dtype=dtype)
    assert target.dtype == dtype
    assert target.flags.c_contiguous
    assert target.shape == (height, output_width * n_filters)

    dname = _dtype_name[dtype]
    _kernels[dname]['convolve_dna_sequence_kernel'](
        input_seq,
        target,
        conv_filter,
        bias,
        dtype_idx(width),
        dtype_idx(height),
        dtype_idx(filter_width),
        dtype_idx(n_filters),
        block=block,
        grid=grid,
        stream=stream,
        shared=shared)

    return target


def convolve_sequence_gradient_wrapper():
    target_tmp_cache = []
    
    def f(mat, df_output, filter_width, n_filters,
          target=None, stream=None):

        assert mat.flags.c_contiguous
        assert df_output.flags.c_contiguous

        dtype = df_output.dtype
        assert dtype in (np.float32,) # np.float64)

        height, width = mat.shape
        halo_width = filter_width - 1
        output_width = width - halo_width
        df_output_width = df_output.shape[1]

        n_elements = height*width        

        block = (filter_width, MULTIPROCESSOR_COUNT, 1)
        grid = (n_filters,
                min(div_up(output_width*height, block[1]), 192 / 2), 1)
        n_input_elements = ceil_div(block[1], output_width) * width + halo_width
        get_shared = lambda block, grid: block[0] * filter_width * N_LETTERS * \
                     np.dtype(dtype).itemsize + \
                     block[1] * filter_width * np.dtype(dtype).itemsize + \
                     (n_input_elements + halo_width) * np.dtype('|S1').itemsize
        shared = get_shared(block, grid)
        while np.prod(block) > MAX_THREADS_PER_BLOCK or \
              shared > MAX_SHARED_MEMORY_PER_BLOCK:
            block = (block[0], block[1] / 2, 1)
            shared = get_shared(block, grid)
        
        if target is None:
            target = gpuarray.empty((n_filters, N_LETTERS * filter_width), dtype)

        target_tmp = None
        if grid[1]:
            for x in target_tmp_cache:
                if x.shape == (grid[1], n_filters, filter_width, N_LETTERS) and \
                   x.dtype == dtype:
                    target_tmp = x
            if target_tmp is None:
                target_tmp = gpuarray.empty(
                    (grid[1], n_filters, filter_width, N_LETTERS), dtype=dtype)
                target_tmp_cache.append(target_tmp)
        else:
            target_tmp = target

        dname = _dtype_name[dtype]
        _kernels[dname]['convolve_dna_sequence_gradient_kernel'](
            mat,
            df_output,
            target_tmp,
            np.uint32(width),
            np.uint32(height),
            np.uint32(filter_width),
            np.uint32(n_filters),
            block=block, grid=grid, shared=shared,
            stream=stream)

        if grid[1]:
            block_sum = (MULTIPROCESSOR_COUNT * 2, 1, 1)
            grid_sum = (100, 1, 1)

            _kernels[dname]['gradient_reduce_kernel'](
                target_tmp,
                target,
                np.uint32(target.size),
                np.uint32(grid[1]),
                block=block_sum, grid=grid_sum,
                stream=stream)

        return target
    return f
convolve_sequence_gradient = convolve_sequence_gradient_wrapper()


def max_pool(mat, pool_size, n_filters,
             target=None, argmax=None, stream=None,
             time_kernel=False):
    assert mat.flags.c_contiguous
    assert len(mat.shape) == 2

    dtype = mat.dtype
    assert dtype in (np.float32, ) # np.float64)
    assert pool_size <= mat.shape[1]

    height, width = mat.shape
    assert not width % n_filters
    width /= n_filters

    assert not width % pool_size
    pooled_width = width // pool_size

    block = (n_filters, 2 * MULTIPROCESSOR_COUNT, 1)
    grid = (ceil_div(n_filters, block[0]),
            min(ceil_div(pooled_width, block[1]), 192 / 8), 1)

    while np.prod(block) > MAX_THREADS_PER_BLOCK:
        if block[0] > 1:
            block = (block[0] / 2, block[1], 1)
        else:
            block = (1, block[1] / 2, 1)
        grid = (ceil_div(n_filters, block[0]),
                min(ceil_div(pooled_width, block[1]), 192 / 4), 1)
    
    if target is not None:
        assert target.dtype == dtype
        assert target.flags.c_contiguous
        assert target.shape == (height, pooled_width * n_filters)
    else:
        target = gpuarray.empty(
            (height, pooled_width * n_filters),
            dtype)

    if argmax is not None:
        assert argmax.dtype == np.uint32
        assert argmax.shape == target.shape
        assert argmax.flags.c_contiguous
    else:
        argmax = gpuarray.empty(target.shape, np.uint32)

    dname = _dtype_name[dtype]

    t = _kernels[dname]['max_pool_kernel'](
        mat,
        target,
        argmax,
        np.uint32(height),
        np.uint32(width),
        np.uint32(n_filters),
        np.uint32(pool_size),
        sampler.state,
        block=block, grid=grid, stream=stream, time_kernel=time_kernel)

    if time_kernel:
        return target, argmax, t
    else:
        return target, argmax


def max_pool_gradient(mat, argmax, df_output, n_filters,
                      target=None, stream=None, time_kernel=False):
    dtype = mat.dtype
    assert mat.flags.c_contiguous
    assert argmax.flags.c_contiguous
    assert df_output.flags.c_contiguous
    assert dtype in (np.float32, ) # np.float64)

    height, width = mat.shape
    assert not width % n_filters
    width /= n_filters

    width_pooled = df_output.shape[1]
    assert not width_pooled % n_filters
    width_pooled /= n_filters
    
    assert not width % width_pooled
    pool_size = width // width_pooled

    block = (min(n_filters, MULTIPROCESSOR_COUNT), pool_size, 1)
    grid_func = lambda block: (ceil_div(n_filters, block[0]),
                               min(ceil_div(height * width, block[1]), 192 / 4), 1)
    grid = grid_func(block)
    shared_func = lambda block: block[0] * block[1] / pool_size * \
             (np.dtype(dtype).itemsize + np.dtype(np.uint32).itemsize)
    shared = shared_func(block)

    while np.prod(block) > MAX_THREADS_PER_BLOCK or \
          shared > MAX_SHARED_MEMORY_PER_BLOCK:
        if block[1] > pool_size :
            block = (block[0], block[1] / 2, 1)
        else:
            block = (block[0] / 2, block[1], 1)
            
        grid = grid_func(block)
        shared = shared_func(block)

    assert not block[1] % pool_size

    if target is not None:
        assert target.dtype == dtype
        assert target.shape == mat.shape
        assert target.flags.c_contiguous
    else:
        target = gpuarray.empty_like(mat)

    dname = _dtype_name[dtype]
    t = _kernels[dname]['max_pool_gradient_kernel'](
        argmax,
        df_output,
        target,
        np.uint32(height),
        np.uint32(width),
        np.uint32(width_pooled),
        np.uint32(n_filters),
        block=block, grid=grid,
        shared=shared, stream=stream, time_kernel=time_kernel)

    if time_kernel:
        return target, t
    else:
        return target


def sum_delta(delta, n_filters, cache_one_vector=True,
              target_tmp=None, target_sum=None):
    assert delta.flags.c_contiguous
    assert not delta.shape[1] % n_filters
    width = delta.shape[1] / n_filters
    if target_tmp is None:
        target_tmp = gpuarray.empty(delta.shape[1:], delta.dtype)
    target_tmp = matrix_sum_out_axis(delta, 0, cache_one_vector,
                                      target=target_tmp)
    delta_r = target_tmp.reshape((n_filters, width))
    if target_sum is None:
        target_sum = gpuarray.empty(delta_r.shape[:1], delta_r.dtype)
    target_sum = matrix_sum_out_axis(delta_r, 1, cache_one_vector,
                                      target=target_sum).ravel()
    return target_sum


