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

filters_in_per_block = 4 # min(4, n_filters_in)
positions_per_block = 2 # min(2, filter_width)
filters_out_per_block = 4 # min(4, n_filters_out)
filters_per_iter = 4

_source_modules = {dtype: SourceModule(_code.render(dtype=dtype, dtype_idx=dtype_idx,
                                                    filters_in_per_block=filters_in_per_block,
                                                    positions_per_block=positions_per_block,
                                                    filters_out_per_block=filters_out_per_block,
                                                    filters_per_iter=filters_per_iter),
                                       include_dirs=[_src_dir], no_extern_c=True)
                   for dtype, dtype_idx in (('float', 'unsigned int'),
                                            # ('double', 'unsigned long'))}
                                            )}

_kernels = {dtype: {f_name + '_kernel': sm.get_function(f_name)
                    for f_name in ('convolve_dna_sequence',
                                   'convolve_dna_sequence_gradient',
                                   'convolve_1d',
                                   'convolve_1d_grad_filters',
                                   'convolve_1d_grad_input',
                                   'gradient_reduce',
                                   'max_pool',
                                   'max_pool_gradient',
                                   'sum_pool',
                                   'sum_pool_gradient')}
                    for dtype, sm in _source_modules.iteritems()}

_dtype_name = {np.dtype(np.float32): 'float', np.dtype(np.float64): 'double'}

# Tell PyCUDA about the types of kernel arguments
_kernels['float']['convolve_dna_sequence_kernel'].prepare('PPPPIIII')
_kernels['float']['convolve_dna_sequence_gradient_kernel'].prepare('PPPIIII')
_kernels['float']['convolve_1d_kernel'].prepare('PPPPIIIII')
_kernels['float']['convolve_1d_grad_filters_kernel'].prepare('PPPIIIII')
_kernels['float']['convolve_1d_grad_input_kernel'].prepare('PPPIIIII')
_kernels['float']['gradient_reduce_kernel'].prepare('PPII')
_kernels['float']['max_pool_kernel'].prepare('PPPIIIIP')
_kernels['float']['max_pool_gradient_kernel'].prepare('PPPIIII')
_kernels['float']['sum_pool_kernel'].prepare('PPIIII')
_kernels['float']['sum_pool_gradient_kernel'].prepare('PPIIII')

def convolve_sequence(input_seq, conv_filter, bias,
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
    filter_width = conv_filter.shape[1]

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
        target = gpuarray.empty((height, output_width, n_filters),
                                dtype=dtype, allocator=memory_pool.allocate)
    assert target.dtype == dtype
    assert target.flags.c_contiguous
    assert target.shape == (height, output_width, n_filters)

    dname = _dtype_name[dtype]
    _kernels[dname]['convolve_dna_sequence_kernel'].prepared_async_call(
        grid, block, stream,
        input_seq.gpudata,
        target.gpudata,
        conv_filter.gpudata,
        bias.gpudata,
        dtype_idx(width),
        dtype_idx(height),
        dtype_idx(filter_width),
        dtype_idx(n_filters),
        shared_size=shared)

    return target

def convolve_1d(mat, filters, bias, target=None, stream=None):
    assert mat.flags.c_contiguous
    assert filters.flags.c_contiguous
    assert bias.flags.c_contiguous

    dtype = filters.dtype
    assert dtype in (np.float32, ) # np.float64)

    if dtype == np.float32:
        dtype_idx = np.uint32
    else:
        dtype_idx = np.uint64
    
    assert bias.shape[0] == filters.shape[0]

    height, width, n_filters_in = mat.shape

    n_filters_out, filter_width = filters.shape[:2]

    halo_width = filter_width - 1
    output_width = width - halo_width

    grid_func = lambda block: (ceil_div(n_filters_out, block[0]),
                               min(ceil_div(height * output_width, block[1]), 48), 1)
    shared_func = lambda block, f_iter: \
                  ((block[1] + 2 * halo_width) * f_iter + 
                   block[0] * filter_width * f_iter +
                   block[0]) * np.dtype(dtype).itemsize
    
    block = (min(4, n_filters_out),
             min(10 * MULTIPROCESSOR_COUNT, output_width), 1)
    grid = grid_func(block)
    shared = shared_func(block, filters_per_iter)

    while shared > MAX_SHARED_MEMORY_PER_BLOCK:
        # if filters_per_iter > 4:
        #     filters_per_iter /= 2
        # else:
        if block[0] > 1:
            block = (block[0] - 1, block[1], 1)
        else:
            block = (block[0], block[1] / 2, 1)
        grid = grid_func(block)
        shared = shared_func(block, filters_per_iter)

    assert np.product(block) <= MAX_THREADS_PER_BLOCK
    assert shared <= MAX_SHARED_MEMORY_PER_BLOCK

    if target is None:
        target = gpuarray.empty((height, output_width, n_filters_out),
                                dtype=dtype, allocator=memory_pool.allocate)
    assert target.dtype == dtype
    assert target.flags.c_contiguous
    assert target.shape == (height, output_width, n_filters_out)

    dname = _dtype_name[dtype]
    _kernels[dname]['convolve_1d_kernel'].prepared_async_call(
        grid, block, stream,
        mat.gpudata,
        target.gpudata,
        filters.gpudata,
        bias.gpudata,
        dtype_idx(width),
        dtype_idx(height),
        dtype_idx(filter_width),
        dtype_idx(n_filters_in),
        dtype_idx(n_filters_out),
        shared_size=shared)

    return target

def convolve_sequence_gradient(mat, df_output,
                               filter_width, n_filters,
                               target=None, stream=None):

    assert mat.flags.c_contiguous
    assert df_output.flags.c_contiguous

    dtype = df_output.dtype
    assert dtype in (np.float32,) # np.float64)

    height, width = mat.shape
    halo_width = filter_width - 1
    output_width = width - halo_width

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
        grid = (n_filters,
                min(div_up(output_width*height, block[1]), 192 / 2), 1)
        shared = get_shared(block, grid)

    if target is None:
        target = gpuarray.empty((n_filters, filter_width, N_LETTERS), dtype,
                                allocator=memory_pool.allocate)

    target_tmp = None
    if grid[1]:
        target_tmp = gpuarray.empty(
            (grid[1], n_filters, filter_width, N_LETTERS), dtype=dtype,
            allocator=memory_pool.allocate)
    else:
        target_tmp = target

    dname = _dtype_name[dtype]
    _kernels[dname]['convolve_dna_sequence_gradient_kernel'].prepared_async_call(
        grid, block, stream,
        mat.gpudata,
        df_output.gpudata,
        target_tmp.gpudata,
        np.uint32(width),
        np.uint32(height),
        np.uint32(filter_width),
        np.uint32(n_filters),
        shared_size=shared)

    if grid[1]:
        block_sum = (MULTIPROCESSOR_COUNT * 2, 1, 1)
        grid_sum = (100, 1, 1)

        _kernels[dname]['gradient_reduce_kernel'].prepared_async_call(
            grid_sum, block_sum, stream,
            target_tmp.gpudata,
            target.gpudata,
            np.uint32(target.size),
            np.uint32(grid[1]))

    return target

def convolve_1d_gradient_filters(input_data, df_output, filter_width,
                                 target=None, stream=None):
    dtype = np.dtype(np.float32)
    
    assert input_data.flags.c_contiguous
    assert df_output.flags.c_contiguous
    assert input_data.dtype == dtype
    assert df_output.dtype == dtype

    height, input_width, n_filters_in = input_data.shape
    output_width, n_filters_out = df_output.shape[1:]

    halo_width = filter_width - 1
    assert output_width == input_width - halo_width

    if target is None:
        target = gpuarray.zeros((n_filters_out, filter_width, n_filters_in), dtype,
                                allocator=memory_pool.allocate)
    else:
        target.fill(0.)
        
    assert target.flags.c_contiguous
    assert target.dtype == dtype
    assert target.shape == (n_filters_out, filter_width, n_filters_in)

    # filters_in_per_block = 4 # min(4, n_filters_in)
    # positions_per_block = 1 # min(1, filter_width)
    # filters_out_per_block = 4 # min(4, n_filters_out)
    elements_per_block = min(height * output_width, 8)

    n_blocks = ceil_div(n_filters_in, filters_in_per_block) * \
               ceil_div(filter_width, positions_per_block) * \
               ceil_div(n_filters_out, filters_out_per_block)

    block = (filters_in_per_block * positions_per_block * filters_out_per_block, elements_per_block, 1)
    grid = (n_blocks, min(div_up(height * output_width, elements_per_block), 64), 1)
    n_input_elements = block[1] + 2 * halo_width    
    
    shared = (
        block[1] * filters_out_per_block +
        block[0] +
        n_input_elements * filters_in_per_block
    ) * np.dtype(dtype).itemsize

    dname = _dtype_name[dtype]
    _kernels[dname]['convolve_1d_grad_filters_kernel'].prepared_async_call(
        grid, block, stream,
        input_data.gpudata,
        df_output.gpudata,
        target.gpudata,
        np.uint32(input_width),
        np.uint32(height),
        np.uint32(filter_width),
        np.uint32(n_filters_in),
        np.uint32(n_filters_out),
        shared_size=shared
    )

    return target

def convolve_1d_gradient_input(df_output, filters,
                               target=None, stream=None):
    dtype = np.dtype(np.float32)

    assert df_output.flags.c_contiguous
    assert filters.flags.c_contiguous
    assert df_output.dtype == dtype
    assert filters.dtype == dtype

    n_filters_out, filter_width, n_filters_in = filters.shape
    height, output_width = df_output.shape[:2]

    halo_width = filter_width - 1
    input_width = output_width + halo_width

    if target is None:
        target = gpuarray.zeros((height, input_width, n_filters_in), dtype,
                                allocator=memory_pool.allocate)
    else:
        target.fill(0.)

    assert target.flags.c_contiguous
    assert target.dtype == dtype
    assert target.shape == (height, input_width, n_filters_in)

    elements_per_block = min(height * output_width, 4)

    n_blocks = ceil_div(n_filters_in, filters_in_per_block) * \
               ceil_div(filter_width, positions_per_block) * \
               ceil_div(n_filters_out, filters_out_per_block)

    block = (filters_in_per_block * positions_per_block * filters_out_per_block,
             elements_per_block, 1)
    grid = (n_blocks, min(ceil_div(height * output_width, elements_per_block), 8), 1)
    
    n_input_elements = block[1] + halo_width + positions_per_block - 1

    shared = (
        block[1] * filters_out_per_block +
        block[0] +
        n_input_elements * filters_in_per_block
    ) * dtype.itemsize

    dname = _dtype_name[dtype]
    _kernels[dname]['convolve_1d_grad_input_kernel'].prepared_async_call(
        grid, block, stream,
        df_output.gpudata,
        filters.gpudata,
        target.gpudata,
        np.uint32(input_width),
        np.uint32(height),
        np.uint32(filter_width),
        np.uint32(n_filters_in),
        np.uint32(n_filters_out),
        shared_size=shared,
    )

    return target


def max_pool(mat, pool_size,
             target=None, argmax=None, stream=None):
    assert mat.flags.c_contiguous
    assert len(mat.shape) == 3

    dtype = mat.dtype
    assert dtype in (np.float32, ) # np.float64)
    assert pool_size <= mat.shape[1]

    height, width, n_filters = mat.shape
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
            (height, pooled_width, n_filters),
            dtype, allocator=memory_pool.allocate)

    if argmax is not None:
        assert argmax.dtype == np.uint32
        assert argmax.shape == target.shape
        assert argmax.flags.c_contiguous
    else:
        argmax = gpuarray.empty(target.shape, np.uint32, allocator=memory_pool.allocate)

    dname = _dtype_name[dtype]

    _kernels[dname]['max_pool_kernel'].prepared_async_call(
        grid, block, stream,
        mat.gpudata,
        target.gpudata,
        argmax.gpudata,
        np.uint32(height),
        np.uint32(width),
        np.uint32(n_filters),
        np.uint32(pool_size),
        sampler.state)

    return target, argmax

def sum_pool(mat, pool_size,
             target=None, stream=None):
    assert mat.flags.c_contiguous
    assert len(mat.shape) == 3

    dtype = mat.dtype
    assert dtype in (np.float32, ) # np.float64)
    assert pool_size <= mat.shape[1]

    height, width, n_filters = mat.shape
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
        assert target.shape == (height, pooled_width, n_filters)
    else:
        target = gpuarray.empty(
            (height, pooled_width, n_filters),
            dtype, allocator=memory_pool.allocate)

    dname = _dtype_name[dtype]

    _kernels[dname]['sum_pool_kernel'].prepared_async_call(
        grid, block, stream,
        mat.gpudata,
        target.gpudata,
        np.uint32(height),
        np.uint32(width),
        np.uint32(n_filters),
        np.uint32(pool_size))

    return target

def max_pool_gradient(mat, argmax, df_output,
                      target=None, stream=None):
    dtype = mat.dtype
    assert mat.flags.c_contiguous
    assert argmax.flags.c_contiguous
    assert df_output.flags.c_contiguous
    assert dtype in (np.float32, ) # np.float64)

    height, width, n_filters = mat.shape

    width_pooled = df_output.shape[1]
    
    # assert not width % width_pooled
    pool_size = ceil_div(width, width_pooled)

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
    _kernels[dname]['max_pool_gradient_kernel'].prepared_async_call(
        grid, block, stream,
        argmax.gpudata,
        df_output.gpudata,
        target.gpudata,
        np.uint32(height),
        np.uint32(width),
        np.uint32(width_pooled),
        np.uint32(n_filters),
        shared_size=shared)

    return target

def sum_pool_gradient(mat, df_output,
                      target=None, stream=None):
    dtype = mat.dtype
    assert mat.flags.c_contiguous
    assert df_output.flags.c_contiguous
    assert dtype in (np.float32, ) # np.float64)

    height, width, n_filters = mat.shape

    width_pooled = df_output.shape[1]
    
    # assert not width % width_pooled
    pool_size = ceil_div(width, width_pooled)

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
    _kernels[dname]['sum_pool_gradient_kernel'].prepared_async_call(
        grid, block, stream,
        df_output.gpudata,
        target.gpudata,
        np.uint32(height),
        np.uint32(width),
        np.uint32(width_pooled),
        np.uint32(n_filters),
        shared_size=shared)

    return target

def sum_delta(delta, n_filters, cache_one_vector=True):
    assert delta.flags.c_contiguous
    assert not delta.shape[1] % n_filters
    width = delta.shape[1] / n_filters
    target_tmp = matrix_sum_out_axis(delta, 0, cache_one_vector)
    delta_r = target_tmp.reshape((n_filters, width))
    target_sum = matrix_sum_out_axis(delta_r, 1, cache_one_vector).ravel()
    return target_sum


