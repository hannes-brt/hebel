from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
import os
from jinja2 import Template
from . import sequence_conv_root
from neural_nets.pycuda_ops.reductions import matrix_sum_out_axis

_TILE_SIZE_CONV = 1024

_src_dir = os.path.join(sequence_conv_root, 'src')
_code = Template(open(os.path.join(_src_dir, 'convolution_kernels.cu')).read())


_source_modules = {dtype: SourceModule(_code.render(data_type=dtype), 
                                       include_dirs=[_src_dir])
                  for dtype in ('float', 'double')}

_kernels = {dtype: {f_name + '_kernel': sm.get_function(f_name)
                    for f_name in ('convolve_sequence',
                                   'convolve_sequence_gradient',
                                   'gradient_reduce',
                                   'max_pool',
                                   'max_pool_gradient')}
                    for dtype, sm in _source_modules.iteritems()}

_dtype_name = {np.dtype(np.float32): 'float', np.dtype(np.float64): 'double'}

def convolve_sequence(mat, conv_filter, bias,
                      target=None, stream=None):
    dtype = conv_filter.dtype
    assert dtype in (np.float32, np.float64)
    assert bias.shape[0] == conv_filter.shape[0]

    if target is not None:
        assert target.dtype == dtype
        assert mat.shape == target.shape[:2]

    height, width = mat.shape
    n_filters = conv_filter.shape[0]
    width_filter = conv_filter.shape[1] // 4

    block = (_TILE_SIZE_CONV, 1, 1)
    grid = (int(np.ceil(mat.size / float(block[0]))),
            n_filters, 1)
    shared = ((_TILE_SIZE_CONV + width_filter - 1) * np.dtype(dtype).itemsize +
              4 * width_filter * np.dtype(dtype).itemsize)  # filter_shared
    
    if target is None:
        target = gpuarray.empty((height, n_filters, width),
                                dtype=dtype)

    dname = _dtype_name[dtype]
    _kernels[dname]['convolve_sequence_kernel'](
        mat, target, conv_filter, bias,
        np.uint32(width), np.uint32(height),
        np.uint32(width_filter),
        np.uint32(n_filters),
        block=block, grid=grid, stream=stream, 
        shared=shared)

    return target

def convolve_sequence_gradient(mat, df_output, filter_width, n_filters,
                               target=None, stream=None, block_size=1024):
    stride = 4
    dtype = df_output.dtype
    assert dtype in (np.float32, np.float64)

    assert mat.shape[1] == df_output.shape[2]

    height, width = mat.shape
    n_elements = mat.size

    block = (block_size, 1, 1)
    grid = (int(np.ceil(n_elements / float(block_size))), n_filters, 1)
    shared = (filter_width - 1 + block_size +  # df_output_share
              stride * block_size              # df_weights_reduce
              ) * np.dtype(dtype).itemsize

    if target is not None:
        assert target.dtype == dtype
        assert target.shape == (n_filters, stride*filter_width, grid[0])
    else:
        target = gpuarray.empty((n_filters, stride*filter_width, 
                                 grid[0]), dtype=dtype)

    dname = _dtype_name[dtype]
    _kernels[dname]['convolve_sequence_gradient_kernel'](
        mat, df_output, target,
        np.uint32(mat.shape[1]), np.uint32(mat.shape[0]),
        np.uint32(filter_width), np.uint32(n_filters),
        block=block, grid=grid, shared=shared, stream=stream)

    if grid[0] > 0:
        target_sum = gpuarray.empty((n_filters, stride*filter_width), dtype)
        block_sum = (max((1, 2**int(np.ceil(np.log2(grid[0])-1)))), 1, 1)
        grid_sum = (n_filters, stride*filter_width, 1)
        shared = block_sum[0] * np.dtype(dtype).itemsize

        _kernels[dname]['gradient_reduce_kernel'](
            target, target_sum,
            np.uint32(n_filters), np.uint32(filter_width),
            np.uint32(grid[0]),
            block=block_sum, grid=grid_sum,
            shared=shared, stream=stream)
    else:
        target_sum = target.reshape((n_filters, stride*filter_width))

    return target_sum

def max_pool(mat, pool_size, target=None, argmax=None, stream=None):
    dtype = mat.dtype
    assert dtype in (np.float32, np.float64)
    assert pool_size <= mat.shape[2]

    height, n_filters, width = mat.shape

    block = (2**int(np.ceil(np.log2(width))), 1, 1)
    grid = (int(np.ceil(width / pool_size)), height, n_filters)
    shared = block[0]*np.dtype(dtype).itemsize + \
      block[0]*np.dtype(np.uint32).itemsize

    if target is not None:
        assert target.dtype == dtype
        assert target.shape == (height,
                                n_filters,
                                width / pool_size)
    else:
        target = gpuarray.empty(
            (height, n_filters, width / pool_size),
            dtype)

    if argmax is not None:
        assert argmax.dtype == np.uint32
        assert argmax.shape == target.shape
    else:
        argmax = gpuarray.empty(
            (height, n_filters, width / pool_size),
            np.uint32)
    
    dname = _dtype_name[dtype]

    _kernels[dname]['max_pool_kernel'](
        mat, target, argmax, np.uint32(height), np.uint32(width), np.uint32(pool_size),
        block=block, grid=grid, shared=shared, stream=stream)

    return target, argmax

def max_pool_gradient(mat, argmax,
                      df_output, pool_size, target=None, stream=None):
    dtype = mat.dtype
    assert dtype in (np.float32, np.float64)

    height, n_filters, width = mat.shape

    block = (pool_size, 1, 1)
    grid = (int(np.ceil(width / float(pool_size))), n_filters, height)

    if target is not None:
        assert target.dtype == dtype
        assert target.shape == mat.shape
    else:
        target = gpuarray.empty_like(mat)

    dname = _dtype_name[dtype]
    _kernels[dname]['max_pool_gradient_kernel'](
        argmax, df_output, target, 
        np.uint32(height), np.uint32(width), 
        np.uint32(argmax.shape[2]),
        block=block, grid=grid, stream=stream)

    return target

def sum_delta(delta, cache_one_vector=True):
    delta_r = delta.reshape((delta.shape[0], delta.shape[1]*delta.shape[2]))
    delta_sum_a = matrix_sum_out_axis(delta_r, 0)
    delta_r = delta_sum_a.reshape((delta.shape[1], delta.shape[2]))
    delta_sum_b = matrix_sum_out_axis(delta_r, 1).ravel()
    return delta_sum_b
