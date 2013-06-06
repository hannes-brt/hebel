from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
import os
from jinja2 import Template
from . import sequence_conv_root

_TILE_SIZE_CONV = 128
_MAX_WIDTH_FILTER = 15

_code = Template(open(os.path.join(sequence_conv_root, 
    'src', 'convolution_kernels.cu')).read())

_source_modules = {dtype: SourceModule(_code.render(TILE_SIZE_CONV=_TILE_SIZE_CONV,
                                                    MAX_WIDTH_FILTER=_MAX_WIDTH_FILTER,
                                                    data_type=dtype))
                  for dtype in ('float', 'double')}

_kernels = {dtype: {f_name + '_kernel': sm.get_function(f_name)
                    for f_name in ('convolve_sequence',
                                   'convolve_sequence_gradient',
                                   'gradient_reduce',
                                   'max_pool',
                                   'max_pool_gradient')}
                    for dtype, sm in _source_modules.iteritems()}

_dtype_name = {np.dtype(np.float32): 'float', np.dtype(np.float64): 'double'}

def convolve_sequence(mat, conv_filter, stride=4,
                      target=None, stream=None):
    dtype = mat.dtype
    assert dtype in (np.float32, np.float64)
    assert conv_filter.dtype == dtype

    if target is not None:
        assert target.dtype == dtype
        assert mat.shape // stride == target.shape[:2]

    height, width = mat.shape
    n_filters, width_filter = conv_filter.shape

    block = (_TILE_SIZE_CONV, 1, 1)
    grid = (int(np.ceil(mat.shape[1] / float(block[0]))),
            int(np.ceil(mat.shape[0] / float(block[1]))),
            1)

    if target is None:
        target_width = int(np.ceil(width / float(stride)))
        target = gpuarray.empty((n_filters, height, target_width),
                                dtype=dtype)

    dname = _dtype_name[dtype]
    _kernels[dname]['convolve_sequence_kernel'](
        mat, target, conv_filter,
        np.uint32(width), np.uint32(height),
        np.uint32(width_filter),
        np.uint32(n_filters),
        np.uint32(stride),
        block=block, grid=grid, stream=stream)

    return target

def convolve_sequence_gradient(mat, df_output, filter_width, n_filters,
                               target=None, stream=None, block_size=1024):
    stride = 4
    dtype = mat.dtype
    assert dtype in (np.float32, np.float64)
    assert df_output.dtype == dtype

    assert mat.shape[1] // stride == df_output.shape[2]

    height, width = mat.shape
    n_elements = height * width

    block = (block_size, 1, 1)
    grid = (int(np.ceil(n_elements / float(block[0]))), 1, 1)
    shared = ((filter_width / stride) - 1 + block_size / stride +  # df_output_share
              block_size # df_weights_reduce
              ) * np.dtype(dtype).itemsize

    if target is not None:
        assert target.dtype == dtype
        assert target.shape == (n_filters, filter_width)
    else:
        target = gpuarray.empty((n_filters, filter_width, 
                                 grid[0]), dtype=dtype)

    dname = _dtype_name[dtype]
    _kernels[dname]['convolve_sequence_gradient_kernel'](
        mat, df_output, target,
        np.uint32(mat.shape[1]), np.uint32(mat.shape[0]),
        np.uint32(filter_width), np.uint32(n_filters),
        block=block, grid=grid, shared=shared, stream=stream)

    target_sum = gpuarray.empty((n_filters, filter_width), dtype)
    block_sum = (max((1, 2**int(np.ceil(np.log2(grid[0])-1)))), 1, 1)
    grid_sum = (n_filters, filter_width, 1)
    shared = block_sum[0] * np.dtype(dtype).itemsize

    _kernels[dname]['gradient_reduce_kernel'](
        target, target_sum,
        np.uint32(n_filters), np.uint32(filter_width),
        np.uint32(grid[0]),
        block=block_sum, grid=grid_sum,
        shared=shared, stream=stream)

    if np.any(np.isnan(target_sum.get())):
        import pudb; pudb.set_trace()
    
    return target_sum

def max_pool(mat, pool_size, target=None, stream=None):
    dtype = mat.dtype
    assert dtype in (np.float32, np.float64)
    assert pool_size <= mat.shape[2]

    n_filters, height, width = mat.shape

    block = (2**int(np.ceil(np.log2(width))), 1, 1)
    grid = (int(np.ceil(width / pool_size)), height, n_filters)
    shared = block[0]*np.dtype(dtype).itemsize

    if target is not None:
        assert target.dtype == dtype
        assert target.shape == (n_filters, 
                                height,
                                width / pool_size)
    else:
        target = gpuarray.empty(
            (n_filters, height, width / pool_size),
            dtype)
    
    dname = _dtype_name[dtype]

    _kernels[dname]['max_pool_kernel'](
        mat, target, np.uint32(height), np.uint32(width), np.uint32(pool_size),
        block=block, grid=grid, shared=shared, stream=stream)

    return target 

def max_pool_gradient(mat, mat_pooled, df_output, pool_size, target=None, stream=None):
    dtype = mat.dtype
    assert dtype in (np.float32, np.float64)

    n_filters, height, width = mat.shape

    block = (pool_size, 1, 1)
    grid = (int(np.ceil(width / float(pool_size))), height, n_filters)

    if target is not None:
        assert target.dtype == dtype
        assert target.shape == mat.shape
    else:
        target = gpuarray.empty_like(mat)

    dname = _dtype_name[dtype]
    _kernels[dname]['max_pool_gradient_kernel'](
        mat, mat_pooled, df_output, target, np.uint32(height),
        np.uint32(width), np.uint32(mat_pooled.shape[2]),
        block=block, grid=grid, stream=stream)

    return target
