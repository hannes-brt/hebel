from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
import os
from jinja2 import Template
from .. import neural_nets_root

TILE_SIZE_CONV = 128
TILE_SIZE_GRAD_CONV = 32
MAX_WIDTH_FILTER = 15
MAX_NUM_FILTERS = 100

code = Template(open(os.path.join(neural_nets_root, 
    'src', 'convolution_kernels.cu')).read())

source_modules = {dtype: SourceModule(code.render(TILE_SIZE_CONV=TILE_SIZE_CONV,
                                                  TILE_SIZE_GRAD_CONV=TILE_SIZE_GRAD_CONV,
                                                  MAX_WIDTH_FILTER=MAX_WIDTH_FILTER,
                                                  MAX_NUM_FILTERS=MAX_NUM_FILTERS,
                                                  data_type=dtype))
                  for dtype in ('float', 'double')}

kernels = {dtype: {f_name + '_kernel': sm.get_function(f_name)
                   for f_name in ('conv1d_matrix_mult_filter',
                                  'conv1d_sequence',
                                  'conv1d_grad_weights',
                                  'conv1d_grad_weights_sequence',
                                  'conv1d_grad_weights_sum',
                                  'max_pool',
                                  'max_pool_grad')}
           for dtype, sm in source_modules.iteritems()}

dtype_name = {np.dtype(np.float32): 'float', np.dtype(np.float64): 'double'}

def conv1d_matrix_mult_filter(mat, conv_filter, stride=1, 
                              sequence_conv=False,
                              target=None, stream=None):
    dtype = mat.dtype
    assert dtype in (np.float32, np.float64)
    assert conv_filter.dtype == dtype
    
    if target is not None:
        assert target.dtype == dtype
        assert mat.shape // stride == target.shape[:2]

    height, width = mat.shape
    n_filters, width_filter = conv_filter.shape
    
    block = (TILE_SIZE_CONV, 1, 1)
    grid = (int(np.ceil(mat.shape[1] / float(block[0]))),
            int(np.ceil(mat.shape[0] / float(block[1]))),
            1)
    
    if target is None:
        target_width = int(np.ceil(width / float(stride)))
        target = gpuarray.empty((n_filters, height, target_width),
                                dtype=dtype)

    if sequence_conv:
        kernel_name = 'conv1d_sequence_kernel'
    else:
        kernel_name = 'conv1d_matrix_mult_filter_kernel'

    dname = dtype_name[dtype]
    kernels[dname][kernel_name](
        mat, target, conv_filter,
        np.uint32(width), np.uint32(height),
        np.uint32(width_filter),
        np.uint32(n_filters),
        np.uint32(stride),
        block=block, grid=grid,
        stream=stream)
        
    return target

def conv1d_grad_weights(mat, df_output, filter_width, n_filters,
                        sequence_conv=False,
                        target=None, stream=None):
    dtype = mat.dtype
    assert dtype in (np.float32, np.float64)
    assert df_output.dtype == dtype

    height, width = mat.shape

    block = (TILE_SIZE_GRAD_CONV, TILE_SIZE_GRAD_CONV, 1)
    grid = (int(np.ceil(mat.shape[1] / float(block[0]))),
            int(np.ceil(mat.shape[0] / float(block[1]))),
            1)

    if target is not None:
        assert target.dtype == dtype
        assert target.shape == (n_filters, filter_width)
    else:
        target = gpuarray.empty((n_filters, filter_width, 
                                 grid[1], grid[0]), dtype=dtype)

    dname = dtype_name[dtype]

    if sequence_conv:
        kernel_name = 'conv1d_grad_weights_sequence_kernel'
    else:
        kernel_name = 'conv1d_grad_weights_kernel'  

    kernels[dname][kernel_name](
        mat, df_output, target,
        np.uint32(width), np.uint32(height),
        np.uint32(filter_width), np.uint32(n_filters),
        block=block, grid=grid, stream=stream)

    sum_height = grid[1]
    sum_width = grid[0]
    target_sum = gpuarray.empty((n_filters, filter_width), dtype)
    block_sum = (2**int(np.ceil(np.log2(sum_height*sum_width)-1)), 1, 1)
    grid_sum = (n_filters, filter_width, 1)
    shared = block_sum[0]*np.dtype(dtype).itemsize

    kernels[dname]['conv1d_grad_weights_sum_kernel'](
        target, target_sum,
        np.uint32(n_filters), np.uint32(filter_width),
        np.uint32(sum_height*sum_width),
        block=block_sum, grid=grid_sum, 
        shared=shared, stream=stream)

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
    
    dname = dtype_name[dtype]

    kernels[dname]['max_pool_kernel'](
        mat, target, np.uint32(height), np.uint32(width), np.uint32(pool_size),
        block=block, grid=grid, shared=shared, stream=stream)

    return target 

def max_pool_grad(mat, mat_pooled, df_output, pool_size, target=None, stream=None):
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

    dname = dtype_name[dtype]
    kernels[dname]['max_pool_grad_kernel'](
        mat, mat_pooled, df_output, target, np.uint32(height),
        np.uint32(width), np.uint32(mat_pooled.shape[2]),
        block=block, grid=grid, stream=stream)

    return target
