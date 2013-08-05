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
                      input_offset=0, target_offset=0, 
                      width=None,
                      target=None, stream=None):

    assert mat.flags.c_contiguous
    assert conv_filter.flags.c_contiguous
    assert bias.flags.c_contiguous
    
    dtype = conv_filter.dtype
    assert dtype in (np.float32, np.float64)
    assert bias.shape[0] == conv_filter.shape[0]

    height, total_width = mat.shape

    if input_offset > 0:
        assert target is not None
        assert width is not None
        assert input_offset + width < total_width
    
    if width is None:
        width = total_width

    n_filters = conv_filter.shape[0]
    width_filter = conv_filter.shape[1] // 4

    block = (_TILE_SIZE_CONV, 1, 1)
    grid = (int(np.ceil(mat.size / float(block[0]))),
            n_filters, 1)
    shared = ((_TILE_SIZE_CONV + width_filter - 1) * np.dtype(dtype).itemsize +
              4 * width_filter * np.dtype(dtype).itemsize)  # filter_shared
    
    if target is None:
        assert target_offset == 0
        total_target_width = n_filters*total_width
        target = gpuarray.empty((height, total_target_width),
                                dtype=dtype)
    else:
        assert target.dtype == dtype
        assert target.flags.c_contiguous
        total_target_width = target.shape[1]
        assert target_offset + n_filters * width <= total_target_width

    dname = _dtype_name[dtype]
    _kernels[dname]['convolve_sequence_kernel'](
        mat, 
        target, 
        conv_filter, 
        bias,
        np.uint32(input_offset),
        np.uint32(width),
        np.uint32(total_width),
        np.uint32(height),
        np.uint32(width_filter),
        np.uint32(total_target_width),
        np.uint32(target_offset),
        np.uint32(n_filters),
        block=block, 
        grid=grid, 
        stream=stream, 
        shared=shared)

    return target

def convolve_sequence_gradient(mat, df_output, filter_width, n_filters,
                               input_offset=0, df_output_offset=0, width=None,
                               target=None, stream=None, block_size=1024):

    assert mat.flags.c_contiguous
    assert df_output.flags.c_contiguous
    
    stride = 4
    dtype = df_output.dtype
    assert dtype in (np.float32, np.float64)

    height, total_width = mat.shape

    if input_offset > 0:
        assert width is not None
        assert input_offset + width <= total_width

    total_df_output_width = df_output.shape[1]

    if df_output_offset > 0:
        assert width is not None
        assert df_output_offset + n_filters * width <= total_df_output_width
    
    if width is None:
        width = total_width
    n_elements = height*width

    block = (block_size, 1, 1)
    grid = (int(np.ceil(n_elements / float(block_size))), n_filters, 1)
    shared = (filter_width - 1 + block_size +  # df_output_share
              stride * block_size              # df_weights_reduce
              ) * np.dtype(dtype).itemsize

    if target is not None:
        assert target.dtype == dtype
        assert target.shape == (n_filters, stride*filter_width, grid[0])
        assert targets.flags.c_contiguous
    else:
        target = gpuarray.empty((n_filters, stride*filter_width,
                                 grid[0]), dtype=dtype)        

    dname = _dtype_name[dtype]
    _kernels[dname]['convolve_sequence_gradient_kernel'](
        mat, 
        df_output, 
        target,
        np.uint32(input_offset),
        np.uint32(df_output_offset),
        np.uint32(total_width),
        np.uint32(total_df_output_width),
        np.uint32(width),
        np.uint32(height),
        np.uint32(filter_width), 
        np.uint32(n_filters),
        block=block, grid=grid, shared=shared, stream=stream)

    if grid[0] > 0:
        target_sum = gpuarray.empty((n_filters, stride*filter_width), dtype)
        block_sum = (max((1, 2**int(np.ceil(np.log2(grid[0])-1)))), 1, 1)
        grid_sum = (n_filters, stride*filter_width, 1)
        shared = block_sum[0] * np.dtype(dtype).itemsize

        _kernels[dname]['gradient_reduce_kernel'](
            target, 
            target_sum,
            np.uint32(n_filters), 
            np.uint32(filter_width),
            np.uint32(grid[0]),
            block=block_sum, grid=grid_sum,
            shared=shared, stream=stream)
    else:
        target_sum = target.reshape((n_filters, stride*filter_width))

    return target_sum

def max_pool(mat, pool_size, n_filters, width=None, 
             input_offset=0, pooled_offset=0,
             target=None, argmax=None, stream=None):
    assert mat.flags.c_contiguous
    
    dtype = mat.dtype
    assert dtype in (np.float32, np.float64)
    assert pool_size <= mat.shape[1] / n_filters

    height, total_width = mat.shape

    if width is None:
        assert input_offset == 0
        # assert pooled_offset == 0
        assert not total_width % n_filters
        width = total_width / n_filters

    pooled_width = int(np.ceil(width / float(pool_size)))

    block = (2**int(np.ceil(np.log2(width))), 1, 1)
    grid = (pooled_width, height, n_filters)
    shared = block[0]*np.dtype(dtype).itemsize + \
      block[0]*np.dtype(np.uint32).itemsize

    if target is not None:
        assert target.dtype == dtype
        assert target.flags.c_contiguous
        total_width_pooled = target.shape[1]
    else:
        total_width_pooled = n_filters*pooled_width
        target = gpuarray.empty(
            (height, total_width_pooled),
            dtype)

    if argmax is not None:
        assert argmax.dtype == np.uint32
        assert argmax.shape == target.shape
        assert argmax.flags.c_contiguous
    else:
        argmax = gpuarray.empty(target.shape, np.uint32)
    
    dname = _dtype_name[dtype]

    _kernels[dname]['max_pool_kernel'](
        mat, 
        target, 
        argmax, 
        np.uint32(input_offset),
        np.uint32(height), 
        np.uint32(total_width),
        np.uint32(width),
        np.uint32(pooled_offset),
        np.uint32(total_width_pooled),
        np.uint32(pool_size),
        block=block, grid=grid, shared=shared, stream=stream)

    return target, argmax

def max_pool_gradient(mat, argmax,
                      df_output, pool_size, n_filters,
                      width=None,
                      width_pooled=None,
                      input_offset=0, pooled_offset=0,
                      target=None, stream=None):
    dtype = mat.dtype
    assert mat.flags.c_contiguous
    assert argmax.flags.c_contiguous
    assert df_output.flags.c_contiguous
    assert dtype in (np.float32, np.float64)

    height, total_width = mat.shape

    if width is None:
        assert input_offset == 0
        # assert pooled_offset == 0
        assert not total_width % n_filters
        width = total_width / n_filters

    total_pooled_width = df_output.shape[1]

    if width_pooled is None:
        assert not total_pooled_width % n_filters
        width_pooled = total_pooled_width / n_filters
        assert pooled_offset == 0

    assert total_pooled_width >= n_filters * width_pooled

    block = (pool_size, 1, 1)
    grid = (width_pooled, n_filters, height)

    if target is not None:
        assert target.dtype == dtype
        assert target.shape == mat.shape
        assert target.flags.c_contiguous
    else:
        target = gpuarray.empty_like(mat)

    dname = _dtype_name[dtype]
    _kernels[dname]['max_pool_gradient_kernel'](
        argmax, 
        df_output, 
        target,
        np.uint32(input_offset),
        np.uint32(height), 
        np.uint32(total_width),
        np.uint32(width), 
        np.uint32(pooled_offset),
        np.uint32(total_pooled_width),
        np.uint32(width_pooled),
        block=block, grid=grid, stream=stream)

    return target

def sum_delta(delta, n_filters, cache_one_vector=True):
    assert delta.flags.c_contiguous
    assert not delta.shape[1] % n_filters
    width = delta.shape[1] / n_filters
    delta_sum_a = matrix_sum_out_axis(delta, 0)
    delta_r = delta_sum_a.reshape((n_filters, width))
    delta_sum_b = matrix_sum_out_axis(delta_r, 1).ravel()
    return delta_sum_b
