from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np

CONV_BLOCK_SIZE = (8, 8, 1)
MAX_WIDTH_FILTER = 15
MAX_NUM_FILTERS = 100
assert CONV_BLOCK_SIZE[0] == CONV_BLOCK_SIZE[1]

code = """
#define CEILING(x) (int)(x) + (1 - (int)((int)((x) + 1) - (x)))

#define TILE_SIZE %(TILE_SIZE)d
#define MAX_WIDTH_FILTER %(MAX_WIDTH_FILTER)d
#define MAX_NUM_FILTERS %(MAX_NUM_FILTERS)d

__global__ void conv1d_matrix_mult_filter(const %(data_type)s *input,
    %(data_type)s *target, const %(data_type)s *filter,
    const unsigned int width, const unsigned int height,
    const unsigned int filter_width, const unsigned int n_filters,
    const unsigned int stride) {
    
    const unsigned int i = blockIdx.y*blockDim.y+threadIdx.y;
    const unsigned int j = blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned int lin_idx = i*width+j;
    const unsigned int row_start = i*width;
    const unsigned int target_width = CEILING((double) width / stride);
    unsigned int shared_idx, input_idx;    
    
    const unsigned int shared_width = TILE_SIZE+MAX_WIDTH_FILTER-1;
    __shared__ %(data_type)s input_shared[TILE_SIZE*(shared_width)];
    
    const unsigned int halo_width = filter_width / 2;
    
    if (i < height) {
        int halo_index_left = (blockIdx.x-1)*blockDim.x+threadIdx.x;
        if (threadIdx.x >= blockDim.x-halo_width) {
            shared_idx = threadIdx.y*shared_width + 
                threadIdx.x-(blockDim.x-halo_width);
            input_idx = row_start + halo_index_left;
            input_shared[shared_idx] = 
                (halo_index_left < 0) ? 0 : input[input_idx];
        }
    }
    
    shared_idx = threadIdx.y*shared_width+halo_width+threadIdx.x;
    input_shared[shared_idx] = (j < width && i < height) ? input[lin_idx] : 0;
       
    if (i < height) {
        int halo_index_right = (blockIdx.x+1)*blockDim.x+threadIdx.x;
        if (threadIdx.x < halo_width) {
            shared_idx = threadIdx.y*shared_width+blockDim.x+threadIdx.x+halo_width;
            input_idx = row_start+halo_index_right;
            input_shared[shared_idx] =
                (halo_index_right >= width) ? 0 : input[input_idx];
        }
    }
    __syncthreads();
  
    unsigned int filter_idx, target_idx;
    if (!(j%%stride) && i < height && j < width) {
        for (int f=0; f < n_filters; f++) {
            %(data_type)s Pvalue = 0.;
            for (int k=0; k < filter_width; k++) {
                shared_idx = threadIdx.y*shared_width+threadIdx.x+k;
                filter_idx = f*filter_width+k;
                Pvalue += input_shared[shared_idx]*filter[filter_idx];
            }
            target_idx = f*target_width*height+i*target_width+j/stride;
            target[target_idx] = Pvalue;
        }        
    }
}
"""

source_module_float = SourceModule(code % {'TILE_SIZE': CONV_BLOCK_SIZE[0],
                                           'MAX_WIDTH_FILTER': MAX_WIDTH_FILTER,
                                           'MAX_NUM_FILTERS': MAX_NUM_FILTERS,
                                           'data_type': 'float'})
source_module_double = SourceModule(code % {'TILE_SIZE': CONV_BLOCK_SIZE[0],
                                            'MAX_WIDTH_FILTER': MAX_WIDTH_FILTER,
                                            'MAX_NUM_FILTERS': MAX_NUM_FILTERS,
                                            'data_type': 'double'})

conv1d_matrix_mult_filter_kernel_float = source_module_float.get_function('conv1d_matrix_mult_filter')
conv1d_matrix_mult_filter_kernel_double = source_module_double.get_function('conv1d_matrix_mult_filter')

def conv1d_matrix_mult_filter(mat, conv_filter, stride=1, target=None, stream=None):
    dtype = mat.dtype
    assert dtype in (np.float32, np.float64)
    assert conv_filter.dtype == dtype
    
    if target is not None:
        assert target.dtype == dtype
        assert mat.shape // stride == target.shape[:2]

    # import pudb
    # pudb.set_trace()
        
    height, width = mat.shape
    n_filters, width_filter = conv_filter.shape
    
    block = CONV_BLOCK_SIZE
    grid = (int(np.ceil(mat.shape[1] / float(block[0]))),
            int(np.ceil(mat.shape[0] / float(block[1]))),
            1)
    
    if target is None:
        target_width = int(np.ceil(width / float(stride)))
        target = gpuarray.empty((n_filters, height, target_width),
                                dtype=dtype)

    if dtype == np.float32:
        kernel = conv1d_matrix_mult_filter_kernel_float
    elif dtype == np.float64:
        kernel = conv1d_matrix_mult_filter_kernel_double
    else:
        raise ValueError

    kernel(mat, target, conv_filter,
           np.int32(width), np.int32(height),
           np.int32(width_filter),
           np.int32(n_filters),
           np.int32(stride),
           block=block, grid=grid,
           stream=stream)
        
    return target
