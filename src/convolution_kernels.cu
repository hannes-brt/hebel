#include "float.h"
#define CEILING(x) (int)(x) + (1 - (int)((int)((x) + 1) - (x)))

#define TILE_SIZE_CONV {{ TILE_SIZE_CONV }}
#define MAX_WIDTH_FILTER {{ MAX_WIDTH_FILTER }}

__global__ void convolve_sequence(const {{ data_type }} *input,
    {{ data_type }} *target, const {{ data_type }} *filter,
    const unsigned int width, const unsigned int height,
    const unsigned int filter_width, const unsigned int n_filters,
    const unsigned int stride) {

    /* Performs a 1D convolution on each row of a matrix with 
       multiple filters. Filter size must be even and input is
       padded on the right with zeros.
    */
    
    const unsigned int i = blockIdx.y;
    const unsigned int j = blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned int lin_idx = i*width+j;
    const unsigned int row_start = i*width;
    const unsigned int target_width = CEILING((double) width / stride);
    unsigned int shared_idx, input_idx;    
    
    const unsigned int shared_width = TILE_SIZE_CONV+MAX_WIDTH_FILTER-1;
    __shared__ {{ data_type }} input_shared[shared_width];
    
    const unsigned int halo_width = filter_width - 1;
    
    shared_idx = threadIdx.x;
    input_shared[shared_idx] = (j < width && i < height) ? input[lin_idx] : 0;
    __syncthreads();

    if (i < height) {
        int halo_index_right = (blockIdx.x+1)*blockDim.x+threadIdx.x;
        if (threadIdx.x < halo_width) {
            shared_idx = blockDim.x+threadIdx.x;
            input_idx = row_start+halo_index_right;
            input_shared[shared_idx] =
                (halo_index_right >= width) ? 0 : input[input_idx];
        }
    }
    __syncthreads();
  
    unsigned int filter_idx, target_idx;
    if (!(j%stride) && i < height && j < width) {
        for (int f=0; f < n_filters; f++) {
            {{ data_type }} Pvalue = 0.;
            for (int k=0; k < filter_width; k++) {
                shared_idx = threadIdx.x+k;
                filter_idx = f*filter_width+k;
                Pvalue += input_shared[shared_idx]*filter[filter_idx];
            }
            target_idx = f*target_width*height+i*target_width+j/stride;
            target[target_idx] = Pvalue;
        }        
    }
}

__global__ void gradient_reduce(const {{ data_type }} *df_weights,
    {{ data_type }} *df_weights_sum, const unsigned int n_filters,
    const unsigned int filter_width, const unsigned int n_elements) {

    /* Completes the reduction operation of conv1d_grad_weight
    */
    
    const unsigned int tid = threadIdx.x;
    const unsigned int df_weights_idx = blockIdx.x*filter_width*n_elements+
        blockIdx.y*n_elements+threadIdx.x;
    
    extern __shared__ {{ data_type }} sdata[];
    
    sdata[tid] = (tid<n_elements) ? df_weights[df_weights_idx] : 0;
    if (tid+blockDim.x < n_elements)
        sdata[tid] += df_weights[df_weights_idx+blockDim.x];
    __syncthreads();
    
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            sdata[tid] += sdata[tid+s];
        }
        __syncthreads();
    }
    
    if (tid==0) {        
        const unsigned int df_weights_sum_idx = blockIdx.x*filter_width+blockIdx.y;
        df_weights_sum[df_weights_sum_idx] = sdata[0];
    }
}

__global__ void convolve_sequence_gradient(
  const {{ data_type }} *input, const {{ data_type }} *df_output,
  {{ data_type }} *df_weights, const unsigned int width,
  const unsigned int height, const unsigned int filter_width,
  const unsigned int n_filters) {
  
  const unsigned int stride = 4;
  const unsigned int tx = threadIdx.x;
  const unsigned int input_idx = blockIdx.x*blockDim.x+tx;
  const unsigned int column = input_idx % width;
  const unsigned int len_input = height*width;
  const unsigned int len_df_output = len_input / stride;

  unsigned int df_weights_idx, output_idx, shared_idx, df_output_offset;
  int halo_idx;
  
  const unsigned int halo_width = (filter_width / stride) - 1;
  const unsigned int shared_width = halo_width + blockDim.x / stride;
  extern __shared__ {{ data_type }} sdata[];
  {{ data_type }} *df_output_shared = sdata;
  {{ data_type }} *df_weights_reduce = df_output_shared + shared_width;

  const {{ data_type }} input_element = 
    (input_idx < len_input) ? input[input_idx] : 0.;

  for (unsigned int f=0; f < n_filters; f++) {
    // Load halo elements on the left
    halo_idx = (blockIdx.x-1)*blockDim.x+tx;
    if (tx < halo_width) {
      output_idx = f*len_df_output+blockIdx.x*blockDim.x/stride+tx-halo_width;
      shared_idx = tx;
      df_output_shared[shared_idx] = 
	(halo_idx < 0) ? 0. : df_output[output_idx];
    }

    if (tx < blockDim.x/stride) {
      output_idx = f*len_df_output+blockIdx.x*blockDim.x/stride+tx;
      df_output_shared[tx+halo_width] = 
	((blockIdx.x*blockDim.x/stride+tx) < len_df_output) ? 
	df_output[output_idx] : 0.;
    }

    __syncthreads();

    for (unsigned int k=0; k<(halo_width+1); k++) {
      df_output_offset = (halo_width-k)*stride;
      df_weights_reduce[tx] =
      	(column >= df_output_offset)
      	? input_element * df_output_shared[tx/stride+k] : 0.;
      __syncthreads();

      for (unsigned int s=blockDim.x/2; s>=stride; s>>=1) {
	if (tx<s) {
	  df_weights_reduce[tx] += df_weights_reduce[tx+s];
	}
	__syncthreads();
      }

      if (tx<stride) {
	df_weights_idx = 
	  f*filter_width*gridDim.x+
	  (tx+df_output_offset)*gridDim.x+
	  blockIdx.x;
	df_weights[df_weights_idx] = df_weights_reduce[tx];
      }
    }
  }
}

__global__ void max_pool(const {{ data_type }} *mat,
    {{ data_type }} *target, 
    const unsigned int height,
    const unsigned int width,
    const unsigned int pool_size) {

    /* Perform 1D max-pooling on all rows of a matrix
    */
    
    const unsigned int tx = threadIdx.x;
    const unsigned int i = blockIdx.y;
    const unsigned int j = blockIdx.x*pool_size+tx;
    const unsigned int mat_idx = blockIdx.z*height*width+i*width+j;
    
    extern __shared__ {{ data_type }} sdata[];
    
    sdata[tx] = (i < height && j < width && tx < pool_size) ? mat[mat_idx] : -FLT_MAX;
    __syncthreads();
    
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tx<s && sdata[tx+s] > sdata[tx]) {
            sdata[tx] = sdata[tx+s];
        }
        __syncthreads();
    }
    
    if (tx==0) {
      const unsigned int target_idx = blockIdx.y*gridDim.z*gridDim.x+
	blockIdx.z*gridDim.x+blockIdx.x;
        target[target_idx] = sdata[0];
    }
}

__global__ void max_pool_gradient(
    const {{ data_type }} *mat,
    const {{ data_type }} *mat_pooled,
    const {{ data_type }} *df_output,
    {{ data_type }} *df_input,
    const unsigned int height,
    const unsigned int width,
    const unsigned int width_pooled) {

    /* Gradient of max-pooling operation
    */
    
    const unsigned int tx = threadIdx.x;
    const unsigned int bx = blockIdx.x;
    const unsigned int by = blockIdx.y;
    const unsigned int bz = blockIdx.z;
    const unsigned int n_filters = gridDim.y;
    const unsigned int lin_idx = bz*n_filters*width+by*width+bx*blockDim.x+tx;
    
    const {{ data_type }} max_element = mat_pooled[bz*n_filters*width_pooled+
						   by*width_pooled+bx];
    const {{ data_type }} df_output_element = df_output[bz*n_filters*width_pooled+
							by*width_pooled+bx];

    if (bx*blockDim.x+tx < width) {
        df_input[bz*n_filters*width+by*width+bx*blockDim.x+tx] =
            (mat[lin_idx] == max_element) ? df_output_element : 0.;
    }
}
