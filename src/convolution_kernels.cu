#include <float.h>
#include <limits.h>
#include "convolution_kernels.h"

__global__ void convolve_sequence(const nucleotide_t *input,
				  {{ data_type }} *target, 
				  const {{ data_type }} *filter,
				  const {{ data_type }} *bias, 
				  const unsigned int input_offset,
				  const unsigned int width,
				  const unsigned int total_width,
				  const unsigned int height, 
				  const unsigned int filter_width, 
				  const unsigned int n_filters) {

  /* Performs a 1D convolution on each row of a matrix with 
     multiple filters. Filter size must be even and input is
     padded on the right with zeros.
  */
    
  const unsigned int f = blockIdx.y;
  const unsigned int lin_idx = blockIdx.x*blockDim.x+threadIdx.x;
  const unsigned int i = lin_idx / width;
  const unsigned int j = lin_idx % width;
  const unsigned int row_start = i*width;
  const unsigned int filter_elements = 4*filter_width; // Actual number of elements in filter
  const {{ data_type }} bias_filter = bias[f];
  unsigned int shared_idx, input_idx, target_idx;
  nucleotide_t nt;
    
  const unsigned int shared_width = (blockDim.x+filter_width-1);
  extern __shared__ {{ data_type }} sdata[];
  nucleotide_t *input_shared = (nucleotide_t*) sdata;
  {{ data_type }} *filter_shared = sdata + shared_width;
    
  const unsigned int halo_width = filter_width - 1;
    
  shared_idx = threadIdx.x;
  input_idx = i*total_width + input_offset + j;
  input_shared[shared_idx] = (i < height) ? input[input_idx] : DNA_N;
  __syncthreads();

  if (i < height) {
    int halo_index_right = (blockIdx.x+1)*blockDim.x+threadIdx.x;
    if (threadIdx.x < halo_width) {
      shared_idx = blockDim.x+threadIdx.x;
      input_idx = row_start+halo_index_right;
      input_shared[shared_idx] =
	(halo_index_right >= width) ? DNA_N : input[input_idx];
    }
  }

  if (threadIdx.x < filter_elements)
    filter_shared[threadIdx.x] = filter[f*filter_elements+threadIdx.x];
  __syncthreads();
  
  if (i < height) {
    {{ data_type }} Pvalue = bias_filter;
    for (int k=0; k < filter_width; k++) {
      if (j+k < width) {
	shared_idx = threadIdx.x+k;
	nt = input_shared[shared_idx];
      
	if (CHECK_NT(nt, DNA_A))
	  Pvalue += filter_shared[4*k];

	if (CHECK_NT(nt, DNA_C))
	  Pvalue += filter_shared[4*k+1];

	if (CHECK_NT(nt, DNA_G))
	  Pvalue += filter_shared[4*k+2];

	if (CHECK_NT(nt, DNA_T))
	  Pvalue += filter_shared[4*k+3];
      }
    }
    target_idx = i*n_filters*total_width + f*total_width + input_offset + j;
    target[target_idx] = Pvalue;
  }
}

__global__ void gradient_reduce(const {{ data_type }} *df_weights,
				{{ data_type }} *df_weights_sum, 
				const unsigned int n_filters,
				const unsigned int filter_width, 
				const unsigned int n_elements) {

    /* Completes the reduction operation of conv1d_grad_weight
    */
    
    const unsigned int tid = threadIdx.x;
    const unsigned int filter_elements = 4*filter_width;
    const unsigned int df_weights_idx = blockIdx.x*filter_elements*n_elements+
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
        const unsigned int df_weights_sum_idx = blockIdx.x*filter_elements+blockIdx.y;
        df_weights_sum[df_weights_sum_idx] = sdata[0];
    }
}

__global__ void convolve_sequence_gradient(const nucleotide_t *input, 
					   const {{ data_type }} *df_output,
					   {{ data_type }} *df_weights, 
					   const unsigned int input_offset,
					   const unsigned int total_width,
					   const unsigned int width,
					   const unsigned int height, 
					   const unsigned int filter_width,
					   const unsigned int n_filters) {
  
  const unsigned int stride = 4;
  const unsigned int tx = threadIdx.x;
  const unsigned int f = blockIdx.y;
  const unsigned int lin_idx = blockIdx.x*blockDim.x+tx;
  const unsigned int row = lin_idx / width;
  const unsigned int column = lin_idx % width;
  const unsigned int input_idx = row*total_width + input_offset + column;
  const unsigned int column_start_block = (blockIdx.x*blockDim.x)%width; // Column of first thread in block
  const unsigned int row_start_block = (blockIdx.x*blockDim.x)/width; // Row of first thread in block
  const unsigned int len_input = height*width;

  unsigned int df_weights_idx, output_idx, shared_idx, df_output_offset;
  int halo_idx;
  
  const unsigned int halo_width = filter_width - 1;
  const unsigned int shared_width = halo_width + blockDim.x;
  extern __shared__ {{ data_type }} sdata[];
  {{ data_type }} *df_output_shared = sdata;
  {{ data_type }} *df_weights_reduce = df_output_shared + shared_width;

  const nucleotide_t input_element = 
    (lin_idx < len_input) ? input[input_idx] : DNA_N;

  // Load halo elements on the left
  if (tx < halo_width) {
    output_idx = row_start_block*n_filters*total_width +
      f*total_width + input_offset + column_start_block - halo_width + tx;
    shared_idx = tx;
    halo_idx = column_start_block - halo_width + tx;
    df_output_shared[shared_idx] = 
      (halo_idx < 0) ? 0. : df_output[output_idx];
  }

  if (tx < blockDim.x) {
    output_idx = row*n_filters*total_width + f*total_width + input_offset + column;
    df_output_shared[tx+halo_width] = 
      (column < width && row < height) ?
      df_output[output_idx] : 0.;
  }
  
  __syncthreads();

  for (unsigned int k=0; k<filter_width; k++) {
    df_output_offset = halo_width-k;

    if (column >= df_output_offset) {
      df_weights_reduce[stride*tx] = (CHECK_NT(input_element, DNA_A)) ?
    	df_output_shared[tx+k] : 0.;

      df_weights_reduce[stride*tx+1] = (CHECK_NT(input_element, DNA_C)) ?
    	df_output_shared[tx+k] : 0.;

      df_weights_reduce[stride*tx+2] = (CHECK_NT(input_element, DNA_G)) ?
    	df_output_shared[tx+k] : 0.;

      df_weights_reduce[stride*tx+3] = (CHECK_NT(input_element, DNA_T)) ?
    	df_output_shared[tx+k] : 0.;

    } else {
      df_weights_reduce[stride*tx] = 0.;
      df_weights_reduce[stride*tx+1] = 0.;
      df_weights_reduce[stride*tx+2] = 0.;
      df_weights_reduce[stride*tx+3] = 0.;
    }

    __syncthreads();

    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
      if (tx<s) {
    	df_weights_reduce[stride*tx] += df_weights_reduce[stride*(tx+s)];
    	df_weights_reduce[stride*tx+1] += df_weights_reduce[stride*(tx+s)+1];
    	df_weights_reduce[stride*tx+2] += df_weights_reduce[stride*(tx+s)+2];
    	df_weights_reduce[stride*tx+3] += df_weights_reduce[stride*(tx+s)+3];
      }
      __syncthreads();
    }
    
    if (tx<stride) {
      df_weights_idx =
      	f * stride * filter_width * gridDim.x +
      	(tx + stride * df_output_offset) * gridDim.x +
      	blockIdx.x;
      df_weights[df_weights_idx] = df_weights_reduce[tx];
    }
  }
}

__global__ void max_pool(const {{ data_type }} *mat,
			 {{ data_type }} *target, 
			 unsigned int *argmax,
			 const unsigned int input_offset,
			 const unsigned int height,
			 const unsigned int total_width,
			 const unsigned int width,
			 const unsigned int pooled_offset,
			 const unsigned int total_width_pooled,
			 const unsigned int pool_size) {

  /* Perform 1D max-pooling on all rows of a matrix
   */
    
  const unsigned int tx = threadIdx.x;
  const unsigned int i = blockIdx.y;
  const unsigned int j = blockIdx.x*pool_size+tx;
  const unsigned int f = blockIdx.z;
  const unsigned int n_filters = gridDim.z;
  const unsigned int mat_idx = i*n_filters*total_width + f*total_width + input_offset + j;
    
  extern __shared__ {{ data_type }} sdata[];
  {{ data_type }} *max_shared = sdata;
  unsigned int *argmax_shared = (unsigned int*) (max_shared + blockDim.x);
    
  max_shared[tx] = (i < height && j < width && tx < pool_size) ? mat[mat_idx] : -FLT_MAX;
  argmax_shared[tx] = (i < height && j < width && tx < pool_size) ? j : UINT_MAX;
  __syncthreads();
    
  for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
    if (tx<s && sdata[tx+s] > sdata[tx]) {
      max_shared[tx] = max_shared[tx+s];
      argmax_shared[tx] = argmax_shared[tx+s];
    }
    __syncthreads();
  }
    
  if (tx==0) {
    const unsigned int target_idx = i*n_filters*total_width_pooled +
      f*total_width_pooled + pooled_offset + blockIdx.x;
    target[target_idx] = max_shared[0];
    argmax[target_idx] = argmax_shared[0];
  }
}

__global__ void max_pool_gradient(
    const unsigned int *argmax,
    const {{ data_type }} *df_output,
    {{ data_type }} *df_input,
    const unsigned int input_offset,
    const unsigned int height,
    const unsigned int total_width,
    const unsigned int width,
    const unsigned int pooled_offset,
    const unsigned int total_width_pooled,
    const unsigned int width_pooled) {

    /* Gradient of max-pooling operation
    */
    
    const unsigned int tx = threadIdx.x;
    const unsigned int bx = blockIdx.x;
    const unsigned int f = blockIdx.y;
    const unsigned int row = blockIdx.z;
    const unsigned int n_filters = gridDim.y;
    const unsigned int column = bx*blockDim.x+tx;
    
    const unsigned int pooled_idx = row*n_filters*total_width_pooled +
      f*total_width_pooled + pooled_offset + bx;
    const unsigned int max_idx = argmax[pooled_idx];
    const {{ data_type }} df_output_element = df_output[pooled_idx];

    if (column < width) {
        df_input[row*n_filters*total_width +
		 f*total_width + input_offset + 
		 column] =
            (column == max_idx) ? df_output_element : 0.;
    }
}
