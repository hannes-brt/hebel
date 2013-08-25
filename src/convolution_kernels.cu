// Copyright (C) 2013  Hannes Bretschneider

// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License along
// with this program; if not, write to the Free Software Foundation, Inc.,
// 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

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
				  const unsigned int total_target_width,
				  const unsigned int target_offset,
				  const unsigned int n_filters) {

  /*
    Convolves a set of filters with an input sequence. This function
    can operate on a subset of columns of the input array and output
    the result to a subset of columns of the output array.

    *input : pointer to the input sequence
    *target : pointer to the output array
    *filter : pointer to the filter bank
    *bias : pointer to the array of biases
    input_offset : column offset when working 
      on a subset of the input array
    width : width of input sequence
    total_width: total width/row stride of the input sequence 
      (for operating on a subset of the input)
    height : number of rows in the input array
    filter_width : width of the filters in nucleotides
    total_target_width : total width/row stride of the target array
    target_offset : column offset into the target array
    n_filters : number of filters

  */
    
  const unsigned int f = blockIdx.y;
  const unsigned int lin_idx = blockIdx.x*blockDim.x+threadIdx.x;
  const unsigned int i = lin_idx / width;
  const unsigned int j = lin_idx % width;
  const unsigned int row_start = i*total_width + input_offset;
  const unsigned int filter_elements = STRIDE*filter_width; // Actual number of elements in filter
  const {{ data_type }} bias_filter = bias[f];
  unsigned int shared_idx, input_idx, target_idx;
  nucleotide_t nt;
    
  const unsigned int shared_width = (blockDim.x+filter_width-1);
  extern __shared__ {{ data_type }} sdata[];
  nucleotide_t *input_shared = (nucleotide_t*) sdata;
  {{ data_type }} *filter_shared = sdata + shared_width;
    
  const unsigned int halo_width = filter_width - 1;
    
  // Load input into shared memory
  shared_idx = threadIdx.x;
  input_idx = i*total_width + input_offset + j;
  input_shared[shared_idx] = (i < height) ? input[input_idx] : DNA_N;
  __syncthreads();

  // Load halo elements on right side
  if (i < height) {
    int halo_index_right = (blockIdx.x+1)*blockDim.x+threadIdx.x;
    if (threadIdx.x < halo_width) {
      shared_idx = blockDim.x+threadIdx.x;
      input_idx = row_start+halo_index_right;
      input_shared[shared_idx] =
	(halo_index_right >= width) ? DNA_N : input[input_idx];
    }
  }

  // Load filter elements into shared memory
  if (threadIdx.x < filter_elements)
    filter_shared[threadIdx.x] = filter[f*filter_elements+threadIdx.x];
  __syncthreads();
  
  // Perform convolution
  if (i < height) {
    {{ data_type }} Pvalue = bias_filter;
    for (int k=0; k < filter_width; k++) {
      if (j+k < width) {
	shared_idx = threadIdx.x+k;
	nt = input_shared[shared_idx];
      
	if (CHECK_NT(nt, DNA_A))
	  Pvalue += filter_shared[STRIDE*k];

	if (CHECK_NT(nt, DNA_C))
	  Pvalue += filter_shared[STRIDE*k+1];

	if (CHECK_NT(nt, DNA_G))
	  Pvalue += filter_shared[STRIDE*k+2];

	if (CHECK_NT(nt, DNA_T))
	  Pvalue += filter_shared[STRIDE*k+3];

	if (CHECK_NT(nt, DNA_R)) {
	  Pvalue += .5 * filter_shared[STRIDE*k];
	  Pvalue += .5 * filter_shared[STRIDE*k+2];
	}

	if (CHECK_NT(nt, DNA_Y)) {
	  Pvalue += .5 * filter_shared[STRIDE*k+1];
	  Pvalue += .5 * filter_shared[STRIDE*k+3];
	}
      }
    }

    // Write output
    target_idx = i*total_target_width + target_offset + f*width + j;
    target[target_idx] = Pvalue;
  }
}

__global__ void gradient_reduce(const {{ data_type }} *df_weights,
				{{ data_type }} *df_weights_sum, 
				const unsigned int n_filters,
				const unsigned int filter_width, 
				const unsigned int n_elements) {

  /* 
     Reduction operation necessary to complete the gradient computation
  */
    
  const unsigned int tid = threadIdx.x;
  const unsigned int filter_elements = STRIDE*filter_width;
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
					   const unsigned int df_output_offset,
					   const unsigned int total_input_width,
					   const unsigned int total_df_output_width,
					   const unsigned int width,
					   const unsigned int height, 
					   const unsigned int filter_width,
					   const unsigned int n_filters) {

  /*
    Compute the gradient of the convolution operation with respect to the filter weights
    
    *input : pointer to the input sequence
    *df_output : pointer to the incoming gradient from the next layer
    *df_weights : pointer to output array for gradient wrt filter weights
    input_offset : column offset into input array
    df_output_offset : column offset into df_output
    total_input_width : total_width/row stride of input array
    total_df_output_width : total_width/row stride of df_output
    width : input width
    height : number of input rows
    filter_width : width of filters
    n_filters : number of filters in filter bank

  */

  
  const unsigned int tx = threadIdx.x;
  const unsigned int f = blockIdx.y;
  const unsigned int lin_idx = blockIdx.x*blockDim.x+tx;
  const unsigned int row = lin_idx / width;
  const unsigned int column = lin_idx % width;
  const unsigned int input_idx = row*total_input_width + input_offset + column;
  const unsigned int column_start_block = (blockIdx.x*blockDim.x)%width; // Column of first thread in block
  const unsigned int row_start_block = (blockIdx.x*blockDim.x)/width; // Row of first thread in block
  const unsigned int len_input = height*width;

  unsigned int df_weights_idx, output_idx, shared_idx, df_output_shift;
  int halo_idx;

  // Define dynamically sized shared memory
  const unsigned int halo_width = filter_width - 1;
  const unsigned int shared_width = halo_width + blockDim.x;
  extern __shared__ {{ data_type }} sdata[];
  {{ data_type }} *df_output_shared = sdata;
  {{ data_type }} *df_weights_reduce = df_output_shared + shared_width;

  // Load input element
  const nucleotide_t input_element = 
    (lin_idx < len_input) ? input[input_idx] : DNA_N;

  // Load halo elements on the left into shared memory
  if (tx < halo_width) {
    output_idx = row_start_block*total_df_output_width +
      df_output_offset + f*width + column_start_block - halo_width + tx;
    shared_idx = tx;
    halo_idx = column_start_block - halo_width + tx;
    df_output_shared[shared_idx] = 
      (halo_idx < 0) ? 0. : df_output[output_idx];
  }

  // Load remaining shared memory elements
  if (tx < blockDim.x) {
    output_idx = row*total_df_output_width + df_output_offset + f*width + column;
    df_output_shared[tx+halo_width] = 
      (column < width && row < height) ?
      df_output[output_idx] : 0.;
  }
  
  __syncthreads();

  // Compute gradients
  for (unsigned int k=0; k<filter_width; k++) {
    df_output_shift = halo_width-k;

    if (column >= df_output_shift) {
      df_weights_reduce[STRIDE*tx] = (CHECK_NT(input_element, DNA_A)) ?
    	df_output_shared[tx+k] : 0.;

      df_weights_reduce[STRIDE*tx+1] = (CHECK_NT(input_element, DNA_C)) ?
    	df_output_shared[tx+k] : 0.;

      df_weights_reduce[STRIDE*tx+2] = (CHECK_NT(input_element, DNA_G)) ?
    	df_output_shared[tx+k] : 0.;

      df_weights_reduce[STRIDE*tx+3] = (CHECK_NT(input_element, DNA_T)) ?
    	df_output_shared[tx+k] : 0.;

      if (CHECK_NT(input_element, DNA_R)) {
	df_weights_reduce[STRIDE*tx] = .5 * df_output_shared[tx+k];
	df_weights_reduce[STRIDE*tx+2] = .5 * df_output_shared[tx+k];
      }
    
      if (CHECK_NT(input_element, DNA_Y)) {
	df_weights_reduce[STRIDE*tx+1] = .5 * df_output_shared[tx+k];
	df_weights_reduce[STRIDE*tx+3] = .5 * df_output_shared[tx+k];
      }

    } else {
      df_weights_reduce[STRIDE*tx] = 0.;
      df_weights_reduce[STRIDE*tx+1] = 0.;
      df_weights_reduce[STRIDE*tx+2] = 0.;
      df_weights_reduce[STRIDE*tx+3] = 0.;
    }

    __syncthreads();

    // Stage 1 reduction
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
      if (tx<s) {
	df_weights_reduce[STRIDE*tx] += df_weights_reduce[STRIDE*(tx+s)];
	df_weights_reduce[STRIDE*tx+1] += df_weights_reduce[STRIDE*(tx+s)+1];
	df_weights_reduce[STRIDE*tx+2] += df_weights_reduce[STRIDE*(tx+s)+2];
	df_weights_reduce[STRIDE*tx+3] += df_weights_reduce[STRIDE*(tx+s)+3];
      }
      __syncthreads();
    }
    
    // Write output
    if (tx<STRIDE) {
      df_weights_idx =
	f * STRIDE * filter_width * gridDim.x +
	(tx + STRIDE * df_output_shift) * gridDim.x +
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
  const unsigned int mat_idx = i*total_width + input_offset + f*width + j;
  const unsigned int width_pooled = CEILING(({{ data_type }}) width / pool_size);
    
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
    const unsigned int target_idx = i*total_width_pooled +
      pooled_offset + f*width_pooled + blockIdx.x;
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
  const unsigned int column = bx*blockDim.x+tx;
    
  const unsigned int pooled_idx = row*total_width_pooled +
    pooled_offset + f*width_pooled + bx;
  const unsigned int max_idx = argmax[pooled_idx];
  const {{ data_type }} df_output_element = df_output[pooled_idx];

  if (column < width) {
    df_input[row*total_width + input_offset +
	     f*width + column] =
      (column == max_idx) ? df_output_element : 0.;
  }
}

__global__ void sum_pool(const {{ data_type }} *mat,
			 {{ data_type }} *target, 
			 const unsigned int input_offset,
			 const unsigned int height,
			 const unsigned int total_width,
			 const unsigned int width,
			 const unsigned int pooled_offset,
			 const unsigned int total_width_pooled,
			 const unsigned int pool_size) {

  /* Perform 1D sum-pooling on all rows of a matrix
   */
    
  const unsigned int tx = threadIdx.x;
  const unsigned int i = blockIdx.y;
  const unsigned int j = blockIdx.x*pool_size+tx;
  const unsigned int f = blockIdx.z;
  const unsigned int mat_idx = i*total_width + input_offset + f*width + j;
  const unsigned int width_pooled = CEILING(({{ data_type }}) width / pool_size);
    
  extern __shared__ {{ data_type }} sdata[];
  {{ data_type }} *sum_shared = sdata;
    
  sum_shared[tx] = (i < height && j < width && tx < pool_size) ? mat[mat_idx] : 0;
  __syncthreads();
    
  for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
    if (tx<s) {
      sum_shared[tx] += sum_shared[tx+s];
    }
    __syncthreads();
  }
    
  if (tx==0) {
    const unsigned int target_idx = i*total_width_pooled +
      pooled_offset + f*width_pooled + blockIdx.x;
    target[target_idx] = sum_shared[0];
  }
}

__global__ void sum_pool_gradient(
				  const {{ data_type }} *df_output,
				  {{ data_type }} *df_input,
				  const unsigned int input_offset,
				  const unsigned int height,
				  const unsigned int total_width,
				  const unsigned int width,
				  const unsigned int pooled_offset,
				  const unsigned int total_width_pooled,
				  const unsigned int width_pooled) {

  /* Gradient of sum-pooling operation
   */
    
  const unsigned int tx = threadIdx.x;
  const unsigned int bx = blockIdx.x;
  const unsigned int f = blockIdx.y;
  const unsigned int row = blockIdx.z;
  const unsigned int column = bx*blockDim.x+tx;
    
  const unsigned int pooled_idx = row*total_width_pooled +
    pooled_offset + f*width_pooled + bx;
  const {{ data_type }} df_output_element = df_output[pooled_idx];

  if (column < width) {
    df_input[row*total_width + input_offset +
	     f*width + column] = df_output_element;
  }
}

__global__ void fully_connected_layer(const nucleotide_t *input,
				      {{ data_type }} *target, 
				      const {{ data_type }} *filter,
				      const {{ data_type }} *bias,
				      const unsigned int input_offset,
				      const unsigned int width,
				      const unsigned int total_width,
				      const unsigned int height, 
				      const unsigned int total_target_width,
				      const unsigned int target_offset,
				      const unsigned int n_filters) {

  /* This is a simple variation of the convolution operation where the
     filter is the same size as the input. 
  */

  const unsigned int f = blockIdx.z;
  const unsigned int tx = threadIdx.x;
  const unsigned int dimy = blockDim.y;
  const unsigned int i = blockIdx.x*blockDim.x+threadIdx.x; // Row
  const unsigned int j = threadIdx.y; // Column
  const unsigned int filter_elements = STRIDE*width; // Actual number of elements in filter
  const nucleotide_t input_element = (i < height && j < width) ? 
    input[i*total_width+input_offset+j] : 0. ;
    
  extern __shared__ {{ data_type }} sdata[];
  {{ data_type }} *filter_shared = sdata; // size: filter_elements
  {{ data_type }} *output_shared = sdata + filter_elements; // size: blockDim.x * width

  // Load filter elements into shared memory
  const unsigned int tid = threadIdx.x*blockDim.y+threadIdx.y;
  if (tid < filter_elements)
    filter_shared[tid] = filter[f*filter_elements+tid];
  __syncthreads();

  const {{ data_type }} filter_A = filter_shared[j*STRIDE];
  const {{ data_type }} filter_C = filter_shared[j*STRIDE+1];
  const {{ data_type }} filter_G = filter_shared[j*STRIDE+2];
  const {{ data_type }} filter_T = filter_shared[j*STRIDE+3];

  // Compute output
  if (i < height && j < width) {
    {{ data_type }} Pvalue;
    if (CHECK_NT(input_element, DNA_A))
      Pvalue = filter_A;

    if (CHECK_NT(input_element, DNA_C))
      Pvalue = filter_C;

    if (CHECK_NT(input_element, DNA_G))
      Pvalue = filter_G;

    if (CHECK_NT(input_element, DNA_T))
      Pvalue = filter_T;

    if (CHECK_NT(input_element, DNA_R))
      Pvalue = .5 * (filter_A + filter_G);

    if (CHECK_NT(input_element, DNA_Y))
      Pvalue = .5 * (filter_C + filter_T);

    if (j == 0)
      Pvalue += bias[f];

    // Write output to shared memory
    output_shared[tx*dimy+j] = Pvalue;
  } else {
    output_shared[tx*dimy+j] = 0.;
  }
  __syncthreads();

  // Sum up all the filter elements
  for (unsigned int s=blockDim.y/2; s>0; s>>=1) {
    if (j < s) {
      output_shared[tx*dimy+j] += output_shared[tx*dimy+j+s];
    }
    __syncthreads();
  }

  // Write final output
  if (i < height and j == 0) {
    const unsigned int target_idx = i*total_target_width+target_offset+f;
    target[target_idx] = output_shared[tx*dimy];
  }
}

__global__ void fully_connected_layer_gradient(const nucleotide_t *input,
					       const {{ data_type }} *df_output,
					       {{ data_type }} *df_weights,
					       const unsigned int input_offset,
					       const unsigned int df_output_offset,
					       const unsigned int total_input_width,
					       const unsigned int total_df_output_width,
					       const unsigned int width,
					       const unsigned int height,
					       const unsigned int n_filters) {

  const unsigned int tx = threadIdx.x;
  const unsigned int i = blockDim.x*blockIdx.x+tx; // Row
  const unsigned int j = threadIdx.y; // Column
  const unsigned int f = threadIdx.z; // Filter
  const unsigned int input_idx = i*total_input_width+input_offset+j;
  
  const nucleotide_t input_element = input[input_idx];

  extern __shared__ {{ data_type }} sdata[];
  {{ data_type }} *df_output_shared = sdata; // size: blockDim.x
  {{ data_type }} *df_weights_shared = df_output_shared + blockDim.x; // size: blockDim.x * width * STRIDE

  // Load df_output into shared memory
  if (j == 0) {
    const unsigned int df_output_idx = i*total_df_output_width+df_output_offset+f;
    df_output_shared[tx] = df_output[df_output_idx];
  }
  __syncthreads();

  const {{ data_type }} df_output_element = df_output_shared[tx];
  
  // Compute gradient
  const unsigned int df_weights_shared_idx = tx*STRIDE*width+j*STRIDE;
  if (i < height && j < width) {
    // DNA_A
    df_weights_shared[df_weights_shared_idx] = CHECK_NT(input_element, DNA_A) ?
      df_output_element : 0;
    // DNA_C
    df_weights_shared[df_weights_shared_idx+1] = CHECK_NT(input_element, DNA_C) ?
      df_output_element : 0;
    // DNA_G
    df_weights_shared[df_weights_shared_idx+2] = CHECK_NT(input_element, DNA_G) ?
      df_output_element : 0;
    // DNA_T
    df_weights_shared[df_weights_shared_idx+3] = CHECK_NT(input_element, DNA_T) ?
      df_output_element : 0;

    if (CHECK_NT(input_element, DNA_R)) {
      df_weights_shared[df_weights_shared_idx] = .5 * df_output_element;
      df_weights_shared[df_weights_shared_idx+2] = .5 * df_output_element;
    }

    if (CHECK_NT(input_element, DNA_Y)) {
      df_weights_shared[df_weights_shared_idx+1] = .5 * df_output_element;
      df_weights_shared[df_weights_shared_idx+3] = .5 * df_output_element;
    }
  } else {
    df_weights_shared[df_weights_shared_idx] = 0.;
    df_weights_shared[df_weights_shared_idx+1] = 0.;
    df_weights_shared[df_weights_shared_idx+2] = 0.;
    df_weights_shared[df_weights_shared_idx+2] = 0.;
  }
  __syncthreads();

  // Stage 1 reduction
  unsigned int df_weights_shared_idx_next;
  for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
    if (tx < s) {
      df_weights_shared_idx_next = (tx+s)*STRIDE*width+j*STRIDE;
      df_weights_shared[df_weights_shared_idx] += df_weights_shared[df_weights_shared_idx_next];
      df_weights_shared[df_weights_shared_idx+1] += df_weights_shared[df_weights_shared_idx_next+1];
      df_weights_shared[df_weights_shared_idx+2] += df_weights_shared[df_weights_shared_idx_next+2];
      df_weights_shared[df_weights_shared_idx+3] += df_weights_shared[df_weights_shared_idx_next+3];
    }
    __syncthreads();
  }
  
  // Write output
  if (tx==0) {
    const unsigned int df_weights_idx = blockIdx.x*STRIDE*width+j*STRIDE;
    df_weights[df_weights_idx] = df_weights_shared[j*STRIDE];
    df_weights[df_weights_idx+1] = df_weights_shared[j*STRIDE+1];
    df_weights[df_weights_idx+2] = df_weights_shared[j*STRIDE+2];
    df_weights[df_weights_idx+3] = df_weights_shared[j*STRIDE+3];
  }
}
