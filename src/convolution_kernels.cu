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
#include <stdio.h>
#include <assert.h>
#include <vector_types.h>
#include "convolution_kernels.h"

typedef {{ dtype }} data_t;
typedef {{ dtype }}4 vec_t;
typedef {{ dtype_idx }} idx_t;

__global__ void convolve_dna_sequence(const nucleotide_t *input,
                                      data_t *output,
                                      const data_t *filters,
                                      const data_t *bias,
                                      const idx_t input_width,
                                      const idx_t input_height,
                                      const idx_t filter_width,
				      const idx_t n_filters) {

  /*
   *  Compute the convolution of a matrix of DNA sequences with a set of filters
   * 
   *  The convolution is performed without zero-padding on either side,
   *  so the input sequences must be explicitely padded prior to
   *  calling the kernel. Any character other than A,C,G,T,R,Y, or N
   *  can be used, but ' ' (space) is recommended.
   * 
   *  Arguments (shape):
   *    input (input_height, input_width) : 
   *      Array of the input sequence 
   *    output (input_height, input_width - filter_width + 1) :
   *      Empty array for the output
   *    filters (n_filters, filter_width x 4) :
   *      Array of convolution filters
   *    bias (n_filters) :
   *      Biases for convolution
   *    input_width : 
   *      Second dimension of input
   *    input_height :
   *      First dimension of input
   *    filter_width :
   *      Second dimension of filters
   *    n_filters : 
   *      First dimension of filters
   *  
   *  Launch instructions: 
   *    This kernel uses one thread for each output element.
   * 
   *    blockDim.x : Number of filters per block
   *      (blockDim.x * gridDim.x) >= n_filters is required.
   *    blockDim.y : Number of output elements per block; can be any
   *      size and (blockDim.y * gridDim.y) <
   *      (output_width*input_height) is allowed.
   *    shared : Size of shared memory in bytes
   *      DIV_UP(input_shared_size, 4) * sizeof(nucleotide_t) + // input_shared
   *      blockDim.x * filter_width * N_LETTERS * sizeof(data_t) + // filter_shared
   *      blockDim.x * sizeof(data_t) // bias_shared
   *
   */
  
  idx_t shared_idx, input_idx, row, col, output_idx, block_origin,
    block_origin_input, n_input_elements;
  nucleotide_t nt;
  data_t pvalue;
  vec_t ff;

  const idx_t filters_per_block = blockDim.x;
  const idx_t N = input_width * input_height;
  const idx_t halo_width = filter_width - 1;
  const idx_t output_width = input_width - halo_width;
  const idx_t n_filter_elements = filter_width * filters_per_block;
  const idx_t filter_idx = threadIdx.x * filter_width; // First element of filter
  const idx_t f = threadIdx.x + blockIdx.x * blockDim.x;

  // Setup shared memory
  extern __shared__ data_t sdata[];
  vec_t *filter_shared = (vec_t*) sdata; // Shared memory for filters
  data_t *bias_shared = (data_t*) (filter_shared + n_filter_elements); // Biases 
  nucleotide_t *input_seq_shared = (nucleotide_t*) (bias_shared + n_filters); // Input
  
  // Load filter elements into shared memory
  for (shared_idx = threadIdx.x + blockDim.x * threadIdx.y;
       shared_idx < n_filter_elements;
       shared_idx += blockDim.x * blockDim.y)
    filter_shared[shared_idx] =
      ((vec_t*) filters)[blockIdx.x*blockDim.x*filter_width+shared_idx];
			    
  // Load biases into shared memory
  if (threadIdx.y == 0)
    bias_shared[threadIdx.x] = bias[f];
  __syncthreads();

  // Outer loop
  for (output_idx = blockIdx.y * blockDim.y + threadIdx.y;
       output_idx < DIV_UP(output_width*input_height, blockDim.y);
       output_idx += blockDim.y * gridDim.y) {
    
    block_origin = output_idx - threadIdx.y;
    row = ROW(output_idx, output_width);
    col = COLUMN(output_idx, output_width);
    
    block_origin_input = OUTPUT_TO_INPUT_IDX(block_origin, 
					     input_width, output_width);

    n_input_elements = OUTPUT_TO_INPUT_IDX(block_origin + blockDim.y, 
					   input_width, output_width) -
                       block_origin_input;

    // Load input into shared memory
    for (shared_idx = threadIdx.x + blockDim.x * threadIdx.y;
	 shared_idx < n_input_elements + halo_width;
	 shared_idx += blockDim.x * blockDim.y) {
      input_idx = block_origin_input + shared_idx;
      input_seq_shared[shared_idx] = (input_idx < N) ? input[input_idx] : ' ';
    }
    __syncthreads();

    shared_idx = OUTPUT_TO_INPUT_IDX(output_idx, input_width, output_width) - block_origin_input;

    // Perform convolution
    if (row < input_height & f < n_filters) {
      pvalue = bias_shared[threadIdx.x];
      for (idx_t k=0; k < filter_width; k++) {
	nt = input_seq_shared[shared_idx + k];
	ff = filter_shared[filter_idx+k];
      
	if (CHECK_NT(nt, DNA_A))
	  pvalue += ff.x;
        
	if (CHECK_NT(nt, DNA_C))
	  pvalue += ff.y;

	if (CHECK_NT(nt, DNA_G))
	  pvalue += ff.z;

	if (CHECK_NT(nt, DNA_T))
	  pvalue += ff.w;
        
	if (CHECK_NT(nt, DNA_R)) {
	  pvalue += .5 * ff.x;
	  pvalue += .5 * ff.z;
	}
      
	if (CHECK_NT(nt, DNA_Y)) {
	  pvalue += .5 * ff.y;
	  pvalue += .5 * ff.w;
	}
      
	if (CHECK_NT(nt, DNA_N)) {
	  pvalue += .25 * ff.x;
	  pvalue += .25 * ff.y;
	  pvalue += .25 * ff.z;
	  pvalue += .25 * ff.w;
	}
      }
    
      // Write output
      output[n_filters * output_width * row + n_filters * col + f] = pvalue;
    }
    __syncthreads();
  }
}

__global__ void convolve_dna_sequence_gradient(const nucleotide_t *input,
					       const data_t *df_output,
					       data_t *df_filters,
					       const idx_t input_width,
					       const idx_t input_height,
					       const idx_t filter_width,
					       const idx_t n_filters) {

  /* 
   *  Compute the gradients of the convolve_dna_sequence function with
   *  respect to the filters.
   * 
   *  Arguments (shape):
   *    input (input_height, input_width) :
   *      The DNA sequence that was convolved in the forward pass.
   *    df_output (input_height, input_width - filter_width + 1) :
   *      Gradient that is backpropagated from next layer.
   *    df_filters (gridDim.y, n_filters, filter_width, 4) :
   *      Empty array to store the computed gradients. If gridDim.y > 1, 
   *      this must be reduced along the leading dimension after running 
   *      this kernel.
   *    input_width :
   *      Second dimension of input.
   *    input_height :
   *      First dimension of input.
   *    filter_width :
   *      Third dimension of df_filters.
   *    n_filters :
   *      Second dimension of n_filters.
   * 
   *  Launch instructions: 
   *    This kernel uses one thread for each * element of df_filters
   *    and df_output.
   *
   *    blockDim.x : Must be a multiple of filter_width and
   *      (blockDim.x * gridDim.x) >= (n_filters * filter_width).
   *    blockDim.y : Specifies how many elements of df_output are
   *      processed per block. Can be any value and
   *      (blockDim.y * gridDim.y) < 
   *      (input_height * (input_width - filter_width + 1))
   *      is allowed.
   *    shared : Shared memory requirements in bytes are
   *      blockDim.x * filter_width * 4 * sizeof(data_t) + // df_filters_shared
   *      blockDim.y * filter_width * sizeof(data_t) + // df_output_shared
   *      (n_input_elements + halo_width) * sizeof(nucleotide_t) // input_shared
   *
   */

  idx_t output_idx, input_idx, n_input_elements, 
    shared_idx, block_origin_output, block_origin_input, 
    row, col, row_shared, col_shared,f;
  vec_t df_filter_thread = {0., 0., 0., 0.}; // Accumulate gradients in here
  nucleotide_t nt;
  data_t df;
  
  const idx_t halo_width = filter_width - 1;
  const idx_t output_width = input_width - halo_width;

  const idx_t N_input = input_width * input_height;
  const idx_t N_output = output_width * input_height * n_filters;
  const idx_t filter_idx = ROW(threadIdx.x + blockIdx.x * blockDim.x, filter_width);
  const idx_t filter_idx_shared = ROW(threadIdx.x, filter_width);
  const idx_t filter_pos = COLUMN(threadIdx.x + blockIdx.x * blockDim.x, filter_width);

  const idx_t filters_per_block = ROW(blockDim.x, filter_width);

  // Setup shared memory
  __shared__ extern vec_t df_filters_shared[];
  data_t *df_output_shared = (data_t*) (df_filters_shared + blockDim.x);
  nucleotide_t *input_shared =
    (nucleotide_t*) (df_output_shared + blockDim.y * filter_width);

  // Zero shared memory
  if (threadIdx.y == 0)
    df_filters_shared[threadIdx.x] = (vec_t) {0., 0., 0., 0.};
  __syncthreads();

  // Outer loop
  for (output_idx = threadIdx.y + blockIdx.y * blockDim.y;
       output_idx < output_width * input_height;
       output_idx += blockDim.y * gridDim.y) {

    block_origin_output = output_idx - threadIdx.y;
    block_origin_input = 
      OUTPUT_TO_INPUT_IDX(block_origin_output, input_width, output_width);
    row = ROW(output_idx, output_width);
    col = COLUMN(output_idx, output_width);

    n_input_elements = 
      OUTPUT_TO_INPUT_IDX(block_origin_output + blockDim.y, 
			  input_width, output_width) - 
      block_origin_input;

    // Load df_output into shared memory
    for (shared_idx = threadIdx.x + threadIdx.y * blockDim.x;
    	 shared_idx < blockDim.y * filters_per_block;
    	 shared_idx += blockDim.x * blockDim.y) {
      row_shared = ROW(block_origin_output + shared_idx / 
		       filters_per_block, output_width);
      col_shared = COLUMN(block_origin_output + shared_idx /
			  filters_per_block, output_width);
      f = blockIdx.x * filters_per_block + shared_idx % filters_per_block;
      input_idx = row_shared * output_width * n_filters +
	col_shared * n_filters + f;
      df_output_shared[shared_idx] = input_idx < N_output ?
    	df_output[input_idx] : 0.;
    }

    // Load input into shared memory
    for (shared_idx = threadIdx.x + threadIdx.y * blockDim.x;
	 shared_idx < n_input_elements + halo_width;
	 shared_idx += blockDim.x * blockDim.y) {
      input_idx = block_origin_input + shared_idx;
      input_shared[shared_idx] = input_idx < N_input ?
	input[block_origin_input + shared_idx] : ' ';
    }
    __syncthreads();
    
    // Read df_output element
    input_idx = row * output_width * filters_per_block + 
      col * filters_per_block;
    shared_idx = input_idx - 
      block_origin_output * filters_per_block + 
      filter_idx_shared;
    df = df_output_shared[shared_idx];
    
    // Compute gradient
    if (filter_idx < n_filters & filter_pos < filter_width) {
      shared_idx = 
	OUTPUT_TO_INPUT_IDX(output_idx, input_width, output_width) 
	- block_origin_input + filter_pos;

      nt = input_shared[shared_idx];

      if (CHECK_NT(nt, DNA_A))
	df_filter_thread.x += df;

      if (CHECK_NT(nt, DNA_C))
	df_filter_thread.y += df;
    
      if (CHECK_NT(nt, DNA_G))
	df_filter_thread.z += df;

      if (CHECK_NT(nt, DNA_T))
	df_filter_thread.w += df;

      if (CHECK_NT(nt, DNA_R)) {
	df_filter_thread.x += .5 * df;
	df_filter_thread.z += .5 * df;
      }

      if (CHECK_NT(nt, DNA_Y)) {
	df_filter_thread.y += .5 * df;
	df_filter_thread.w += .5 * df;
      }

      if (CHECK_NT(nt, DNA_N)) {
	df_filter_thread.x += .25 * df;
	df_filter_thread.y += .25 * df;
	df_filter_thread.z += .25 * df;
	df_filter_thread.w += .25 * df;
      }
    }
    __syncthreads();
  }
  
  // Add to shared memory
  {% if dtype == 'float' %}
  atomicAdd(&df_filters_shared[threadIdx.x].x, df_filter_thread.x);
  atomicAdd(&df_filters_shared[threadIdx.x].y, df_filter_thread.y);
  atomicAdd(&df_filters_shared[threadIdx.x].z, df_filter_thread.z);
  atomicAdd(&df_filters_shared[threadIdx.x].w, df_filter_thread.w);
  {% endif %}
  __syncthreads();

  // Write to global memory
  for (shared_idx = threadIdx.x + blockDim.x * threadIdx.y;
       shared_idx < filters_per_block * filter_width;
       shared_idx += blockDim.x * blockDim.y) {
    f = blockIdx.y * n_filters * filter_width +
      blockIdx.x * filters_per_block * filter_width + 
      shared_idx;
    ((vec_t*) df_filters)[f] = df_filters_shared[shared_idx];
  }
}

__global__ void gradient_reduce(const data_t* df_filters,
				data_t* df_filters_reduced,
				const idx_t df_filters_size,
				const idx_t reduction_size) {

  /*
   * Sum out the leading dimension of df_filters to complete the
   * gradient computation.
   *
   * Arguments (shape):
   *   df_filters (reduction_size, df_filters_size) :
   *     Output of convolve_dna_sequence_gradient.
   *   df_filters_reduced (df_filters_size) :
   *     Empty array to store the result.
   *   df_filters_size :
   *     Total size of df_filters_reduced (product of all dimensions).
   *   reduction_size :
   *     Leading dimension of df_filters. This will be equal to
   *     gridDim.y from convolve_dna_sequence_gradient.
   *
   * Launch instructions:
   *   This kernel uses one thread for every element of df_filters_reduced.
   *
   *   blockDim.x : May be any value and (blockDim.x * gridDim.x) <
   *   df_filters_size is allowed.
   *
   */

  idx_t element_idx;
  vec_t p, tmp;

  for (element_idx = threadIdx.x + blockIdx.x * blockDim.x;
       element_idx < df_filters_size / 4;
       element_idx += blockDim.x * gridDim.x) {
    p = (vec_t) {0., 0., 0., 0.};
    for (idx_t i = 0; i < reduction_size; i++) {
      tmp = ((vec_t*) df_filters)[i * df_filters_size / 4 + element_idx];
      p = (vec_t) {p.x + tmp.x, p.y + tmp.y,
		   p.z + tmp.z, p.w + tmp.w};
    }
    ((vec_t*) df_filters_reduced)[element_idx] = p;
  }
}

__global__ void max_pool() {}

__global__ void max_pool_gradient() {}

__global__ void sum_pool() {}

__global__ void sum_pool_gradient() {}
