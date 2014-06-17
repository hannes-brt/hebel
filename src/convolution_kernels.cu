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
#include <curand_kernel.h>
#include "convolution_kernels.h"

typedef {{ dtype }} data_t;
typedef {{ dtype }}4 vec_t;
typedef {{ dtype_idx }} idx_t;

extern "C"
{

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
  
    idx_t shared_idx, idx, row, col, output_idx, block_origin,
      block_origin_input, n_input_elements;
    nucleotide_t nt;
    data_t pvalue;
    vec_t ff;

    const idx_t filters_per_block = min(n_filters, (BX + 1) * BDX) -
      BX * BDX;
    const idx_t N = input_width * input_height;
    const idx_t halo_width = filter_width - 1;
    const idx_t output_width = input_width - halo_width;
    const idx_t n_filter_elements = filter_width * filters_per_block;
    const idx_t filter_idx = TX * filter_width; // First element of filter
    const idx_t f = TX + BX * BDX;

    // Setup shared memory
    extern __shared__ data_t sdata[];
    vec_t *filter_shared = (vec_t*) sdata; // Shared memory for filters
    data_t *bias_shared = (data_t*) (filter_shared + n_filter_elements); // Biases 
    nucleotide_t *input_seq_shared = (nucleotide_t*) (bias_shared + filters_per_block); // Input
  
    // Load filter elements into shared memory
    for (shared_idx = TX + BDX * TY;
	 shared_idx < n_filter_elements;
	 shared_idx += BDX * BDY) {
      idx = BX*BDX*filter_width+shared_idx;
      assert(idx < (n_filters*filter_width));
      filter_shared[shared_idx] = ((vec_t*) filters)[idx];
    }
			    
    // Load biases into shared memory
    if (TY == 0 & f < n_filters)
      bias_shared[TX] = bias[f];
    __syncthreads();

    // Outer loop
    for (output_idx = BY * BDY + TY;
	 output_idx < DIV_UP(output_width*input_height, BDY);
	 output_idx += BDY * GDY) {
    
      block_origin = output_idx - TY;
      row = ROW(output_idx, output_width);
      col = COLUMN(output_idx, output_width);
    
      block_origin_input = OUTPUT_TO_INPUT_IDX(block_origin, 
					       input_width, output_width);

      n_input_elements = OUTPUT_TO_INPUT_IDX(block_origin + BDY, 
					     input_width, output_width) -
	block_origin_input;

      // Load input into shared memory
      for (shared_idx = TX + BDX * TY;
	   shared_idx < n_input_elements + halo_width;
	   shared_idx += BDX * BDY) {
	idx = block_origin_input + shared_idx;
	input_seq_shared[shared_idx] = (idx < N) ? input[idx] : ' ';
      }
      __syncthreads();

      shared_idx = OUTPUT_TO_INPUT_IDX(output_idx, input_width, output_width) - block_origin_input;
      assert(shared_idx < (CEIL_DIV(BDY, output_width) * input_width + halo_width));

      // Perform convolution
      if (row < input_height & f < n_filters) {
	pvalue = bias_shared[TX];
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
	idx = n_filters * output_width * row + n_filters * col + f;
	assert(idx < (input_height * output_width * n_filters));
	output[idx] = pvalue;
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
    const idx_t filter_idx = ROW(TX + BX * BDX, filter_width);
    const idx_t filter_idx_shared = ROW(TX, filter_width);
    const idx_t filter_pos = COLUMN(TX + BX * BDX, filter_width);

    const idx_t filters_per_block = ROW(BDX, filter_width);

    // Setup shared memory
    __shared__ extern vec_t df_filters_shared[];
    data_t *df_output_shared = (data_t*) (df_filters_shared + BDX);
    nucleotide_t *input_shared =
      (nucleotide_t*) (df_output_shared + BDY * filter_width);

    // Zero shared memory
    if (TY == 0)
      df_filters_shared[TX] = (vec_t) {0., 0., 0., 0.};
    __syncthreads();

    // Outer loop
    for (output_idx = TY + BY * BDY;
	 output_idx < output_width * input_height;
	 output_idx += BDY * GDY) {

      block_origin_output = output_idx - TY;
      block_origin_input = 
	OUTPUT_TO_INPUT_IDX(block_origin_output, input_width, output_width);
      row = ROW(output_idx, output_width);
      col = COLUMN(output_idx, output_width);

      n_input_elements = 
	OUTPUT_TO_INPUT_IDX(block_origin_output + BDY, 
			    input_width, output_width) - 
	block_origin_input;

      // Load df_output into shared memory
      for (shared_idx = TX + TY * BDX;
	   shared_idx < BDY * filters_per_block;
	   shared_idx += BDX * BDY) {
	row_shared = ROW(block_origin_output + shared_idx / 
			 filters_per_block, output_width);
	col_shared = COLUMN(block_origin_output + shared_idx /
			    filters_per_block, output_width);
	f = BX * filters_per_block + shared_idx % filters_per_block;
	input_idx = row_shared * output_width * n_filters +
	  col_shared * n_filters + f;
	df_output_shared[shared_idx] = input_idx < N_output ?
	  df_output[input_idx] : 0.;
      }

      // Load input into shared memory
      for (shared_idx = TX + TY * BDX;
	   shared_idx < n_input_elements + halo_width;
	   shared_idx += BDX * BDY) {
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
    atomicAdd(&df_filters_shared[TX].x, df_filter_thread.x);
    atomicAdd(&df_filters_shared[TX].y, df_filter_thread.y);
    atomicAdd(&df_filters_shared[TX].z, df_filter_thread.z);
    atomicAdd(&df_filters_shared[TX].w, df_filter_thread.w);
    {% endif %}
    __syncthreads();

    // Write to global memory
    for (shared_idx = TX + BDX * TY;
	 shared_idx < filters_per_block * filter_width;
	 shared_idx += BDX * BDY) {
      f = BY * n_filters * filter_width +
	BX * filters_per_block * filter_width + 
	shared_idx;
      ((vec_t*) df_filters)[f] = df_filters_shared[shared_idx];
    }
  }

  __global__ void gradient_reduce(const data_t* df_filters,
				  data_t* df_filters_reduced,
				  const idx_t df_filters_size,
				  const idx_t reduction_size) {

    /* 
     *  Sum out the leading dimension of df_filters to complete the
     *  gradient computation.
     * 
     *  Arguments (shape):
     *    df_filters (reduction_size, df_filters_size) :
     *      Output of convolve_dna_sequence_gradient.
     *    df_filters_reduced (df_filters_size) :
     *      Empty array to store the result.
     *    df_filters_size :
     *      Total size of df_filters_reduced (product of all dimensions).
     *    reduction_size :
     *      Leading dimension of df_filters. This will be equal to
     *      gridDim.y from convolve_dna_sequence_gradient.
     * 
     *  Launch instructions:
     *    This kernel uses one thread for every element of df_filters_reduced.
     * 
     *    blockDim.x : May be any value and (blockDim.x * gridDim.x) <
     *    df_filters_size is allowed.
     * 
     */

    idx_t element_idx;
    vec_t p, tmp;

    for (element_idx = TX + BX * BDX;
	 element_idx < df_filters_size / 4;
	 element_idx += BDX * GDX) {
      p = (vec_t) {0., 0., 0., 0.};
      for (idx_t i = 0; i < reduction_size; i++) {
	tmp = ((vec_t*) df_filters)[i * df_filters_size / 4 + element_idx];
	p = (vec_t) {p.x + tmp.x, p.y + tmp.y,
		     p.z + tmp.z, p.w + tmp.w};
      }
      ((vec_t*) df_filters_reduced)[element_idx] = p;
    }
  }

  __global__ void max_pool(const data_t *input,
			   data_t *output,
			   idx_t *argmax,
			   const idx_t height,
			   const idx_t input_width,
			   const idx_t n_filters,
			   const idx_t pooling_size,
			   curandState_t *rand_state) {
  
    /*
     *  Perfom the max-pooling operation. The max-pooling operation
     *  implemented here is non-overlapping and restricted to sizes of
     *  the pooling region that are divisors of the width of the input.
     *  
     *  Arguments (shape) :
     *    input (height, input_width, n_filters) :
     *      Input to the max-pooling layer
     *    output (height, input_width / pooling_size, n_filters) :
     *      Pooled output
     *    argmax (height, input_width / pooling_size, n_filters) : The
     *      index of the maximum value in each pooling region. The argmax
     *      is taken with respect to the pooling region, not with respect
     *      to the entire array. When ties occur, they are broken
     *      randomly.
     *    height : 
     *      First dimension of input
     *    input_width :
     *      Second dimension of input
     *    n_filters :
     *      Third dimension of input
     *    pooling_size :
     *      Size of the pooling regions. Must evenly divide input_width.
     *    rand_state :
     *      Random number generator state. This is used to randomly brake
     *      ties.
     *
     *  Launch instructions :
     *    blockDim.x : The number of filters per block; requires
     *    (blockDim.x * gridDim.x) >= n_filters.
     *    blockDim.y : The number of positions per block. Can be any
     *    value and (blockDim.y * gridDim.y) < input_width is allowed.
     *  
     */

    idx_t output_idx, idx, input_origin, argmax_val, i;
    data_t comp_val, output_val;

    const idx_t output_width = input_width / pooling_size;
    const idx_t N_output = height * output_width;
    const idx_t N_input = height * input_width;
    const idx_t filter_idx = TX + BX * BDX;

    if (filter_idx < n_filters) {
      for (output_idx = TY + BY * BDY;
	   output_idx < N_output;
	   output_idx += BDY * GDY) {
	input_origin = output_idx * pooling_size;

	output_val = -FLT_MAX;

	for (i = 0; i < pooling_size; i++) {
	  idx = (input_origin + i) * n_filters + filter_idx;
	  assert(idx < (N_input * n_filters));
	  comp_val = input[idx];
	
	  if (comp_val > output_val || 
	      (comp_val == output_val && 
	       curand_uniform(rand_state))) {
	    output_val = comp_val;
	    argmax_val = i;
	  } 
	}
 
	idx = output_idx * n_filters + filter_idx;
	output[idx] = output_val;
	argmax[idx] = argmax_val;
      }
    }
  }

  __global__ void max_pool_gradient(const idx_t *argmax,
				    const data_t *backprop_gradient,
				    data_t *max_pool_gradient,
				    const idx_t height,
				    const idx_t width,
				    const idx_t width_pooled,
				    const idx_t n_filters) {
    
    /*
     *  The gradient for the max-pooling operation.
     *
     *  Arguments (shape) :
     *    argmax (height, width_pooled, n_filters) :
     *      The argmax returned from the forward pass.
     *    backprop_gradient (height, width_pooled, n_filters) :
     *      The backpropagated gradient from the layer above.
     *    max_pool_gradient (height, width, n_filters) :
     *      Array to store the computed max-pool gradient.
     *    height :
     *      First dimension of argmax.
     *    width :
     *      Second dimension of max_pool_gradient.
     *    width_pooled :
     *      Second dimension of argmax.
     *    n_filters :
     *      Third dimension of argmax.
     *
     *  Launch instructions :
     *    blockDim.x : The number of filters per block. It is 
     *      required that blockDim.x * gridDim.x >= n_filters.
     *    blockDim.y : The number of input positions (height * width)
     *      per block. Requires blockDim.y % pooling_size == 0, but
     *      blockDim.y * gridDim.y < height * width is allowed.
     *    shared : 
     *      blockDim.x * blockDim.y * (sizeof(unsigned int) + sizeof(float))
     *
     */
  
    idx_t input_idx, output_origin, shared_idx, idx, argmax_val;
    data_t gradient_val;

    const idx_t pooling_size = width / width_pooled;
    const idx_t filter_idx = TX + BX * BDX;

    // Setup shared memory
    extern __shared__ data_t sdata[];
    data_t *bp_grad_shared = sdata;
    idx_t *argmax_shared = (idx_t*) (bp_grad_shared + (BDX * BDY) / pooling_size);

    if (filter_idx < n_filters) {
      // Outer loop
      for (input_idx = TY + BY * BDY;
	   input_idx < height * width;
	   input_idx += (BDY * GDY)) {
	
	output_origin = (input_idx - TY) / pooling_size;

	// Load shared memory
	if (TY < (BDY / pooling_size)) {
	  shared_idx = BDX * TY + TX;
	  idx = n_filters * (output_origin + TY) + filter_idx;
	  if (idx < (height * width_pooled * n_filters)) {
	    bp_grad_shared[shared_idx] =
	      backprop_gradient[idx];
	    argmax_shared[shared_idx] =
	      argmax[idx];
	  }
	}
	__syncthreads();

	// Write output
	shared_idx = BDX * (TY / pooling_size) + TX;
	argmax_val = argmax_shared[shared_idx];
	gradient_val = ((TY % pooling_size) == argmax_val) ?
	  bp_grad_shared[shared_idx] : 0.;

	idx = n_filters * input_idx + filter_idx;
	max_pool_gradient[idx] = gradient_val;
	__syncthreads();
      }
    }
  }

  __global__ void sum_pool(const data_t *input,
			   data_t *output,
			   const idx_t height,
			   const idx_t input_width,
			   const idx_t n_filters,
			   const idx_t pooling_size) {
    idx_t output_idx, idx, input_origin, i;
    data_t output_val;

    const idx_t output_width = input_width / pooling_size;
    const idx_t N_output = height * output_width;
    const idx_t N_input = height * input_width;
    const idx_t filter_idx = TX + BX * BDX;

    if (filter_idx < n_filters) {
      for (output_idx = TY + BY * BDY;
	   output_idx < N_output;
	   output_idx += BDY * GDY) {
	input_origin = output_idx * pooling_size;

	output_val = 0;

	for (i = 0; i < pooling_size; i++) {
	  idx = (input_origin + i) * n_filters + filter_idx;
	  assert(idx < (N_input * n_filters));
	  output_val += input[idx];
	}
 
	idx = output_idx * n_filters + filter_idx;
	output[idx] = output_val;
      }
    }
  }

  __global__ void sum_pool_gradient() {}
}
