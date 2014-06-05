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

typedef {{ dtype }} float_type;
typedef {{ dtype_idx }} idx_type;

__global__ void convolve_dna_sequence(const nucleotide_t *input_sequence,
                                      float_type *target,
                                      const float_type *filter,
                                      const float_type *bias,
                                      const idx_type input_width,
                                      const idx_type input_height,
                                      const idx_type filter_width,
				      const idx_type n_filters) {
  
  idx_type shared_idx, input_idx, target_idx, row, col, output_idx, block_origin,
    block_origin_input, n_input_elements;
  nucleotide_t nt;
  float_type pvalue;

  const idx_type filters_per_block = blockDim.x;
  const idx_type N = input_width * input_height;
  const idx_type halo_width = filter_width - 1;
  const idx_type output_width = input_width - halo_width;
  const idx_type n_filter_elements = N_LETTERS * filter_width * filters_per_block;
  const idx_type filter_idx = threadIdx.x * filter_width * N_LETTERS; // First element of filter
  const idx_type f = threadIdx.x + blockIdx.x * blockDim.x;

  // Setup shared memory
  extern __shared__ float_type sdata[];
  const idx_type input_shared_size = CEIL_DIV(blockDim.y, output_width) * input_width + halo_width;
  nucleotide_t *input_seq_shared = (nucleotide_t*) sdata; // Shared memory for input sequence
  // Shared memory for filters
  float_type *filter_shared = (float_type*) (input_seq_shared + DIV_UP(input_shared_size, 4)); 
  float_type *bias_shared = filter_shared + n_filter_elements;
  
  // Load filter elements into shared memory
  for (shared_idx = threadIdx.x + blockDim.x * threadIdx.y;
       shared_idx < n_filter_elements;
       shared_idx += blockDim.x * blockDim.y)
    filter_shared[shared_idx] = 
      filter[blockIdx.x*blockDim.x*N_LETTERS*filter_width+shared_idx];
			    
  // Load biases into shared memory
  if (threadIdx.y == 0)
    bias_shared[threadIdx.x] = bias[f];

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
      input_seq_shared[shared_idx] = (input_idx < N) ? input_sequence[input_idx] : ' ';
    }
    __syncthreads();

    shared_idx = OUTPUT_TO_INPUT_IDX(output_idx, input_width, output_width) - block_origin_input;

    // Perform convolution
    if (row < input_height & f < n_filters) {
      pvalue = bias_shared[threadIdx.x];
      for (idx_type k=0; k < filter_width; k++) {
	nt = input_seq_shared[shared_idx + k];
      
	if (CHECK_NT(nt, DNA_A))
	  pvalue += filter_shared[filter_idx+N_LETTERS*k];
        
	if (CHECK_NT(nt, DNA_C))
	  pvalue += filter_shared[filter_idx+N_LETTERS*k+1];

	if (CHECK_NT(nt, DNA_G))
	  pvalue += filter_shared[filter_idx+N_LETTERS*k+2];

	if (CHECK_NT(nt, DNA_T))
	  pvalue += filter_shared[filter_idx+N_LETTERS*k+3];
        
	if (CHECK_NT(nt, DNA_R)) {
	  pvalue += .5 * filter_shared[filter_idx+N_LETTERS*k];
	  pvalue += .5 * filter_shared[filter_idx+N_LETTERS*k+2];
	}
      
	if (CHECK_NT(nt, DNA_Y)) {
	  pvalue += .5 * filter_shared[filter_idx+N_LETTERS*k+1];
	  pvalue += .5 * filter_shared[filter_idx+N_LETTERS*k+3];
	}
      
	if (CHECK_NT(nt, DNA_N)) {
	  pvalue += .25 * filter_shared[filter_idx+N_LETTERS*k];
	  pvalue += .25 * filter_shared[filter_idx+N_LETTERS*k+1];
	  pvalue += .25 * filter_shared[filter_idx+N_LETTERS*k+2];
	  pvalue += .25 * filter_shared[filter_idx+N_LETTERS*k+3];
	}
      }
    
      // Write output
      target_idx = n_filters * output_width * row + n_filters * col + f;
      target[target_idx] = pvalue;
    }
    __syncthreads();
  }
}

__global__ void convolve_dna_sequence_gradient() {}

__global__ void gradient_reduce() {}

__global__ void max_pool() {}

__global__ void max_pool_gradient() {}

__global__ void sum_pool() {}

__global__ void sum_pool_gradient() {}
