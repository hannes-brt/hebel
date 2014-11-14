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

#include <stdio.h>
#include <vector_types.h>
#include "convolution_kernels.h"

typedef {{ dtype }} data_t;

extern "C"
{
  __global__ void sequenceToFloatKernel(const unsigned n, const unsigned w, 
					const nucleotide_t* sequence, data_t* output)
  {
    const unsigned idx = TX + BX * BDX;
    const unsigned i = ROW(idx, w);
    const unsigned j = COLUMN(idx, w);
    
    if ((i < n) && (j < w)) {

      const nucleotide_t nt = sequence[idx];
      data_t a, c, g, t;

      if (CHECK_NT(nt, DNA_A)) {
	a = 1.;
	c = 0.;
	g = 0.;
	t = 0.;
      }
        
      else if (CHECK_NT(nt, DNA_C)) {
	a = 0.;
	c = 1.;
	g = 0.;
	t = 0.;
      }

      else if (CHECK_NT(nt, DNA_G)) {
	a = 0.;
	c = 0.;
	g = 1.;
	t = 0.;
      }

      else if (CHECK_NT(nt, DNA_T)) {
	a = 0.;
	c = 0.;
	g = 0.;
	t = 1.;
      }
        
      else if (CHECK_NT(nt, DNA_R)) {
	a =  .5;
	c = 0.;
	g =  .5;
	t = 0.;
      }
      
      else if (CHECK_NT(nt, DNA_Y)) {
	a = 0.;
	c =  .5;
	g = 0.;
	t =  .5;
      }
      
      else if (CHECK_NT(nt, DNA_N)) {
	a = .25;
	c = .25;
	g = .25;
	t = .25;
      }

      output[i * 4 * w + 0 * w + j] = a;
      output[i * 4 * w + 1 * w + j] = c;
      output[i * 4 * w + 2 * w + j] = g;
      output[i * 4 * w + 3 * w + j] = t;
    }
  }
}
