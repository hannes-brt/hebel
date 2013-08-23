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

// The ceiling function
#define CEILING(x) (int)(x) + (1 - (int)((int)((x) + 1) - (x)))

typedef char nucleotide_t;

typedef enum {
  DNA_A = 'A',
  DNA_C = 'C',
  DNA_G = 'G',
  DNA_T = 'T',
  DNA_R = 'R',
  DNA_Y = 'Y',
  DNA_N = 'N'
} nucleotide;

// Check if nucleotide nt is letter l
#define CHECK_NT(nt, l) (nt == (nucleotide_t) l)

#define STRIDE 4
