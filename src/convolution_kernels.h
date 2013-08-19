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

// All four bit binary patterns
#define B0000 0
#define B0001 1
#define B0010 2
#define B0011 3
#define B0100 4
#define B0101 5
#define B0110 6
#define B0111 7
#define B1000 8
#define B1001 9
#define B1010 10
#define B1011 11
#define B1100 12
#define B1101 13
#define B1110 14
#define B1111 15

typedef char nucleotide_t;

typedef enum {
  DNA_A = B1000,
  DNA_C = B0100,
  DNA_G = B0010,
  DNA_T = B0001,
  DNA_R = B1010,
  DNA_Y = B0101,
  DNA_N = B0000
} nucleotide;

// Check if nucleotide nt is letter l
#define CHECK_NT(nt, l) (nt & (nucleotide_t) l)
