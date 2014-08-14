# Copyright (C) 2013  Hannes Bretschneider

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

from . import eps
from .. import memory_pool
from .reductions import max_by_axis
from .matrix import add_vec_to_mat
from .reductions import matrix_sum_out_axis
from .elementwise import nan_to_zeros
from pycuda import cumath, gpuarray
import numpy as np

def logsumexp(mat):
    max_dim = max_by_axis(mat, 1)
    tmp = add_vec_to_mat(mat, max_dim, 0, substract=True)

    tmp = cumath.exp(tmp)
    
    tmp = matrix_sum_out_axis(tmp, 1)
    tmp = cumath.log(tmp)
    max_dim += tmp
    return max_dim

def softmax(mat):
    tmp = gpuarray.empty_like(mat)
    L = logsumexp(mat)
    tmp = add_vec_to_mat(mat, L, substract=True)
    tmp = cumath.exp(tmp)
    return tmp

def cross_entropy(x, y):
    loss = y * cumath.log(x + eps)
    nan_to_zeros(loss, loss)
    loss = -gpuarray.sum(loss)
    return float(loss.get())

def cross_entropy_logistic(x, y):
    loss = y * cumath.log(x + eps) + (1. - y) * cumath.log(1. - x + eps)
    loss = -gpuarray.sum(loss)
    return loss