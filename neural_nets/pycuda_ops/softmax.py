from . import eps
from .reductions import max_by_axis
from .matrix import add_vec_to_mat
from .reductions import matrix_sum_out_axis
from .elementwise import nan_to_zeros_kernel
from pycuda import cumath, gpuarray
    
def logsumexp(mat):
    max_dim = max_by_axis(mat, 1)
    tmp = add_vec_to_mat(mat, -max_dim, 0)
    L = max_dim + cumath.log(matrix_sum_out_axis(cumath.exp(tmp), 1))
    return L

def softmax(mat):
    L = logsumexp(mat)
    return cumath.exp(add_vec_to_mat(mat, -L, inplace=True))

def cross_entropy(x, y):
    loss = y * cumath.log(x + eps)
    nan_to_zeros_kernel(loss, loss)
    loss = -gpuarray.sum(loss)
    return float(loss.get())

