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

from pycuda import gpuarray
import numpy as np
from math import ceil

def show_filters(W, img_dims, columns=10, normalize=True, **kwargs):
    import matplotlib.pyplot as plt
    if isinstance(W, gpuarray.GPUArray): W = W.get()

    D, N = W.shape

    if normalize:
        W = W - W.min() #[np.newaxis,:]
        W = W / W.max() #[np.newaxis,:]

    rows = int(ceil(N / columns))
        
    fig = plt.figure(1, **kwargs)
    plt.subplots_adjust(left=0., right=.51, wspace=.1, hspace=.01)    

    filters = np.rollaxis(W.reshape(img_dims + (N,)), 2)
    filters = np.vstack([np.hstack(filters[i:i+columns]) for i in range(0, N, columns)])
    plt.axis('off')
    plt.imshow(filters, cmap=plt.cm.gray, interpolation='nearest', figure=fig)
