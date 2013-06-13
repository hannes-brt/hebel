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
