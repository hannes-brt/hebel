from .. import sampler
from . import ParameterfreeLayer
from pycuda import gpuarray

class NoiseLayer(ParameterfreeLayer):
    def __init__(self, n_in, n_filters=1):
        self.n_in = n_in
        self.n_filters = n_filters
        self.n_units = n_in * n_filters
        self.n_units_per_filter = n_in

    def feed_forward(self, input_data, prediction=False):
        noise = self.noise_func(input_data)
        return (input_data + noise,)

    def backprop(self, input_data, df_output, cache=None):
        return tuple(), df_output

    def noise_func(self, x):
        raise NotImplementedError

class GaussianNoiseLayer(NoiseLayer):
    def __init__(self, n_in, n_filters=1, stddev=.1, mean=0):
        self.stddev = stddev
        self.mean = mean
        super(GaussianNoiseLayer, self).__init__(n_in, n_filters)

    def noise_func(self, x):
        noise = gpuarray.empty_like(x)
        sampler.fill_normal(noise)
        noise *= self.stddev
        noise += self.mean
        return noise

class UniformNoiseLayer(NoiseLayer):
    def __init__(self, n_in, n_filters=1, scale=.1, bias=-.5):
        self.scale = scale
        self.bias = bias
        super(UniformNoiseLayer, self).__init__(n_in, n_filters)

    def noise_func(self, x):
        noise = gpuarray.empty_like(x)
        sampler.fill_uniform(noise)
        noise *= self.scale
        noise += self.bias
        return noise

