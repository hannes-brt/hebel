import numpy as np
import cPickle
from itertools import izip
from pycuda import gpuarray
from pycuda.gpuarray import GPUArray
from math import sqrt
from scikits.cuda import linalg
from .. import sampler
from ..pycuda_ops import eps
from ..pycuda_ops.elementwise import sigmoid, df_sigmoid, \
     tanh, df_tanh, relu, df_relu, linear, df_linear, \
     sample_dropout_mask, apply_dropout_mask, sign
from ..pycuda_ops.matrix import add_vec_to_mat
from ..pycuda_ops.reductions import matrix_sum_out_axis


class HiddenLayer(object):
    n_parameters = 2
    W = None
    b = None

    def __init__(self, n_in, n_units,
                 activation_function='sigmoid',
                 dropout=False,
                 parameters=None,
                 weights_scale=None,
                 lr_multiplier=None,
                 l1_penalty_weight=0., l2_penalty_weight=0.):

        self._set_activation_fct(activation_function)

        if weights_scale is None:
            self._set_weights_scale(activation_function, n_in, n_units)
        else:
            self.weights_scale = weights_scale

        if parameters is not None:
            if isinstance(parameters, basestring):
                self.parameters = cPickle.loads(open(parameters))
            else:
                self.W, self.b = parameters
        else:
            self.W = self.weights_scale * \
                     sampler.gen_uniform((n_in, n_units),
                                         dtype=np.float32) \
              - .5 * self.weights_scale

            self.b = gpuarray.zeros((n_units,), dtype=np.float32)

        assert self.W.shape == (n_in, n_units)
        assert self.b.shape == (n_units,)

        self.n_in = n_in
        self.n_units = n_units

        self.lr_multiplier = lr_multiplier if lr_multiplier is not None else \
            2 * [1. / np.sqrt(self.n_in, dtype=np.float32)]

        self.l1_penalty_weight = l1_penalty_weight
        self.l2_penalty_weight = l2_penalty_weight

        self.dropout = dropout

    @property
    def parameters(self):
        return (self.W, self.b)

    @parameters.setter
    def parameters(self, value):
        self.W = value[0] if isinstance(value[0], GPUArray) else \
          gpuarray.to_gpu(value[0])
        self.b = value[1] if isinstance(value[0], GPUArray) else \
          gpuarray.to_gpu(value[1])

    def update_parameters(self, values, stream=None):
        assert len(values) == self.n_parameters

        for (param, (gparam, mult)) \
            in izip((self.W, self.b), values):
            param._axpbyz(1., gparam, mult, param,
                          stream=stream)

    @property
    def architecture(self):
        arch = {'class': self.__class__,
                'n_in': self.n_in,
                'n_units': self.n_units,
                'activation_function': self.activation_function
                if hasattr(self, 'activation_function') else None}
        return arch

    @staticmethod
    def _resolve_activation_fct(activation_function):
        if activation_function == 'sigmoid':
            f = sigmoid
            df = df_sigmoid
        elif activation_function == 'tanh':
            f = tanh
            df = df_tanh
        elif activation_function == 'relu':
            f = relu
            df = df_relu
        elif activation_function == 'linear':
            f = linear
            df = df_linear
        else:
            raise ValueError

        return f, df

    def _set_activation_fct(self, activation_function):
        self.activation_function = activation_function
        self.f, self.df = self._resolve_activation_fct(activation_function)

    def _set_weights_scale(self, activation_function, n_in, n_units):
        if activation_function in ('tanh', 'relu', 'linear'):
            self.weights_scale = sqrt(6. / (n_in + n_units))
        elif activation_function == 'sigmoid':
            self.weights_scale = 4 * sqrt(6. / (n_in + n_units))
        else:
            raise ValueError

    @property
    def l1_penalty(self):
        return float(self.l1_penalty_weight) * gpuarray.sum(abs(self.W)).get()

    @property
    def l2_penalty(self):
        return float(self.l2_penalty_weight) * .5 * \
            gpuarray.sum(self.W ** 2.).get()

    def feed_forward(self, input_data, prediction=False):
        """ Propagate forward through the hidden layer.
        Inputs:
        input_data -- input from the previous layer
        prediction -- (bool) whether predicting or training

        Outputs:
        lin_activations
        activations

        If self.dropout = True and prediction=False:
        Output:
        lin_activations
        activations
        dropout_mask: binary mask of dropped units

        """

        activations = linalg.dot(input_data, self.W)
        activations = add_vec_to_mat(activations, self.b, inplace=True)

        self.f(activations)

        if self.dropout and prediction:
            activations *= .5

        if self.dropout and not prediction:
            dropout_mask = sample_dropout_mask(activations)
            return activations, dropout_mask

        return (activations,)

    def backprop(self, input_data, df_output, cache=None):
        """ Backpropagate through the hidden layer

        Inputs:
        input_data
        df_output: the gradient wrt the output units
        cache (optional): cache object from the forward pass

        Output:
        df_W: gradient wrt the weights
        df_b: gradient wrt the bias
        df_input: gradient wrt the input

        """

        # Get cache if it wasn't provided
        if cache is None:
            cache = self.feed_forward(input_data,
                                      prediction=False)

        if len(cache) == 2:
            activations, dropout_mask = cache
        else:
            activations = cache[0]

        # Multiply the binary mask with the incoming gradients
        if self.dropout and dropout_mask is not None:
            apply_dropout_mask(df_output, dropout_mask)

        # Get gradient wrt activation function
        df_activations = self.df(activations)
        delta = df_activations * df_output

        # Gradient wrt weights
        df_W = linalg.dot(input_data, delta, transa='T')
        # Gradient wrt bias
        df_b = matrix_sum_out_axis(delta, 0)
        # Gradient wrt inputs
        df_input = linalg.dot(delta, self.W, transb='T')

        # L1 weight decay
        if self.l1_penalty_weight:
            df_W -= self.l1_penalty_weight * sign(self.W)

        # L2 weight decay
        if self.l2_penalty_weight:
            df_W -= self.l2_penalty_weight * self.W

        return (df_W, df_b), df_input
