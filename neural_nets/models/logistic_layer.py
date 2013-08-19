import numpy as np
import cPickle
from pycuda import gpuarray
from pycuda import cumath
from math import sqrt
from scikits.cuda import linalg
from .. import sampler
from .top_layer import TopLayer
from ..pycuda_ops import eps
from ..pycuda_ops.elementwise import sign, nan_to_zeros
from ..pycuda_ops.reductions import matrix_sum_out_axis
from ..pycuda_ops.matrix import add_vec_to_mat
from ..pycuda_ops.softmax import softmax, cross_entropy


class LogisticLayer(TopLayer):
    """ A logistic classification layer, using
    cross-entropy and softmax activation.

    """

    act_f = softmax
    loss_f = cross_entropy
    n_parameters = 2

    def __init__(self, n_in, n_out,
                 parameters=None,
                 weights_scale=None,
                 l1_penalty_weight=0., l2_penalty_weight=0.,
                 lr_multiplier=None,
                 test_error_fct='class_error'):
        """ Inputs:
        n_in: number of input units
        n_out: number of output units (classes)
        loss_function: currently only works with cross_entropy

        """

        # Initialize weight using Bengio's rule
        self.weights_scale = 4 * sqrt(6. / (n_in + n_out)) \
                             if weights_scale is None \
                                else weights_scale

        if parameters is not None:
            if isinstance(parameters, basestring):
                self.parameters = cPickle.loads(open(parameters))
            else:
                self.W, self.b = parameters
        else:
            self.W = self.weights_scale * \
                     sampler.gen_uniform((n_in, n_out), dtype=np.float32) \
                     - .5 * self.weights_scale

            self.b = gpuarray.zeros((n_out,), dtype=np.float32)

        self.n_in = n_in
        self.n_out = n_out

        self.test_error_fct = test_error_fct

        self.l1_penalty_weight = l1_penalty_weight
        self.l2_penalty_weight = l2_penalty_weight

        self.lr_multiplier = 2 * [1. / np.sqrt(n_in, dtype=np.float32)] \
          if lr_multiplier is None else lr_multiplier

    @property
    def architecture(self):
        return {'class': self.__class__,
                'n_in': self.n_in,
                'n_out': self.n_out}

    def feed_forward(self, input_data, prediction=False):
        """ Propagate forward through the layer

        Inputs:
        input_data
        return_cache: (bool) whether to return the cache object
        prediction: (bool) whether to half the weights when
            the preceding layer uses dropout

        Outputs:
        activations

        """
        activations = linalg.dot(input_data, self.W)
        activations = add_vec_to_mat(activations, self.b, inplace=True)
        activations = softmax(activations)

        return activations

    def backprop(self, input_data, targets,
                 cache=None):
        """ Backpropagate through the logistic layer

        Inputs:
        input_data
        targets
        get_df_input: (bool) whether to compute and return the
            gradient wrt the inputs
        return_cache: (bool) whether to return the cache
        cache: cache object from forward pass

        """

        if cache is not None:
            activations = cache
        else:
            activations = self.feed_forward(input_data, prediction=False)

        delta = activations - targets
        nan_to_zeros(delta, delta)

        # Gradient wrt weights
        df_W = linalg.dot(input_data, delta, transa='T')
        # Gradient wrt bias
        df_b = matrix_sum_out_axis(delta, 0)

        # Gradient wrt input
        df_input = linalg.dot(delta, self.W, transb='T')

        # L1 penalty
        if self.l1_penalty_weight:
            df_W -= self.l1_penalty_weight * sign(self.W)

        # L2 penalty
        if self.l2_penalty_weight:
            df_W -= self.l2_penalty_weight * self.W

        return (df_W, df_b), df_input

    def test_error(self, input_data, targets, average=True,
                   cache=None, prediction=True):
        if self.test_error_fct == 'class_error':
            test_error = self.class_error
        elif self.test_error_fct == 'kl_error':
            test_error = self.kl_error
        elif self.test_error_fct == 'cross_entropy_error':
            test_error = self.cross_entropy_error
        else:
            raise ValueError('unknown test error function "%s"'
                             % self.test_error_fct)

        return test_error(input_data, targets, average,
                          cache, prediction)

    def cross_entropy_error(self, input_data, targets, average=True,
                            cache=None, prediction=False):
        """ Return the cross entropy error
        """

        if cache is not None:
            activations = cache
        else:
            activations = \
              self.feed_forward(input_data, prediction=prediction)

        loss = cross_entropy(activations, targets)

        if average: loss = loss.mean()
        return loss

    def class_error(self, input_data, targets, average=True,
                    cache=None, prediction=False):
        """ Return the classification error rate
        """

        if cache is not None:
            activations = cache
        else:
            activations = \
              self.feed_forward(input_data, prediction=prediction)

        targets = targets.get().argmax(1)
        class_error = np.sum(activations.get().argmax(1) != targets)

        if average: class_error = class_error.mean()
        return class_error

    def kl_error(self, input_data, targets, average=True,
                 cache=None, prediction=True):
        """ The KL divergence error
        """

        if cache is not None:
            activations = cache
        else:
            activations = \
              self.feed_forward(input_data, prediction=prediction)

        targets_non_nan = gpuarray.empty_like(targets)
        nan_to_zeros(targets, targets_non_nan)
        kl_error = gpuarray.sum(targets_non_nan *
                                (cumath.log(targets_non_nan + eps) -
                                 cumath.log(activations + eps)))
        if average:
            kl_error /= targets.shape[0]
        return float(kl_error.get())
