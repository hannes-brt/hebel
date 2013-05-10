import numpy as np
from pycuda import gpuarray
from pycuda.curandom import rand as curand
from pycuda import cumath
from math import sqrt
from scikits.cuda import linalg
from .pycuda_ops import sigmoid_kernel, df_sigmoid, \
      tanh_kernel, df_tanh, relu_kernel, df_relu, \
      add_vec_to_mat, softmax, cross_entropy, matrix_sum_out_axis

class HiddenLayer(object):
    def __init__(self, n_in, n_units, dropout=False,
                 W = None, b = None,
                 l1_penalty_weight=0., l2_penalty_weight=0.):

        if W is None:
            self.W = self.weights_scale * curand((n_in, n_units), dtype=np.float32) \
              - .5 * self.weights_scale
        else:
            self.W = W
        assert self.W.shape == (n_in, n_units)

        if b is None:
            self.b = gpuarray.zeros((n_units,), dtype=np.float32)
        else:
            self.b
        assert self.b.shape == (n_units,)
            
        self.n_in = n_in
        self.n_units = n_units

        self.l1_penalty_weight = l1_penalty_weight
        self.l2_penalty_weight = l2_penalty_weight

    @property
    def l1_penalty(self):
        return 0.

    @property
    def l2_penalty(self):
        return 0.

    def feed_forward(self, input, dropout_predict=False):
        activations = linalg.dot(input, self.W)
        activations = add_vec_to_mat(activations, self.b, inplace=True)
        
        # activations = gpuarray.empty_like(lin_activations)        
        self.f(activations)
        
        return activations

    def backprop(self, input, df_output, cache=None):
        """ Backpropagate through the hidden layer

        Inputs:
        input
        df_output: the gradient wrt the output units
        cache (optional): cache object from the forward pass

        Output:
        df_W: gradient wrt the weights
        df_b: gradient wrt the bias        
        df_input: gradient wrt the input

        """

        # Get cache if it wasn't provided
        if cache is None:
            cache = self.feed_forward(input)

        activations = cache

        # Get gradient wrt activation function
        df_activations = self.df(activations)
        delta = df_activations * df_output

        df_W = linalg.dot(input, delta, transa='T')     # Gradient wrt weights
        df_b = matrix_sum_out_axis(delta, 0)  # Gradient wrt bias
        df_input = linalg.dot(delta, self.W, transb='T')   # Gradient wrt inputs

        return df_W, df_b, df_input

class TanhHiddenLayer(HiddenLayer):
    def __init__(self, n_in, n_units,
                 dropout=False, W=None, b=None,
                 l1_penalty_weight=0., l2_penalty_weight=0.):
        self.f = tanh_kernel
        self.df = df_tanh

        self.weights_scale = sqrt(6. / (n_in + n_units))

        super(TanhHiddenLayer, self).__init__(n_in, n_units, dropout, W, b,
                                              l1_penalty_weight, l2_penalty_weight)


class SigmoidHiddenLayer(HiddenLayer):
    def __init__(self, n_in, n_units,
                 dropout=False, W=None, b=None,
                 l1_penalty_weight=0., l2_penalty_weight=0.):
        self.f = sigmoid_kernel
        self.df = df_sigmoid

        self.weights_scale = 4 * sqrt(6. / (n_in + n_units))

        super(SigmoidHiddenLayer, self).__init__(n_in, n_units, dropout, W, b,
                                                 l1_penalty_weight, l2_penalty_weight)
        

class ReluHiddenLayer(HiddenLayer):
    def __init__(self, n_in, n_units,
                 dropout=False, W=None, b=None,
                 l1_penalty_weight=0., l2_penalty_weight=0.):
        self.f = relu_kernel
        self.df = df_relu

        self.weights_scale = sqrt(6. / (n_in + n_units))

        super(ReluHiddenLayer, self).__init__(n_in, n_units, dropout, W, b,
                                              l1_penalty_weight, l2_penalty_weight)


class TopLayer(object):
    n_tasks = 1
    
class LogisticLayer(TopLayer):
    """ A logistic classification layer, using
    cross-entropy and softmax activation.

    """

    act_f = softmax

    loss_f = cross_entropy

    def __init__(self, n_in, n_out, 
                 l1_penalty_weight=0., l2_penalty_weight=0.,
                 test_error_fct='class_error'):
        """ Inputs:
        n_in: number of input units
        n_out: number of output units (classes)
        loss_function: currently only works with cross_entropy

        """

        # Initialize weight using Bengio's rule
        weights_scale = 4 * sqrt(6. / (n_in + n_out))
        self.W = weights_scale * curand((n_in, n_out), dtype=np.float32) - .5 * weights_scale
        self.b = gpuarray.zeros((n_out,), np.float32)
        self.n_in = n_in
        self.n_out = n_out

        self.test_error_fct = test_error_fct

        self.l1_penalty_weight = l1_penalty_weight
        self.l2_penalty_weight = l2_penalty_weight

    @property
    def l1_penalty(self):
        # return self.l1_penalty_weight * self.W.abs().sum()
        return 0.

    @property
    def l2_penalty(self):
        # return self.l2_penalty_weight * 0.5 * (self.W ** 2.).sum()
        return 0.

    def feed_forward(self, input, targets, return_cache=False, 
                     dropout_predict=False):
        """ Propagate forward through the layer

        Inputs:
        input
        target: classification targets (may be soft targets/class 
            probabilities, or integer vector)
        return_cache: (bool) whether to return the cache object
        dropout_predict: (bool) whether to half the weights when 
            the preceding layer uses dropout

        Outpus:
        loss: value of the loss function
        cache: (only when return_cache == True)

        """

        # Expand targets if necessary
        # targets_soft = expand_int_targets(targets, self.n_out)
        targets_soft = targets
        
        # if dropout_predict:
        #     # Half the hidden weights
        #     lin_activations = gpu.dot(input, .5 * self.W) + self.b
        # else:
        #     lin_activations = gpu.dot(input, self.W) + self.b

        activations = linalg.dot(input, self.W)
        activations = add_vec_to_mat(activations, self.b, inplace=True)

        activations = softmax(activations)

        loss = cross_entropy(activations, targets_soft)
        if not return_cache:
            return loss
        else:
            return loss, (targets_soft, activations)

    def predict(self, input):
        # if dropout_predict:
        #     # Half the hidden weights
        #     lin_activations = gpu.dot(input, .5 * self.W) + .5 * self.b
        # else:
        #     lin_activations = gpu.dot(input, self.W) + self.b

        activations = linalg.dot(input, self.W)
        activations = add_vec_to_mat(activations, self.b, inplace=True)

        activations = self.act_f(activations)

        return activations

    def backprop(self, input, targets, get_df_input=True, 
                 return_cache=False, cache=None):
        """ Backpropagate through the logistic layer

        Inputs:
        input
        targets
        get_df_input: (bool) whether to compute and return the 
            gradient wrt the inputs
        return_cache: (bool) whether to return the cache
        cache: cache object from forward pass

        """
        
        if cache is not None:
            targets_soft, activations = cache
        else:
            _, (targets_soft, activations) = \
              self.feed_forward(input, targets, return_cache=True,
                                dropout_predict=False)

        delta = activations - targets_soft
        
        df_W = linalg.dot(input, delta, transa='T')    # Gradient wrt weights
        df_b = matrix_sum_out_axis(delta, 0)               # Gradient wrt bias

        if get_df_input:
            df_input = linalg.dot(delta, self.W, transb='T')   # Gradient wrt input
            output = (df_W, df_b, df_input)
        else:
            output = (df_W, df_b)

        # # L1 penalty
        # if self.l1_penalty_weight:
        #     df_W += self.l1_penalty_weight * self.W.sign()

        # # L2 penalty
        # if self.l2_penalty_weight:
        #     df_W += self.l2_penalty_weight * self.W

        if return_cache:
            output += (targets_soft, activations)

        return output

    def test_error(self, input, targets, average=True,
                   cache=None, dropout_predict=False):
        if self.test_error_fct == 'class_error':
            test_error = self.class_error
        elif self.test_error_fct == 'kl_error':
            test_error = self.kl_error
        else:
            raise ValueError('unknown test error function "%s"' 
                             % self.test_error_fct)

        return test_error(input, targets, average,
                          cache, dropout_predict)
    
    def class_error(self, input, targets, average=True, 
                    cache=None, dropout_predict=False):
        """ Return the classification error rate

        """
        
        if cache is not None:
            targets_soft, activations = cache
        else:
            _, (targets_soft, activations) = \
              self.feed_forward(input, targets, return_cache=True,
                                dropout_predict=dropout_predict)

        # if not is_integer_array(targets):
        #     targets = targets_soft.argmax(1)
        targets = targets.get().argmax(1)
        class_error = np.sum(activations.get().argmax(1) != targets)

        if average: class_error = class_error.mean()
        return class_error

    def kl_error(self, input, targets, average=True, 
                 cache=None):
        """ The KL divergence error

        """
        
        if cache is not None:
            targets_soft, activations = cache
        else:
            _, (targets_soft, activations) = \
              self.feed_forward(input, targets, return_cache=True,
                                dropout_predict=dropout)

        kl_error = gpuarray.sum(targets * (cumath.log2(targets + eps) -
                                           cumath.log2(activations + eps)))
        if average:
            kl_error /= targets.shape[0]
        return kl_error

class NeuralNet(object):
    """ A Neural Network Object

    """

    TopLayerClass = LogisticLayer

    def __init__(self, n_in, n_out, layers, activation_function='sigmoid', 
                 dropout=False, l1_penalty_weight=0., l2_penalty_weight=0.,
                 **kwargs):
        self.layers = layers
        self.n_layers = len(layers)

        if l1_penalty_weight is not None and \
           not np.isscalar(l1_penalty_weight) and \
           len(l1_penalty_weight) != (self.n_layers + 1):
            raise ValueError("l1_penalty_weight must be a scalar or have length %d",
                             self.n_layers + 1)

        if l2_penalty_weight is not None and \
           not np.isscalar(l2_penalty_weight) and \
           len(l2_penalty_weight) != (self.n_layers + 1):
            raise ValueError("l2_penalty_weight must be a scalar or have length %d",
                             self.n_layers + 1)

        if np.isscalar(l1_penalty_weight):
            self.l1_penalty_weight_hidden = self.n_layers * [l1_penalty_weight]
            self.l1_penalty_weight_output = l1_penalty_weight
        else:
            self.l1_penalty_weight_hidden = l1_penalty_weight[:-1]
            self.l1_penalty_weight_output = l1_penalty_weight[-1]

        if np.isscalar(l2_penalty_weight):
            self.l2_penalty_weight_hidden = self.n_layers * [l2_penalty_weight]
            self.l2_penalty_weight_output = l2_penalty_weight
        else:
            self.l2_penalty_weight_hidden = l2_penalty_weight[:-1]
            self.l2_penalty_weight_output = l2_penalty_weight[-1]
        
        if type(dropout) is not list:
            if self.n_layers:
                dropout = self.n_layers * [dropout]
            else:
                dropout = [False]

        self.hidden_layers = []
        for i, hidden_layer in enumerate(layers):
            if isinstance(hidden_layer, HiddenLayer):
                self.hidden_layers.append(hidden_layer)
            elif isinstance(hidden_layer, int):
                if activation_function == 'tanh':
                    hidden_layer_class = TanhHiddenLayer
                elif activation_function == 'sigmoid':
                    hidden_layer_class = SigmoidHiddenLayer
                elif activation_function == 'relu':
                    hidden_layer_class = ReluHiddenLayer
                else:
                    raise ValueError('unknown activation function "%s"' % activation_function)

                n_in_hidden = self.hidden_layers[-1].n_units if i > 0 else n_in
                self.hidden_layers.append(
                    hidden_layer_class(n_in_hidden, hidden_layer,
                                       dropout=dropout[i],
                                       l1_penalty_weight=self.l1_penalty_weight_hidden[i],
                                       l2_penalty_weight=self.l2_penalty_weight_hidden[i]))
                
        self.n_units_hidden = [hl.n_units for hl in self.hidden_layers]
        
        n_in_top_layer = self.n_units_hidden[-1] if self.n_units_hidden else n_in

        assert issubclass(self.TopLayerClass, TopLayer)
        self.top_layer = self.TopLayerClass(n_in_top_layer, n_out, 
                                            l1_penalty_weight=self.l1_penalty_weight_output,
                                            l2_penalty_weight=self.l2_penalty_weight_output,
                                            **kwargs)

        # The scaling factor for the learning rate, based on the fan-in of the layer
        self.lr_multiplier = 2 * self.top_layer.n_tasks * [1. / np.sqrt(n_in, dtype=np.float32)]
        for i, n_hidden in enumerate(self.n_units_hidden):
            self.lr_multiplier.extend(2 * [1. / np.sqrt(n_hidden, dtype=np.float32)])
        self.lr_multiplier = np.array(self.lr_multiplier)
            
        self.n_in = n_in
        self.n_out = n_out
        self.dropout=dropout
        if kwargs.has_key('test_error_fct'):
            self.test_error_fct_name = kwargs['test_error_fct']
        self.activation_function = activation_function

    def getParameters(self):
        # Gather the parameters
        parameters = []
        for hl in self.hidden_layers:
            parameters.extend([hl.W, hl.b])
        parameters.extend([self.top_layer.W, self.top_layer.b])
        return parameters

    def setParameters(self, value):
        num_parameters = 2 * (len(self.hidden_layers) + 1)
        if len(value) != num_parameters:
            raise ValueError("Incorrect length of parameter vector. Model has %d parameters, but got %d" %
                             (num_parameters, len(value)))
        
        for i in range(len(self.hidden_layers)):
            self.hidden_layers[i].W = value[2*i]
            self.hidden_layers[i].b = value[2*i+1]

        self.top_layer.W = value[-2]
        self.top_layer.b = value[-1]

    parameters = property(getParameters, setParameters)

    def evaluate(self, input, targets, return_cache=False, dropout_predict=True):
        """ Evaluate the loss function without computing gradients

        """

        if dropout_predict:
            dropout_predict = self.dropout
        elif self.hidden_layers:
            dropout_predict = self.n_layers * [False]
        else:
            dropout_predict = [False]
        
        # Forward pass
        hidden_cache = []
        # Input layer never has dropout
        if self.hidden_layers:
            hidden_cache.append(self.hidden_layers[0].feed_forward(input,
                                                                   dropout_predict=False))

        for i in range(1, self.n_layers):
            hidden_activations = hidden_cache[i - 1]
            # Use dropout predict if previous layer has dropout
            hidden_cache.append(self.hidden_layers[i]
                                .feed_forward(hidden_activations,
                                              dropout_predict=dropout_predict[i - 1]))

        if self.hidden_layers:
            hidden_activations = hidden_cache[-1]
        else:
            hidden_activations = input

        # Use dropout_predict if last hidden layer has dropout
        loss, logistic_cache = \
          self.top_layer.feed_forward(hidden_activations, 
                                      targets, return_cache=True,
                                      dropout_predict=dropout_predict[-1])

        for hl in self.hidden_layers:
            if hl.l1_penalty_weight: loss += hl.l1_penalty
            if hl.l2_penalty_weight: loss += hl.l2_penalty

        if self.top_layer.l1_penalty_weight: loss += self.top_layer.l1_penalty
        if self.top_layer.l2_penalty_weight: loss += self.top_layer.l2_penalty

        if not return_cache:
            return loss
        else:
            return loss, hidden_cache, logistic_cache

    def training_pass(self, input, targets):
        """ Perform a full forward and backward pass through the model

        """
        
        # Forward pass
        loss, hidden_cache, logistic_cache = self.evaluate(input, targets, 
                                                           return_cache=True,
                                                           dropout_predict=False)

        # Backpropagation
        if self.hidden_layers:
            hidden_activations = hidden_cache[-1]
        else:
            hidden_activations = input

        df_top_layer = \
          self.top_layer.backprop(hidden_activations, targets,
                                  return_cache=False,
                                  cache=logistic_cache)
        gradients = list(df_top_layer[:-1][::-1])
        df_hidden = df_top_layer[-1]

        hidden_inputs = [input] + hidden_cache[:-1]
        for hl, hc, hi in \
            zip(self.hidden_layers[::-1], hidden_cache[::-1], 
                hidden_inputs[::-1]):
            df_W, df_b, df_hidden = hl.backprop(hi, df_hidden, cache=hc)
            gradients.extend([df_b, df_W])

        gradients.reverse()

        return loss, gradients

    def test_error(self, input, targets, average=True, cache=None):
        """ Evaulate performance on a test set

        """
        if cache is None:
            loss, hidden_cache, logistic_cache = self.evaluate(input, targets,
                                                               return_cache=True,
                                                               dropout_predict=True)
        else:
            loss, hidden_cache, logistic_cache = cache

        if self.hidden_layers:
            hidden_activations = hidden_cache[-1]
        else:
            hidden_activations = input

        return self.top_layer.test_error(hidden_activations, targets, average=average,
                                         cache=logistic_cache, dropout_predict=True)

    def predict(self, input):
        """ Get predictions from the model
        """
        
        dropout_predict = True

        if self.hidden_layers:
            # Forward pass
            hidden_cache = []
            # Input layer never has dropout
            hidden_cache.append(self.hidden_layers[0].feed_forward(input,
                                                                   dropout_predict=False))

            for i in range(1, self.n_layers):
                hidden_activations = hidden_cache[i - 1][1]
                # Use dropout predict if previous layer has dropout
                hidden_cache.append(self.hidden_layers[i]
                                    .feed_forward(hidden_activations,
                                                  dropout_predict=dropout_predict))

            hidden_activations = hidden_cache[-1][1]

        else:
            hidden_activations = input
            
        # Use dropout_predict if last hidden layer has dropout
        prediction = \
          self.top_layer.predict(hidden_activations, 
                                 dropout_predict=dropout_predict)

        return prediction
