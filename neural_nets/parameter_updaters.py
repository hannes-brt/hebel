from pycuda import gpuarray
from itertools import izip
    
class ParameterUpdater(object):
    def __init__(self, model):
        self.model = model

    def pre_gradient_update(self, stream=None):
        pass

    def post_gradient_update(self, gradients, stream=None):
        pass
    
class SimpleSGDUpdate(ParameterUpdater):
    def post_gradient_update(self, gradients, batch_size, 
                             learning_parameters,
                             stream=None):
        learning_rate = learning_parameters[0]

        multiplier = learning_rate/batch_size
        self.model.update_parameters(gradients, multiplier)

class MomentumUpdate(ParameterUpdater):
    def __init__(self, model):
        self.model = model
        self.velocity = [gpuarray.zeros_like(p) 
                         for p in self.model.parameters]
        
    def post_gradient_update(self, gradients, batch_size, 
                             learning_parameters, stream=None):
        learning_rate, momentum = learning_parameters
        
        for param, gparam, vparam, lr_multiplier in \
          izip(self.model.parameters, gradients, self.velocity, 
              self.model.lr_multiplier):
                # vparam = self.momentum * vparam \
                #  -learning_rate * lr_multiplier / batch_size * gparam
                vparam._axpbyz(momentum,
                    gparam, -learning_rate * lr_multiplier / batch_size,
                    vparam, stream=stream)
                param._axpbyz(1., vparam, 1., param, stream=stream)

class NesterovMomentumUpdate(MomentumUpdate):
    def pre_gradient_update(self):
        """ First step of Nesterov momentum method:
        take step in direction of accumulated gradient
        """

        for (param, vparam) in izip(self.model.parameters, self.velocity):
                # param += vparam
                param._axpbyz(1., vparam, 1., param, stream=None)

    def post_gradient_update(self, gradients, batch_size,
                             learning_parameters, stream=None):
        """ Second step of Nesterov momentum method:
        take step in direction of new gradient and update velocity
        """

        learning_rate, momentum = learning_parameters

        for param, gparam, vparam, lr_multiplier in \
          izip(self.model.parameters, gradients, 
              self.velocity, self.model.lr_multiplier):

            # param -= learning_rate*lr_multiplier/batch_size*gparam
            param._axpbyz(1., gparam, -learning_rate*lr_multiplier/batch_size,
                          param, stream=stream)
            # vparam = momentum*vparam \
            #    - learning_rate*lr_multiplier/batch_size*gparam
            vparam._axpbyz(momentum, gparam, -learning_rate*lr_multiplier/batch_size,
                           vparam, stream=stream)

    # def rmsprop_update(self, gradient):
    #     new_rmsprop_avg = []
    #     new_gradient = []
        
    #     for r, g in izip(self.rmsprop_avg, gradient):
    #         new_r = self.rmsprop * r + (1 - self.rmsprop) * g**2. + eps

    #         if gpu.is_garray(g):
    #             new_g = g / gpu.sqrt(new_r)
    #         else:
    #             new_g = g / np.sqrt(new_r)
    #         new_gradient.append(new_g)
    #         new_rmsprop_avg.append(new_r)

    #     self.rmsprop_avg = new_rmsprop_avg
    #     return new_gradient


