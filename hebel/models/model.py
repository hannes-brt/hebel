class Model(object):
    """ Abstract base-class for a Hebel model
    """

    def __init__(self):
        raise NotImplentedError

    @property
    def parameters(self):
        raise NotImplentedError

    @parameters.setter
    def parameters(self, value):
        raise NotImplentedError

    def update_parameters(self, value):
        raise NotImplentedError

    def evaluate(self, input_data, targets,
                 return_cache=False, prediction=True):
        """ Evaluate the loss function without computing gradients
        """

        raise NotImplentedError

    def training_pass(self, input_data, targets):
        """ Perform a full forward and backward pass through the model
        """

        raise NotImplentedError

    def test_error(self, input_data, targets, average=True, cache=None):
        """ Evaulate performance on a test set

        """
        raise NotImplentedError

    def feed_forward(self, input_data, return_cache=False, prediction=True):
        """ Get predictions from the model
        """

        raise NotImplentedError
