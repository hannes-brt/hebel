class Model(object):
    """ Abstract base-class for a Hebel model
    """

    def __init__(self):
        raise NotImplementedError

    @property
    def parameters(self):
        raise NotImplementedError

    @parameters.setter
    def parameters(self, value):
        raise NotImplementedError

    def update_parameters(self, value):
        raise NotImplementedError

    def evaluate(self, input_data, targets,
                 return_cache=False, prediction=True):
        """ Evaluate the loss function without computing gradients
        """

        raise NotImplementedError

    def training_pass(self, input_data, targets):
        """ Perform a full forward and backward pass through the model
        """

        raise NotImplementedError

    def test_error(self, input_data, targets, average=True, cache=None):
        """ Evaulate performance on a test set

        """
        raise NotImplementedError

    def feed_forward(self, input_data, return_cache=False, prediction=True):
        """ Get predictions from the model
        """

        raise NotImplementedError
