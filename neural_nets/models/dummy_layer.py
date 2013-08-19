from .hidden_layer import HiddenLayer


class DummyLayer(HiddenLayer):
    """ This class has no input and simply passes through its input
    """

    lr_multiplier = []
    n_parameters = 0
    l1_penalty_weight = 0.
    l2_penalty_weight = 0.
    dropout = False

    def __init__(self, n_in):
        self.n_in = n_in
        self.n_units = n_in

    @property
    def parameters(self):
        return []

    @parameters.setter
    def parameters(self, value):
        pass

    def update_parameters(self, values, stream=None):
        pass

    @property
    def l1_penalty(self):
        return 0.

    @property
    def l2_penalty(self):
        return 0.

    def feed_forward(self, input_data, prediction=False):
        assert input_data.shape[1] == self.n_in
        return (input_data,)

    def backprop(self, input_data, df_output, cache=None):
        return tuple(), df_output
