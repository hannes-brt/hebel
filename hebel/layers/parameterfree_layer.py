from .hidden_layer import HiddenLayer

class ParameterfreeLayer(HiddenLayer):
    n_parameters = 0
    lr_multiplier = []
    l1_penalty_weight = 0.
    l2_penalty_weight = 0.

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
