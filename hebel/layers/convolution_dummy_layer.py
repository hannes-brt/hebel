from hebel.layers import DummyLayer


class ConvolutionDummyLayer(DummyLayer):
    def __init__(self, n_in, n_filters=4):
        self.n_filters = n_filters
        super(ConvolutionDummyLayer, self).__init__(n_in)

    @property
    def n_units_per_filter(self):
        return self.n_units