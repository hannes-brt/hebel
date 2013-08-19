from .neural_net import NeuralNet


class LogisticRegression(NeuralNet):
    """ A logistic regression model

    """

    def __init__(self, n_in, n_out, test_error_fct='class_error'):
        super(LogisticRegression, self).\
            __init__(n_in, n_out, [],
                     test_error_fct=test_error_fct)
