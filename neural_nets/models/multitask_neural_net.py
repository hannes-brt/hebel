from neural_net import NeuralNet
from .multitask_top_layer import MultitaskTopLayer

class MultitaskNeuralNet(NeuralNet):
    TopLayerClass = MultitaskTopLayer
