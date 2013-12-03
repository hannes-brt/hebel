# Hebel

GPU-Accelerated Deep Learning Library in Python

Hebel is a library for deep learning with neural networks in Python using GPU acceleration with CUDA through PyCUDA. It implements the most important types of neural network models and offers a variety of different activation functions and training methods such as momentum, Nesterov momentum, dropout, and early stopping.

## Models

Right now, Hebel implements only feed-forward neural networks for classification on one or multiple tasks. Other models such as neural network regression, Autoencoder, Convolutional neural nets, and Restricted Boltzman machines are planned for the future.

Hebel implements dropout as well as L1 and L2 weight decay for regularization.

## Optimization

Hebel implements stochastic gradient descent (SGD) with regular and Nesterov momentum.

## Compatibility

Currently, Hebel will only run on Linux and probably Mac OS X (not tested). Hebel currently won't run in Windows, because scikits.cuda is not supported in Windows.

## Dependencies
- PyCUDA
- scikits.cuda
- numpy
- PyYAML
- skdata (only for MNIST example)

## Getting started
Study the yaml configuration files in `examples/` and run
    
    python train_model.py examples/mnist_neural_net_shallow.yaml
    
The script will create a directory in `examples/mnist` where the models and logs are saved.

Read the Getting started guide at [hebel.readthedocs.org/en/latest/getting_started.html](http://hebel.readthedocs.org/en/latest/getting_started.html) for more information.

## Documentation
[hebel.readthedocs.org](http://hebel.readthedocs.org) (coming slowly)

## What's with the name?
_Hebel_ is the German word for _lever_, one of the oldest tools that humans use. As Archimedes said it: _"Give me a lever long enough and a fulcrum on which to place it, and I shall move the world."_
