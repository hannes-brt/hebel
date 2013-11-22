# Hebel

GPU-Accelerated Deep Learning Library in Python

Litzelstetten is a library for deep learning with neural networks in Python using GPU acceleration with CUDA through PyCUDA. It implements the most important types of neural network models and offers a variety of different activation functions and training methods such as momentum, Nesterov momentum, dropout, and early stopping.

## Compatibility

Currently, Litzelstetten will only run on Linux and probably Mac OS X (not tested). Litzelstetten currently won't run in Windows, because scikits.cuda is not supported in Windows.

## Dependencies
- PyCUDA
- scikits.cuda
- numpy
- yaml
- skdata (only for MNIST example)

## Getting started
Study the yaml configuration files in `examples/` and run
    
    python train_model.py examples/mnist_neural_net_shallow.yaml

## What's with the name?
_Hebel_ is the German word for _lever_, one of the oldest tools that humans use. As Archimedes said it: _"Give me a lever long enough and a fulcrum on which to place it, and I shall move the world."_
