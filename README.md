# Litzelstetten

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
Litzelstetten is a village just outside the city of Konstanz in South-West Germany where the author grew up. Litzelstetten's most famous citizen was [Johann Martin Schleyer](http://en.wikipedia.org/wiki/Johann_Martin_Schleyer) (1831-1912), who was the village's priest and teacher and who invented the universal language Volapük, which could never rival Esperanto in popularity.

Litzelstetten is very fun to say! Here's how: https://soundcloud.com/oceanhug/litzelstetten-pronounciation
