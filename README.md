# Hebel

GPU-Accelerated Deep Learning Library in Python

Hebel is a library for deep learning with neural networks in Python using GPU acceleration with CUDA through PyCUDA. It implements the most important types of neural network models and offers a variety of different activation functions and training methods such as momentum, Nesterov momentum, dropout, and early stopping.

I no longer actively develop Hebel. If you are looking for a deep learning framework in Python, I now recommend [Chainer](https://github.com/pfnet/chainer).

## Models

Right now, Hebel implements feed-forward neural networks for classification and regression on one or multiple tasks. Other models such as Autoencoder, Convolutional neural nets, and Restricted Boltzman machines are planned for the future.

Hebel implements dropout as well as L1 and L2 weight decay for regularization.

## Optimization

Hebel implements stochastic gradient descent (SGD) with regular and Nesterov momentum.

## Compatibility

Currently, Hebel will run on Linux and Windows, and probably Mac OS X (not tested). 

## Dependencies
- PyCUDA
- numpy
- PyYAML
- skdata (only for MNIST example)

## Installation

Hebel is on PyPi, so you can install it with

    pip install hebel

## Getting started
Study the yaml configuration files in `examples/` and run
    
    python train_model.py examples/mnist_neural_net_shallow.yml
    
The script will create a directory in `examples/mnist` where the models and logs are saved.

Read the Getting started guide at [hebel.readthedocs.org/en/latest/getting_started.html](http://hebel.readthedocs.org/en/latest/getting_started.html) for more information.

## Documentation
[hebel.readthedocs.org](http://hebel.readthedocs.org)

## Contact
Maintained by [Hannes Bretschneider](http://github.com/hannes-brt) (hannes@psi.utoronto.ca).
If your are using Hebel, please let me know whether you find it useful and file a Github issue if you find any bugs or have feature requests.

## Citing
[![http://dx.doi.org/10.5281/zenodo.10050](https://zenodo.org/badge/doi/10.5281/zenodo.10050.png)](http://dx.doi.org/10.5281/zenodo.10050)

If you make use of Hebel in your research, please cite it. The BibTeX reference is
    
    @article{Bretschneider:10050,
      author        = "Hannes Bretschneider",
      title         = "{Hebel - GPU-Accelerated Deep Learning Library in Python}",
      month         = "May",
      year          = "2014",
      doi           = "10.5281/zenodo.10050",
      url           = "https://zenodo.org/record/10050",
    }

## What's with the name?
_Hebel_ is the German word for _lever_, one of the oldest tools that humans use. As Archimedes said it: _"Give me a lever long enough and a fulcrum on which to place it, and I shall move the world."_
