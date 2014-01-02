Hebel Changelog
===============

Version 0.01
------------

01-01-2014

* Removed dependency on scikits.cuda (this should make Hebel
  compatible with Windows, but I couldn't test that yet)

* Serious speed-ups by avoiding freeing and reallocating memory for
  temporary objects. Previously, many temporary gpuarrays were
  reallocated in every single minibatch and then discarded, which was
  very inefficient. By using persistent objects for temporary objects
  across minibatches and some other improvements such as doing more
  computations in-place, a roughly 2x speed-up could be realised.
