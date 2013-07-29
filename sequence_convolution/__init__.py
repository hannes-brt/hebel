import os as _os

sequence_conv_root = _os.path.split(
    _os.path.abspath(_os.path.dirname(__file__)))[0]

from scikits.cuda import linalg
linalg.init()

def enum(*sequential, **named):
    # Implementation of enums in Python 2
    enums = dict(zip(sequential, range(len(sequential))), **named)
    reverse = dict((value, key) for key, value in enums.iteritems())
    enums['reverse_mapping'] = reverse
    return type('Enum', (), enums)
