from scikits.cuda import linalg
linalg.init()

import os as _os
neural_nets_root = _os.path.split(
    _os.path.abspath(_os.path.dirname(__file__)))[0]
