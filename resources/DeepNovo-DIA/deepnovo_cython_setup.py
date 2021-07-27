# Copyright 2017 Hieu Tran. All Rights Reserved.
#
# DeepNovo is publicly available for non-commercial uses.
# ==============================================================================

"""TODO(nh2tran): docstring."""

import os
from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(ext_modules=cythonize("deepnovo_cython_modules.pyx"),
      include_dirs=[numpy.get_include(), os.path.join(numpy.get_include(), 'numpy')])
