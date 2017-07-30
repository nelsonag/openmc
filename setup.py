#!/usr/bin/env python

import glob
import numpy as np
try:
    from setuptools import setup
    have_setuptools = True
except ImportError:
    from distutils.core import setup
    have_setuptools = False

try:
    from distutils.extension import Extension
    from Cython.Build import cythonize
    have_cython = True
except ImportError:
    have_cython = False


# Get version information from __init__.py. This is ugly, but more reliable than
# using an import.
with open('openmc/__init__.py', 'r') as f:
    version = f.readlines()[-1].split()[-1].strip("'")

kwargs = {'name': 'openmc',
          'version': version,
          'packages': ['openmc', 'openmc.data', 'openmc.mgxs', 'openmc.model',
                       'openmc.stats'],
          'scripts': glob.glob('scripts/openmc-*'),

          # Metadata
          'author': 'Will Boyd',
          'author_email': 'wbinventor@gmail.com',
          'description': 'OpenMC Python API',
          'url': 'https://github.com/mit-crpg/openmc',
          'classifiers': [
              'Intended Audience :: Developers',
              'Intended Audience :: End Users/Desktop',
              'Intended Audience :: Science/Research',
              'License :: OSI Approved :: MIT License',
              'Natural Language :: English',
              'Programming Language :: Python',
              'Topic :: Scientific/Engineering'
          ]}

if have_setuptools:
    kwargs.update({
        # Required dependencies
        'install_requires': ['six', 'numpy>=1.9', 'h5py', 'scipy', 'pandas>=0.17.0'],

        # Optional dependencies
        'extras_require': {
            'decay': ['uncertainties'],
            'plot': ['matplotlib', 'ipython'],
            'vtk': ['vtk', 'silomesh'],
            'validate': ['lxml'],
            'reconstruct': ['cython'],
            'fast_call': ['cython'],
            'ndpp': ['cython']
        },

        # Data files
        'package_data': {
            'openmc.data': ['mass.mas12', 'fission_Q_data_endfb71.h5']
        },
    })

# If Cython is present, add resonance reconstruction capability
if have_cython:
    extensions = [
        Extension("openmc.data.function_methods_cython",
                  ['openmc/data/function_methods_cython.pyx'],
                  include_dirs=[np.get_include()],
                  extra_compile_args=['-O3'], extra_link_args=['-O3']),
        Extension("openmc.data.reconstruct",
                  ['openmc/data/reconstruct.pyx'],
                  include_dirs=[np.get_include()],
                  extra_compile_args=['-O3'], extra_link_args=['-O3']),
        Extension("openmc.stats.univariate_methods_cython",
                  ['openmc/stats/univariate_methods_cython.pyx'],
                  include_dirs=[np.get_include()],
                  extra_compile_args=['-O3'], extra_link_args=['-O3']),
        Extension("openmc.ndpp",
                  ['openmc/ndpp/*.pyx'],
                  include_dirs=[np.get_include()],
                  extra_compile_args=['-O3'], extra_link_args=['-O3'])]
    kwargs.update({
        'ext_modules': cythonize(extensions)
    })

setup(**kwargs)

