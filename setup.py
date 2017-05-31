#!/usr/bin/env python

try:
    from setuptools import setup
    have_setuptools = True
except ImportError:
    from distutils.core import setup
    have_setuptools = False

kwargs = {'name': 'opendeplete',
          'version': '0.1',
          'packages': ['opendeplete', 'opendeplete.integrator'],
          'scripts': [],

          # Metadata
          'author': 'Colin Josey',
          'author_email': 'cjosey@mit.edu',
          'description': 'OpenDeplete',
          'url': 'https://github.com/mit-crpg/opendeplete',
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
    # Required dependencies
    kwargs['install_requires'] = ['numpy', 'scipy', 'h5py', 'tqdm',
                                  'requests', 'openmc']

setup(**kwargs)
