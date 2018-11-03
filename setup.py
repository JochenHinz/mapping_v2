import numpy, nutils, os

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

extra = {}
try:
  from setuptools import setup, Extension
except:
  from distutils.core import setup, Extension
else:
  extra['install_requires'] = [ 'numpy>=1.8', 'matplotlib>=1.3', 'scipy>=0.13' ]

long_description = """
The mapping library for Python 3, version 2beta.
"""

setup(
  name='mapping_2',
  version='2beta',
  description='Mapping',
  author='Jochen Hinz',
  author_email='j.p.hinz@tudelft.nl',
  url='http://google.com',
  packages=[ 'mapping_2' ],
  long_description=long_description,
  **extra
)
