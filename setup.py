from setuptools import setup
from setuptools import find_packages

import sys

def setup_package():
  install_requires = ['pymc3', 'theano', 'pygpu', 'numpy', 'pandas', 'scanpy', 'plotnine']
  metadata = dict(
      name = 'pycell2location',
      version = '0.01',
      description = 'cell2location: Locating reference single cells and expression programmes to spatial sequencing data (aggregate across cells in a small tissue region)',
      url = 'https://github.com/vitkl/cell2location',
      author = 'Vitalii Kleshchevnikov, Emma Dann, Artem Lomakin, Artem Shmatko, Mika Jain',
      author_email = 'vitalii.kleshchevnikov@sanger.ac.uk',
      license = 'Apache License, Version 2.0',
      packages = find_packages(),
      install_requires = install_requires
    )

  setup(**metadata)

if __name__ == '__main__':
  if sys.version_info < (2,7):
    sys.exit('Sorry, Python < 2.7 is not supported')
    
  setup_package()
