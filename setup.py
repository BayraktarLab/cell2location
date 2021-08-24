import sys

from setuptools import find_packages
from setuptools import setup


def setup_package():
    install_requires = [
        'pymc3>=3.8,<3.10', 'pyro-ppl>=1.7.0', 'scvi-tools>=0.12.2', 'torch>=1.9.0',
        'numpy', 'pandas', 'scanpy',
        'plotnine'
    ]
    metadata = dict(
        name='cell2location',
        version='0.06',
        description='cell2location: High-throughput spatial mapping of cell types',
        url='https://github.com/BayraktarLab/cell2location',
        author='Vitalii Kleshchevnikov, Artem Shmatko, Emma Dann, Artem Lomakin, Alexander Aivazidis',
        author_email='vitalii.kleshchevnikov@sanger.ac.uk',
        license='Apache License, Version 2.0',
        packages=find_packages(),
        install_requires=install_requires
    )

    setup(**metadata)


if __name__ == '__main__':
    if sys.version_info < (2, 7):
        sys.exit('Sorry, Python < 2.7 is not supported')

    setup_package()
