from setuptools import setup, find_packages

long_description = \
'''
A Python Communications Library

This is public package that is intended to function as a communications
code library for use in communicaitons system simulation and development.

To install just download from the git and install

Pip:
PATH>pip install [-e] simpyle_comms
The -e installs the package in develop mode

Manual:
PATH>git clone https://github.com/jsochacki/simpyle_comms.git
Then go to the location that the package was cloned to and then run
PATH\simpyle_comms>python setup.py install [develop]
Specifying develop in place of install installs in develop mode
'''

setup(
    name='simpyle_comms',
    version='0.0.0',
    license='BSD-3 clause',
    description='A Python Communications Library',
    long_description=long_description,
    author='socHACKi',
    author_email='johnsochacki@hotmail.com',
    url='https://github.com/jsochacki',
    packages = find_packages(exclude=['*test*']),
    install_requires=['numpy', 'scipy'],
    keywords = ['modem', 'communications', 'signal processing'],
)
