from setuptools import setup, find_packages

setup(name='mdl',
      packages=find_packages(exclude=['tests']),
      zip_safe=False,
      keywords=['convolutional neural network', 'neural network', 'tensorflow', 'mnist'])
