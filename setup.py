from setuptools import setup, find_packages
import sys


setup(name='qmap',
      packages=[package for package in find_packages() if package.startswith('qmap')],
      description="qmap",
      author="Fabio Pardo",
      url='https://github.com/fabiopardo/qmap',
      author_email="f.pardo@imperial.ac.uk",
      version="0.1")
