from setuptools import setup, find_packages

setup(
    name='ASIC-ML',
    version='0.0.1',
    url='https://github.com/aeftimia/ASIC-ML.git',
    author='Alex Eftimiades',
    author_email='alexeftimiades@gmail.com',
    description='Machine learning for building ASICs',
    packages=find_packages(),    
    install_requires=['numpy', 'pytest', 'torch'],
)
