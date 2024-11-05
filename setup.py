from setuptools import setup, find_packages

setup(
    name='scTransID',
    version='0.1',
    description='A Python package for cell type classification using Transformer models',
    author='Mostafa Malmir',
    author_email='malmir.edumail@gmail.com',
    url='https://https://github.com/msmalmir/scTransID',
    packages=find_packages(),
    install_requires=[
        'torch',
        'scanpy',
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn'
    ],
    python_requires='>=3.6',
)

