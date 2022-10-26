from setuptools import find_packages, setup

# makes project pip installable (pip install -e .) so src can be imported

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='Source code of the project, containing utility functions, visualization code, etc.',
    author='REDACTED',
    license='LICENSE',
)
