from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='Msigdb_terms_research',
    version='0.1',
    packages=find_packages(),
    description='',
    author='Yuhang Huang',
    url='',
    install_requires=requirements,  
    python_requires='>=3.8',
)
