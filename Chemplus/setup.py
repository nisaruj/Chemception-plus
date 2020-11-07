from setuptools import setup, find_packages
import os

dirname = os.path.dirname(__file__)
with open(os.path.join(dirname, 'README.md'), 'r') as fp:
    long_description = fp.read()

setup(
    name='CPCU-Chemplus-PNA',
    version='1.0.0',
    author='P. N. A.',
    # author_email='p.amonpitakpun@gmail.com',
    description='An package',
    long_description=long_description,
    # url='http://pypi.python.org/pypi/PackageName/',
    # license='LICENSE.txt',
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    # scripts=['bin/script1','bin/script2'],
    # license='LICENSE.txt',
    # install_requires=[
    #     "Django >= 1.1.1",
    #     "pytest",
    # ],
)