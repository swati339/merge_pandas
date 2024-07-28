# setup.py
from setuptools import setup, find_packages

setup(
    name='my_project',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'requests',
        'beautifulsoup4',
    ],
    tests_require=[
        'pytest',
    ],
    entry_points={
        'console_scripts': [
            'my_project=app.__init__:main',
        ],
    },
)
