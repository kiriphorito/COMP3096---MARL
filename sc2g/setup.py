#================================
# RESEARCH GROUP PROJECT [RGP]
#================================
# This file is part of the COMP3096 Research Group Project.

from setuptools import setup

setup(
    name='sc2g',
    version='1.0.0',
    install_requires=[
        'gym>=0.9.6',
        'pysc2>=2.0',
        'baselines>=0.1.5',
        'numpy',
        'absl-py',
    ],
)
