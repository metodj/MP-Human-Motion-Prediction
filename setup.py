"""Copyright (c) 2019 AIT Lab, ETH Zurich, Manuel Kaufmann, Emre Aksan

Students and holders of copies of this code, accompanying datasets,
and documentation, are not allowed to copy, distribute or modify
any of the mentioned materials beyond the scope and duration of the
Machine Perception course projects.

That is, no partial/full copy nor modification of this code and
accompanying data should be made publicly or privately available to
current/future students or other parties.
"""
from setuptools import setup, find_packages
"""Setup module for project."""

setup(
        name='mp19-project4-skeleton',
        version='0.2',
        description='Skeleton code for 2019 Machine Perception Human Motion Prediction project.',

        author='Manuel Kaufmann',
        author_email='kamanuel@inf.ethz.ch',

        packages=find_packages(exclude=[]),
        python_requires='>=3.6',
        install_requires=[
                # Add external libraries here.
                'tensorflow-gpu==1.12.0',
                'numpy',
                'matplotlib',
                'pandas',
                'opencv-python',
        ],
)