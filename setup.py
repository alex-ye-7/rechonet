#!/usr/bin/env python3
"""Metadata for package to allow installation with pip."""

import os
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

# https://packaging.python.org/guides/single-sourcing-package-version/
version = {}
with open(os.path.join("rechonet", "__version__.py")) as f:
    exec(f.read(), version)  # pylint: disable=W0122

setuptools.setup(
    name="rechonet",
    description="Implementation of video-based AI cardiac function assessment.",
    version=version["__version__"],
    url="https://github.com/alex-ye-7/rechonet",
    packages=setuptools.find_packages(),
    install_requires=[
        "click",
        "numpy",
        "pandas",
        "torch",
        "torchvision",
        "opencv-python",
        "scikit-image",
        "tqdm",
        "sklearn"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    entry_points={
        "console_scripts": [
            "rechonet=rechonet:main",
        ],
    }

)