# ==============================================================================
# Copyright 2026 Luca Della Libera.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Setup script."""

import os

from setuptools import find_packages, setup


_ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

_VERSION = {}
with open(os.path.join(_ROOT_DIR, "dycast", "version.py")) as f:
    exec(f.read(), _VERSION)

with open(os.path.join(_ROOT_DIR, "README.md"), encoding="utf-8") as f:
    _README = f.read()

_REQUIREMENTS_SETUP = ["setuptools", "wheel"]

with open(os.path.join(_ROOT_DIR, "requirements.txt")) as f:
    _REQUIREMENTS = [line.strip() for line in f.readlines()]


setup(
    name="dycast",
    version=_VERSION["VERSION"],
    description="Dynamic character-aligned speech tokenizer",
    long_description=_README,
    long_description_content_type="text/markdown",
    author="Luca Della Libera",
    author_email="luca.dellalib@gmail.com",
    url="https://github.com/lucadellalib/dycast",
    license="Apache License 2.0",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Environment :: GPU :: NVIDIA CUDA",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed",
    ],
    keywords=[
        "Neural Audio Codecs",
        "Variable Frame Rate",
        "Focal Modulation",
        "Vector Quantization",
        "PyTorch",
        "Speech Tokenization",
        "Speech Compression",
        "Speech Processing",
    ],
    packages=find_packages(include=["dycast"]),
    include_package_data=True,
    install_requires=_REQUIREMENTS,
    setup_requires=_REQUIREMENTS_SETUP,
)
