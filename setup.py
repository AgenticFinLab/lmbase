import io
import os

import setuptools


def get_long_description():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    with io.open(os.path.join(base_dir, "README.md"), encoding="utf-8") as f:
        return f.read()


def get_requirements():
    with open("requirements.txt") as f:
        return f.read().splitlines()


setuptools.setup(
    name="lmbase-learn",
    use_scm_version=True,
    author="",
    license="Apache-2.0",
    description="Packaged version of the lmbase platform for large model research",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/AgenticFinLab/lmbase",
    packages=setuptools.find_packages(exclude=["tests"]),
    python_requires=">=3.6",
    install_requires=get_requirements(),
    extras_require={"tests": ["pytest"]},
    include_package_data=True,
    options={"bdist_wheel": {"python_tag": "py36.py37.py38.py39"}},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Education",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="large-language-model, deep-learning, complex reasoning",
)
