from setuptools import setup
from setuptools import find_packages

setup(
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    version="0.0.1",
)
