from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="hormoglucose",
    version="1.0.0",
    description=("Extension of the SimGlucose type 1 diabetes simulator with "
                 "cortisol, growth hormone and exercise dynamics"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Ludivine Lasserre",
    license="MIT",
    packages=find_packages(include=["hormoglucose", "hormoglucose.*"]),
    include_package_data=True,
    package_data={"hormoglucose": ["params/*.csv"]},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.25",
        "pandas>=2.0",
        "scipy>=1.11",
        "matplotlib>=3.7",
        "gymnasium~=0.29.1",
        "pathos>=0.3.1",
        "setuptools<81",  # pkg_resources is used to locate the parameter files
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    zip_safe=False,
)
