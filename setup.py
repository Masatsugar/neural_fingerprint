import setuptools

from neural_fingerprint import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="neural_fingerprint",
    version=__version__,
    description="Neural chemical fingerprint for predicting chemical properties. "
    "Reimplementation of `Convolutional Networks on Graphs for Learning Molecular Fingerprints by David Duvenaud et al.`",
    classifiers=[
        "Development Status :: 1 - Alpha",
        "License :: MIT`",
        "Programming Language :: Python :: 3.7",
        "Topic :: Cheminformatics :: Featurization",
    ],
    url="http://github.com/Masatsugar/neural_fingerprint",
    author="Masatugar",
    author_email="",
    license="MIT",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "networkx",
        "tqdm",
        "joblib",
        "pandas",
        "matplotlib",
        "seaborn",
    ],
    zip_safe=False,
)
