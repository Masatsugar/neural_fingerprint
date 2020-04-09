# Neural Graph Fingerprint
* [Convolutional Networks on Graphs for Learning Molecular Fingerprints](https://arxiv.org/pdf/1509.09292.pdf)
 by David Duvenaud et al.
 
 This software is implemented by DGL + PyTorch. Note that this version cannot use mini-batch training as it is.
 
## Requirements
Prerequisites include:

* Numpy, matplotlib
* RDkit
* PyTorch
* DGL

## Examples
In `evaluation.py`, you can see the r2 and RMSE score based on ECFP, NFP, RDKit Morgan Fingerprint by using zinc dataset.
