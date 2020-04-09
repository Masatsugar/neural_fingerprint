# Neural Graph Fingerprint
* [Convolutional Networks on Graphs for Learning Molecular Fingerprints](https://arxiv.org/pdf/1509.09292.pdf)
 by David Duvenaud et al.
 
 This software is implemented by DGL + PyTorch. Note that this version cannot use mini-batch nor GPU training as it is.
 
## Requirements
Prerequisites include:

* Numpy, matplotlib, scikit-learn
* RDkit
* PyTorch
* DGL, DGL-LifeSci

## Examples
In `evaluation.py`, you can see the r2 and RMSE score based on ECFP, NFP, RDKit Morgan Fingerprint in zinc dataset.
