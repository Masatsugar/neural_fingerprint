# Neural Graph Fingerprint
* [Convolutional Networks on Graphs for Learning Molecular Fingerprints](https://arxiv.org/pdf/1509.09292.pdf)
 by David Duvenaud et al.

 This software is implemented by DGL + PyTorch. Note that this version cannot use mini-batch nor GPU training as it is.

## Requirements
Prerequisites include:

* Numpy
* RDkit
* PyTorch
* DGL, DGL-LifeSci

## Examples
In `evaluation.py`, you can see the r2 and RMSE score based on ECFP, NFP, RDKit Morgan Fingerprint in zinc dataset.

* minimum usage

```python
from neural_fingerprint import NFPRegressor
import rdkit.Chem as Chem
from rdkit.Chem import Descriptors
import numpy as np

mols = [Chem.MolFromSmiles(smi) for smi in ['CC', 'CCC', 'COC', 'C(=O)CC', 'CNC']]
logP = np.array([Descriptors.MolLogP(mol) for mol in mols])
nfp = NFPRegressor()
nfp.fit(mols, logP, epochs=10, verbose=False)
nfp.predict(mols)
```
