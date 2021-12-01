# Neural Graph Fingerprint
* [Convolutional Networks on Graphs for Learning Molecular Fingerprints](https://arxiv.org/pdf/1509.09292.pdf)
 by David Duvenaud et al.

 This software is implemented by DGL + PyTorch. Note that this version cannot use mini-batch nor GPU training as it is.

## Requirements
Prerequisites include below:

* [RDkit](https://www.rdkit.org/docs/Install.html)
* [PyTorch](https://pytorch.org/)
* [DGL](https://www.dgl.ai/)
* [DGL-LifeSci](https://lifesci.dgl.ai/install/index.html)

After installing them, do commands below:

```shell
git clone https://github.com/Masatsugar/neural_fingerprint.git
python setup.py install
```

## Examples
* minimum usage

```python
from neural_fingerprint import NFPRegressor
import rdkit.Chem as Chem
from rdkit.Chem import Descriptors
import numpy as np

mols = [Chem.MolFromSmiles(smi) for smi in ['CC', 'CCC', 'COC', 'C(=O)CC', 'CNC']]
logP = np.array([Descriptors.MolLogP(mol) for mol in mols])
nfp = NFPRegressor()
nfp.fit(mols, logP, epochs=10)
preds, fps = nfp.predict(mols, return_fps=True)
```