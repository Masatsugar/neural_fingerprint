from typing import List

import dgl
import dgl.function as fn
import numpy
import rdkit
import rdkit.Chem as Chem
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from neural_fingerprint.dglutils import CustomDataset, collate_molgraphs, mol_to_graph
from torch.utils.data import DataLoader

gcn_msg = fn.copy_src(src="h", out="m")
gcn_reduce = fn.sum(msg="m", out="h")


class NFP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, depth=2, nbits=16):
        super(NFP, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, nbits)
        self.softmax = nn.Softmax(dim=1)

        self.depth = depth
        self.nbits = nbits

        self.linear3 = nn.Linear(nbits, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)

    def forward(self, g, n_feat):
        with g.local_scope():
            fps = torch.zeros([1, self.nbits])
            for _ in range(self.depth):
                g.ndata["h"] = n_feat
                g.update_all(gcn_msg, gcn_reduce)
                h = g.ndata["h"]

                r = F.relu(self.linear1(h))
                i = self.softmax(self.linear2(r))
                fps += torch.sum(i, dim=0)

            out = F.relu(self.linear3(fps))
            out = self.linear4(out).squeeze(0)
        return fps, out


class NFPRegressor:
    """Neural Fingerprints: property prediction for molecules

    Convolutional Networks on Graphs for Learning Molecular Fingerprints by David Duvenaud et al.
    https://arxiv.org/pdf/1509.09292.pdf

    Parameters
    ----------
    hidden_dim : int, default=64
        the number of hidden layer units of MLP.

    depth: int, default=2
        the number of message passing.

    nbits: int, default=16
        the number of low dimensional feature space. It is not as necessary as that of ECFP.

    Examples
    --------
    >>> from neural_fingerprint import NFPRegressor
    >>> import rdkit.Chem as Chem
    >>> from rdkit.Chem import Descriptors
    >>> import numpy as np
    >>> mols = [Chem.MolFromSmiles(smi) for smi in ['CC', 'CCC', 'COC', 'C(=O)CC', 'CNC']]
    >>> logP = np.array([Descriptors.MolLogP(mol) for mol in mols])
    >>> nfp = NFPRegressor()
    >>> nfp.fit(mols, logP, epochs=10, verbose=False)
    >>> nfp.predict(mols)
    array([[1.0752354 ],
       [1.4675996 ],
       [0.21585755],
       [0.57562566],
       [0.26256177]], dtype=float32)
    """

    def __init__(self, hidden_dim=64, depth=2, nbits=16):
        self.model = None
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.nbits = nbits

        self.input_dim = None

    def __repr__(self):
        return f"{self.__class__.__name__}(hidden_dim={self.hidden_dim}, depth={self.depth}, nbits={self.nbits})"

    def _preprocess(self, mols, train_y=None):
        if train_y is None:
            train_y = numpy.ones(len(mols))

        trainset = CustomDataset()
        graphs = mol_to_graph(mols, canonical=False)
        if self.input_dim is None:
            self.input_dim = graphs[0].ndata["h"].shape[1]

        smiles = [Chem.MolToSmiles(mol) for mol in mols]

        for smiles, graph, label in zip(smiles, graphs, train_y):
            trainset.add(smiles, graph, label)

        loader = DataLoader(trainset, batch_size=1, collate_fn=collate_molgraphs)

        return loader

    def fit(
        self,
        train_mols: List[rdkit.Chem.rdchem.Mol],
        y_train: numpy.ndarray,
        epochs: int = 10,
        lr: float = 0.01,
        verbose: bool = True,
    ):
        """
        :param train_mols:
        :param y_train:
        :param epochs:
        :param lr:
        :param verbose:
        :return:
        """
        trainloader = self._preprocess(train_mols, y_train)
        model = NFP(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            depth=self.depth,
            nbits=self.nbits,
        )
        self._train(model, trainloader, epochs, lr, verbose)
        return self

    def predict(
        self, test_mols: List[rdkit.Chem.rdchem.Mol], return_fps=True
    ) -> numpy.ndarray:
        if self.model is None:
            raise

        testloader = self._preprocess(test_mols, train_y=None)
        preds, fps = self._eval_pred(self.model, data_loader=testloader)
        if return_fps:
            return preds, fps
        else:
            return preds

    def _train(self, model, train_loader, epochs, lr, verbose):
        loss_func = nn.MSELoss(reduction="none")
        optimizer = optim.Adam(model.parameters(), lr=lr)
        model.train()
        epoch_losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            for ite, batch in enumerate(train_loader):
                _, bg, label, masks = batch
                n_feat = bg.ndata["h"]
                fps, prediction = model(bg, n_feat)
                loss = (loss_func(prediction, label) * (masks != 0).float()).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().item()

            epoch_loss /= ite + 1
            if verbose:
                print(f"Epoch {epoch}, loss {epoch_loss:.4f}")
            epoch_losses.append(epoch_loss)
        self.model = model

    def _eval_pred(self, model, data_loader):
        preds = []
        fps = []
        model.eval()
        with torch.no_grad():
            for _, (smi, bg, label, mask) in enumerate(data_loader):
                n_feat = bg.ndata["h"]
                fp, pred = model(bg, n_feat)
                fps.append(fp.detach().numpy()[0])
                preds.append(pred.detach().numpy()[0])

        fps = numpy.vstack(fps)
        preds = numpy.array(preds)[:, numpy.newaxis]
        return preds, fps
