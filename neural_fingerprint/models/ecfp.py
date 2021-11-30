import copy
from collections import defaultdict
from typing import Dict

import numpy as np
import pandas as pd
import rdkit.Chem as Chem


class ECFP:
    """calculate Extended Connectivity Fingerpritns.

    Note that this implementation of ECFP doesn't consider the duplication of substructures based on chemical properties,
    so that more bit collisions occur than the original one.

    :param mol: rdkit mol object
    :param radius (int):
    :param nbits (int):
    :param n_feat: feature matrix (num of nodes × num of feats)
    """

    def __init__(
        self, mol, radius: int = 2, nbits: int = 2048, n_feat: np.array = None
    ):
        self.mol = mol
        self.radius = radius
        self.nbits = nbits
        self.fps = np.zeros(shape=(self.nbits,), dtype=np.int32)

        if n_feat is None:
            n_feat = self.createNodeFeatures()

        n_feat = np.array(n_feat, dtype=np.int32)
        self.adj = Chem.GetAdjacencyMatrix(mol)

        # concatenate node features.
        self.identifier: Dict[int, Dict[int, int]] = defaultdict(dict)
        for i in range(radius + 1):
            self.identifier[i] = {}
        self.identifier[0].update(
            {
                i: k
                for i, k in enumerate(
                    [
                        hash("".join([str(f) for f in n_feat[i]]))
                        for i in range(len(n_feat))
                    ]
                )
            }
        )

    def _concat_neighbor(self, atom: int, identifier: Dict[int, int]) -> str:
        """adjacency info."""
        idx = [atom] + [i for i in np.nonzero(self.adj[atom])[0]]
        n_feat = np.array([str(f) for f in identifier.values()])
        # invariant
        vec = ",".join([f for f in n_feat[idx]])
        # vec = ",".join([i for i in set([f for f in n_feat[idx]])])
        return vec

    def _calculate(self) -> None:
        for r in range(self.radius):
            for atom in range(len(self.mol.GetAtoms())):
                v = self._concat_neighbor(atom, self.identifier[r])
                self.identifier[r + 1].update({atom: hash(v)})

    def calculate(self):
        """
        :return: result of fingerprint.
        """
        self._calculate()
        index = (
            np.unique(pd.DataFrame(self.identifier).to_numpy().flatten()) % self.nbits
        )
        fps = np.zeros(self.nbits)
        for i in index:
            fps[i] = 1
        return fps

    def createNodeFeatures(self) -> np.ndarray:
        features = np.array(
            [
                [
                    *[a.GetDegree() == i for i in range(5)],
                    *[a.GetExplicitValence() == i for i in range(9)],
                    *[int(a.GetHybridization()) == i for i in range(1, 7)],
                    *[a.GetImplicitValence() == i for i in range(9)],
                    a.GetIsAromatic(),
                    a.GetNoImplicit(),
                    *[a.GetNumExplicitHs() == i for i in range(5)],
                    *[a.GetNumImplicitHs() == i for i in range(5)],
                    *[a.GetNumRadicalElectrons() == i for i in range(5)],
                    a.IsInRing(),
                    *[a.IsInRingSize(i) for i in range(2, 9)],
                    # Add Features
                    a.GetAtomicNum(),
                    a.GetDegree(),
                    a.GetExplicitValence(),
                    a.GetImplicitValence(),
                    a.GetFormalCharge(),
                    a.GetTotalNumHs(),
                ]
                for a in self.mol.GetAtoms()
            ],
            dtype=np.int32,
        )

        return features
