import copy

import numpy as np
import rdkit.Chem as Chem


class ECFP:
    """calculate Extended Connectivity Fingerpritns.
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
        n_atoms = n_feat.shape[0]
        self.adj = Chem.GetAdjacencyMatrix(mol)

        # concatenate node features.
        self.n_feat = np.array(
            ["".join([str(f) for f in n_feat[atom]]) for atom in range(n_atoms)],
            dtype=str,
        )

    def _concat_neighbor(self, atom: int, n_feat: np.ndarray) -> str:
        """adjacency info."""
        nei_id = np.nonzero(self.adj[atom])[0]
        vec = "".join([str(ind) for ind in n_feat[nei_id]])
        return vec

    def calculate(self) -> np.ndarray:
        """
        :return: result of fingerprint.
        """
        n_atoms = self.n_feat.shape[0]
        identifier = copy.deepcopy(self.n_feat)
        for _ in range(0, self.radius):
            for atom in range(n_atoms):
                v = self._concat_neighbor(atom, identifier)
                identifier[atom] = hash(v)
                index = int(identifier[atom]) % self.nbits
                self.fps[index] = 1
        self.identifier = identifier
        return self.fps

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
