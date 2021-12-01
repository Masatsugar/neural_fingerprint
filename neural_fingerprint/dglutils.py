import os
from collections import defaultdict

import dgl
import numpy as np
import rdkit.Chem as Chem
import torch
from dgl import backend as FF
from dgllife.utils import (
    BaseAtomFeaturizer,
    CanonicalAtomFeaturizer,
    CanonicalBondFeaturizer,
    atom_hybridization_one_hot,
    atom_is_aromatic,
    atom_type_one_hot,
    mol_to_bigraph,
)
from rdkit.Chem import ChemicalFeatures, RDConfig

from neural_fingerprint.chemutils import createNodeFeatures


class MyNodeFeaturizer(BaseAtomFeaturizer):
    def __init__(self, atom_data_field="h"):
        super(MyNodeFeaturizer, self).__init__(
            featurizer_funcs={atom_data_field: createNodeFeatures}
        )


class CustomDataset:
    def __init__(self):
        # Initialize Dataset and preprocess data
        self.smiles = []
        self.graphs = []
        self.labels = []

    def __getitem__(self, index):
        # Return the corresponding DGLGraph/label needed for training/evaluation based on index
        return self.smiles[index], self.graphs[index], self.labels[index]

    def __len__(self):
        return len(self.graphs)

    def add(self, smiles, graph, label):
        self.smiles.append(smiles)
        self.graphs.append(graph)

        label = torch.Tensor([label])
        self.labels.append(label)


def collate_molgraphs(data):
    assert len(data[0]) in [
        3,
        4,
    ], "Expect the tuple to be of length 3 or 4, got {:d}".format(len(data[0]))
    if len(data[0]) == 3:
        smiles, graphs, labels = map(list, zip(*data))
        masks = None
    else:
        smiles, graphs, labels, masks = map(list, zip(*data))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)

    if masks is None:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)
    return smiles, bg, labels, masks


def mol_to_graph(mols: list, canonical: bool = False) -> dgl.DGLGraph:
    if canonical:
        graph = [
            mol_to_bigraph(
                mol,
                node_featurizer=CanonicalAtomFeaturizer(),
                edge_featurizer=CanonicalBondFeaturizer(),
            )
            for mol in mols
        ]
    else:
        graph = [mol_to_bigraph(m, node_featurizer=MyNodeFeaturizer()) for m in mols]

    return graph


def alchemy_nodes(mol):
    """Featurization for all atoms in a molecule. The atom indices
    will be preserved.
    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule object
    Returns
    -------
    atom_feats_dict : dict
        Dictionary for atom features
    """
    atom_feats_dict = defaultdict(list)
    is_donor = defaultdict(int)
    is_acceptor = defaultdict(int)

    fdef_name = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    mol_featurizer = ChemicalFeatures.BuildFeatureFactory(fdef_name)
    mol_feats = mol_featurizer.GetFeaturesForMol(mol)
    mol_conformers = mol.GetConformers()
    assert len(mol_conformers) == 1

    for i in range(len(mol_feats)):
        if mol_feats[i].GetFamily() == "Donor":
            node_list = mol_feats[i].GetAtomIds()
            for u in node_list:
                is_donor[u] = 1
        elif mol_feats[i].GetFamily() == "Acceptor":
            node_list = mol_feats[i].GetAtomIds()
            for u in node_list:
                is_acceptor[u] = 1

    num_atoms = mol.GetNumAtoms()
    for u in range(num_atoms):
        atom = mol.GetAtomWithIdx(u)
        atom_type = atom.GetAtomicNum()
        num_h = atom.GetTotalNumHs()
        atom_feats_dict["node_type"].append(atom_type)

        h_u = []
        h_u += atom_type_one_hot(atom, ["H", "C", "N", "O", "F", "S", "Si"])
        h_u.append(atom_type)
        h_u.append(is_acceptor[u])
        h_u.append(is_donor[u])
        h_u += atom_is_aromatic(atom)
        h_u += atom_hybridization_one_hot(
            atom,
            [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
            ],
        )
        h_u.append(num_h)
        atom_feats_dict["n_feat"].append(FF.tensor(np.array(h_u).astype(np.float32)))

    atom_feats_dict["n_feat"] = FF.stack(atom_feats_dict["n_feat"], dim=0)
    atom_feats_dict["node_type"] = FF.tensor(
        np.array(atom_feats_dict["node_type"]).astype(np.int64)
    )

    return atom_feats_dict


def alchemy_edges(mol, self_loop=False):
    """Featurization for all bonds in a molecule.
    The bond indices will be preserved.
    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule object
    self_loop : bool
        Whether to add self loops. Default to be False.
    Returns
    -------
    bond_feats_dict : dict
        Dictionary for bond features
    """
    bond_feats_dict = defaultdict(list)

    mol_conformers = mol.GetConformers()
    assert len(mol_conformers) == 1
    geom = mol_conformers[0].GetPositions()

    num_atoms = mol.GetNumAtoms()
    for u in range(num_atoms):
        for v in range(num_atoms):
            if u == v and not self_loop:
                continue

            e_uv = mol.GetBondBetweenAtoms(u, v)
            if e_uv is None:
                bond_type = None
            else:
                bond_type = e_uv.GetBondType()
            bond_feats_dict["e_feat"].append(
                [
                    float(bond_type == x)
                    for x in (
                        Chem.rdchem.BondType.SINGLE,
                        Chem.rdchem.BondType.DOUBLE,
                        Chem.rdchem.BondType.TRIPLE,
                        Chem.rdchem.BondType.AROMATIC,
                        None,
                    )
                ]
            )
            bond_feats_dict["distance"].append(np.linalg.norm(geom[u] - geom[v]))

    bond_feats_dict["e_feat"] = FF.tensor(
        np.array(bond_feats_dict["e_feat"]).astype(np.float32)
    )
    bond_feats_dict["distance"] = FF.tensor(
        np.array(bond_feats_dict["distance"]).astype(np.float32)
    ).reshape(-1, 1)

    return bond_feats_dict
