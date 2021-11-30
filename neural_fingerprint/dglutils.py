import dgl
import torch
from dgllife.utils import (
    BaseAtomFeaturizer,
    CanonicalAtomFeaturizer,
    CanonicalBondFeaturizer,
    mol_to_bigraph,
)

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
