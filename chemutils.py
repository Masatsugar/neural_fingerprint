import dgl
from dgllife.utils import mol_to_bigraph
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer, BaseAtomFeaturizer
import torch
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor


def createNodeFeatures(atom) -> list:
    features = [
        *[atom.GetDegree() == i for i in range(5)],
        *[atom.GetExplicitValence() == i for i in range(9)],
        *[int(atom.GetHybridization()) == i for i in range(1, 7)],
        *[atom.GetImplicitValence() == i for i in range(9)],
        atom.GetIsAromatic(),
        atom.GetNoImplicit(),
        *[atom.GetNumExplicitHs() == i for i in range(5)],
        *[atom.GetNumImplicitHs() == i for i in range(5)],
        *[atom.GetNumRadicalElectrons() == i for i in range(5)],
        atom.IsInRing(),
        *[atom.IsInRingSize(i) for i in range(2, 9)],
        # Add Features
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetExplicitValence(),
        atom.GetImplicitValence(),
        atom.GetFormalCharge(),
        atom.GetTotalNumHs()]

    return features


class MyNodeFeaturizer(BaseAtomFeaturizer):
    def __init__(self, atom_data_field='h'):
        super(MyNodeFeaturizer, self).__init__(featurizer_funcs={atom_data_field: createNodeFeatures})


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
    assert len(data[0]) in [3, 4], 'Expect the tuple to be of length 3 or 4, got {:d}'.format(len(data[0]))
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
        graph = [mol_to_bigraph(mol,
                                node_featurizer=CanonicalAtomFeaturizer(),
                                edge_featurizer=CanonicalBondFeaturizer()) for mol in mols]
    else:
        graph = [mol_to_bigraph(m, node_featurizer=MyNodeFeaturizer()) for m in mols]

    return graph


def rf_evalation(train_x, test_x, train_y, test_y, args=None):
    if train_x.shape[1] == 1:
        train_pred = train_x
        test_pred = test_x
    else:
        rf = RandomForestRegressor()
        rf.fit(train_x, train_y)
        train_pred = rf.predict(train_x)
        test_pred = rf.predict(test_x)

    r2 = round(r2_score(test_y, test_pred), 4)
    rmse = round(mean_squared_error(test_y, test_pred), 4)
    if args is not None:
        plt.title(f"ECFP r={args.rad}, nbits={args.nbits}, RMSE={rmse}, R2={r2}")
    else:
        plt.title(f"RMSE={rmse}, R2={r2}")

    print(f'r2={r2}, RMSE={rmse}')
    plt.scatter(test_y, test_pred, label='test')
    plt.scatter(train_y, train_pred, label='train', alpha=0.2)
    plt.legend()
    plt.show()


def str2nan(df):
    """convert str to nan for mordred descriptors include string object in data frame.

    :param df: pandas dataframe
    :return:
    """
    train = df.copy()
    filterdColumns = train.columns[train.dtypes == np.object]

    def tonan(x):
        if isinstance(x, str):
            return np.nan
        else:
            return x

    for col in filterdColumns:
        train.loc[:, col] = train.loc[:, col].map(tonan)

    return train


def calc_mordred_desc(mols: list):
    from mordred import Calculator, descriptors

    calc = Calculator(descriptors, ignore_3D=True)
    res = calc.pandas(mols)
    return res

