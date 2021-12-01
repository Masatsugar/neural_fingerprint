from functools import partial

import dgl
import numpy
import pandas
import rdkit.Chem as Chem
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dgl.nn.pytorch import NNConv, Set2Set
from dgllife.utils import (
    BaseAtomFeaturizer,
    BaseBondFeaturizer,
    CanonicalAtomFeaturizer,
    CanonicalBondFeaturizer,
    ConcatFeaturizer,
    EarlyStopping,
    Meter,
    atom_degree_one_hot,
    atom_formal_charge,
    atom_hybridization_one_hot,
    atom_implicit_valence,
    atom_is_aromatic,
    atom_num_radical_electrons,
    atom_total_num_H_one_hot,
    atom_type_one_hot,
    bond_is_in_ring,
    bond_type_one_hot,
    mol_to_bigraph,
    mol_to_complete_graph,
)
from rdkit.Chem import AllChem
from torch.utils.data import DataLoader
from tqdm import tqdm

from neural_fingerprint.dglutils import alchemy_edges, alchemy_nodes


class MPNNModel(nn.Module):
    """
    MPNN from
    `Neural Message Passing for Quantum Chemistry <https://arxiv.org/abs/1704.01212>`__
    Parameters
    ----------
    node_input_dim : int
        Dimension of input node feature, default to be 15.
    edge_input_dim : int
        Dimension of input edge feature, default to be 15.
    output_dim : int
        Dimension of prediction, default to be 12.
    node_hidden_dim : int
        Dimension of node feature in hidden layers, default to be 64.
    edge_hidden_dim : int
        Dimension of edge feature in hidden layers, default to be 128.
    num_step_message_passing : int
        Number of message passing steps, default to be 6.
    num_step_set2set : int
        Number of set2set steps
    num_layer_set2set : int
        Number of set2set layers
    """

    def __init__(
        self,
        node_input_dim=15,
        edge_input_dim=5,
        output_dim=12,
        node_hidden_dim=64,
        edge_hidden_dim=128,
        num_step_message_passing=6,
        num_step_set2set=6,
        num_layer_set2set=3,
    ):
        super(MPNNModel, self).__init__()

        self.num_step_message_passing = num_step_message_passing
        self.lin0 = nn.Linear(node_input_dim, node_hidden_dim)
        edge_network = nn.Sequential(
            nn.Linear(edge_input_dim, edge_hidden_dim),
            nn.ReLU(),
            nn.Linear(edge_hidden_dim, node_hidden_dim * node_hidden_dim),
        )
        self.conv = NNConv(
            in_feats=node_hidden_dim,
            out_feats=node_hidden_dim,
            edge_func=edge_network,
            aggregator_type="sum",
        )
        self.gru = nn.GRU(node_hidden_dim, node_hidden_dim)

        self.set2set = Set2Set(node_hidden_dim, num_step_set2set, num_layer_set2set)

        self.lin1 = nn.Linear(2 * node_hidden_dim, node_hidden_dim)
        self.lin2 = nn.Linear(node_hidden_dim, output_dim)

    def forward(self, g, n_feat, e_feat):
        """Predict molecule labels
        Parameters
        ----------
        g : DGLGraph
            Input DGLGraph for molecule(s)
        n_feat : tensor of dtype float32 and shape (B1, D1)
            Node features. B1 for number of nodes and D1 for
            the node feature size.
        e_feat : tensor of dtype float32 and shape (B2, D2)
            Edge features. B2 for number of edges and D2 for
            the edge feature size.
        Returns
        -------
        res : Predicted labels
        """
        out = F.relu(self.lin0(n_feat))  # (B1, H1)
        h = out.unsqueeze(0)  # (1, B1, H1)

        for i in range(self.num_step_message_passing):
            m = F.relu(self.conv(g, out, e_feat))  # (B1, H1)
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        readout = self.set2set(g, out)
        out = F.relu(self.lin1(readout))
        out = self.lin2(out)
        return out, readout


class DGLDataset(object):
    def __init__(self, method="Simple"):
        self.smiles = []
        self.graphs = []
        self.labels = []
        self.x_feat = []
        self.method = method

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graphs)

    def __getitem__(self, idx):
        """Get the i^th sample.

        Paramters
        ---------
        idx : int
            The sample index.

        Returns
        -------
        (dgl.DGLGraph, int)
            The graph and its label.
        """
        return (
            self.smiles[idx],
            self.graphs[idx],
            self.labels[idx],
        )  # , self.x_feat[idx]

    def add(self, train_mols, label):
        """

        :param train_mols:
        :param label:
        :return:
        """
        if self.method == "MPNN":
            graph = self.__Featurizer(train_mols)
        elif self.method == "MMFF":
            graph = self.__FeaturizerMMFF(train_mols)
        elif self.method == "Canonical":
            graph = self.__CanonicalFeatureize(train_mols)
        elif self.method == "Simple":
            graph = self.__FeaturizerSimple(train_mols)
        else:
            raise ValueError(
                f'Expect method to be "MPNN" or "MMFF" or "Canonical" or "Simple", got "{self.method}"'
            )

        self.smiles = [Chem.MolToSmiles(mol) for mol in train_mols]
        self.graphs = graph
        self.labels = torch.Tensor(label).reshape(-1, 1)

    def __CanonicalFeatureize(self, train_mols) -> list:
        atom_featurizer = CanonicalAtomFeaturizer("n_feat")
        bond_featurizer = CanonicalBondFeaturizer("e_feat")

        train_graph = [
            mol_to_bigraph(
                mol, node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer
            )
            for mol in train_mols
        ]
        return train_graph

    def __Featurizer(self, train_mols) -> list:
        atom_featurizer = BaseAtomFeaturizer(
            {
                "n_feat": ConcatFeaturizer(
                    [
                        partial(
                            atom_type_one_hot,
                            allowable_set=["C", "N", "O", "F", "Si", "P", "S"],
                            encode_unknown=True,
                        ),
                        partial(atom_degree_one_hot, allowable_set=list(range(6))),
                        atom_is_aromatic,
                        atom_formal_charge,
                        atom_num_radical_electrons,
                        partial(atom_hybridization_one_hot, encode_unknown=True),
                        atom_implicit_valence,
                        lambda atom: [0],  # A placeholder for aromatic information,
                        atom_total_num_H_one_hot,
                    ],
                )
            }
        )
        bond_featurizer = BaseBondFeaturizer(
            {"e_feat": ConcatFeaturizer([bond_type_one_hot, bond_is_in_ring])}
        )
        afp_train_graph = [
            mol_to_bigraph(
                mol, node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer
            )
            for mol in tqdm(train_mols)
        ]
        return afp_train_graph

    def __FeaturizerSimple(self, mols) -> list:
        atom_featurizer = BaseAtomFeaturizer(
            {
                "n_feat": ConcatFeaturizer(
                    [
                        # partial(atom_type_one_hot,
                        #        allowable_set=['C', 'N', 'O', 'F', 'Si', 'S'],
                        #        encode_unknown=True),
                        # partial(atom_degree_one_hot, allowable_set=list(range(6))),
                        atom_is_aromatic,
                        atom_formal_charge,
                        atom_num_radical_electrons,
                        partial(atom_hybridization_one_hot, encode_unknown=True),
                        lambda atom: [0],  # A placeholder for aromatic information,
                        atom_total_num_H_one_hot,
                    ],
                )
            }
        )
        bond_featurizer = BaseBondFeaturizer(
            {"e_feat": ConcatFeaturizer([bond_type_one_hot, bond_is_in_ring])}
        )

        train_graph = [
            mol_to_bigraph(
                mol, node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer
            )
            for mol in mols
        ]
        return train_graph

    def __FeaturizerMMFF(self, mols):
        gs = []
        for i, mol in enumerate(tqdm(mols)):
            ret = AllChem.EmbedMolecule(mol, maxAttempts=1000)
            if ret == -1:
                AllChem.EmbedMolecule(mol, useRandomCoords=True)

            g = mol_to_complete_graph(
                mol, node_featurizer=alchemy_nodes, edge_featurizer=alchemy_edges
            )
            gs.append(g)

        return gs


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


# 転移学習の場合
class TransferLearning:
    def __init__(self, in_features=64, out_features=1):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        local_pretrained_path = "pretrained_models/MPNN_Alchemy_pre_trained.pth"

        self.model = MPNNModel(output_dim=12)
        checkpoint = torch.load(local_pretrained_path, map_location=device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.lin2 = nn.Linear(in_features=in_features, out_features=out_features)

    def fit(self, train_mols, train_y):
        # model.to(device)
        # feature_extract = True
        params_to_update = []
        update_param_names = ["lin2.weight", "lin2.bias"]
        for name, param in self.model.named_parameters():
            if name in update_param_names:
                param.requires_grad = True
                params_to_update.append(param)
                print(name)
            else:
                param.requires_grad = False

        # optimizer = optim.RMSprop([{"params": params_to_update, "lr": 1e-4}])


class MPNNRegressor:
    def __init__(
        self, node_hidden_dim=64, edge_hidden_dim=124, output_dim=1, method="Simple"
    ):

        self.model = None
        self.node_input_dim = None
        self.edge_input_dim = None
        self.node_hidden_dim = node_hidden_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.output_dim = output_dim
        self.method = method

        self.verbose = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.args = None
        self.model_init = False

    def __repr__(self):
        return "{0}(node_hidden_dim={1}, edge_hidden_dim={2}, output_dim={3})".format(
            self.__class__.__name__,
            self.node_hidden_dim,
            self.edge_hidden_dim,
            self.output_dim,
        )

    def _preprocess(
        self, mols: list, train_y: numpy.ndarray = None, batch_size: int = 10
    ) -> DataLoader:
        """smilesをgraphに変換し、dataloaderを作成する.

        :param mols:
        :param train_y:
        :return: DataLoader
        """
        if train_y is None:
            train_y = numpy.ones(len(mols))

        trainset = DGLDataset(method=self.method)
        trainset.add(mols, train_y)

        if self.node_input_dim is None:
            self.node_input_dim = trainset[0][1].ndata["n_feat"].shape[1]
            self.edge_input_dim = trainset[0][1].edata["e_feat"].shape[1]

        loader = DataLoader(
            trainset, batch_size=batch_size, collate_fn=collate_molgraphs
        )

        if not self.model_init:
            model = MPNNModel(
                node_input_dim=self.node_input_dim,
                edge_input_dim=self.edge_input_dim,
                node_hidden_dim=self.node_hidden_dim,
                edge_hidden_dim=self.edge_hidden_dim,
                output_dim=self.output_dim,
            )

            self.model_init = True

            if self.device == "cuda":
                self.model = model.cuda()
            else:
                self.model = model

        return loader

    def fit(
        self,
        train_mols: list,
        y_train: numpy.ndarray,
        val_mols=None,
        val_y=None,
        epochs: int = 100,
        batch_size: int = 10,
        lr: float = 0.01,
        patience: int = 100,
        metric="rmse",
    ) -> None:
        """

        :param train_mols:
        :param y_train:
        :param val_mols:
        :param val_y:
        :param epochs:
        :param batch_size:
        :param lr:
        :param patience:
        :param metric: "r2" or "mae" or "rmse" or "roc_auc_score"

        :return:

        """

        train_loader = self._preprocess(train_mols, y_train, batch_size)

        if val_mols:
            val_loader = self._preprocess(train_mols, val_y, batch_size=1)
        else:
            val_loader = None

        self.train(self.model, train_loader, val_loader, epochs, lr, patience, metric)

    def train(self, model, train_loader, val_loader, epochs, lr, patience, metric):
        args = dict(
            patience=patience,
            num_epochs=epochs,
            lr=lr,
            device=self.device,
            metric_name=metric,
        )
        self.args = args
        stopper = EarlyStopping(patience=args["patience"])
        loss_fn = nn.MSELoss(reduction="none")
        optimizer = optim.Adam(model.parameters(), lr=args["lr"])
        for epoch in range(args["num_epochs"]):
            self.run_a_train_epoch(args, epoch, model, train_loader, loss_fn, optimizer)
            # Validation and early stop
            if val_loader is not None:
                val_score = self.run_an_eval_epoch(args, model, val_loader)
                early_stop = stopper.step(val_score, model)
                print(
                    "epoch {:d}/{:d}, validation {} {:.4f}, best validation {} {:.4f}".format(
                        epoch + 1,
                        args["num_epochs"],
                        args["metric_name"],
                        val_score,
                        args["metric_name"],
                        stopper.best_score,
                    )
                )
                if early_stop:
                    self.model = model
                    self.args = args
                    break

        self.model = model
        self.args = args

    def regress(self, args, model, bg):
        h = bg.ndata["n_feat"]
        e = bg.edata["e_feat"]
        h, e = h.to(args["device"]), e.to(args["device"])
        return model(bg, h, e)

    def run_an_eval_epoch(self, args, model, data_loader):
        model.eval()
        eval_meter = Meter()
        with torch.no_grad():
            for batch_id, batch_data in enumerate(data_loader):
                smiles, bg, labels, masks = batch_data
                labels = labels.to(args["device"])
                prediction, _ = self.regress(args, model, bg)
                eval_meter.update(prediction, labels, masks)
            total_score = numpy.mean(eval_meter.compute_metric(args["metric_name"]))
        return total_score

    def run_a_train_epoch(
        self, args, epoch, model, data_loader, loss_criterion, optimizer
    ):
        model.train()
        train_meter = Meter()
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, masks = batch_data
            labels, masks = labels.to(args["device"]), masks.to(args["device"])
            prediction, _ = self.regress(args, model, bg)
            loss = (loss_criterion(prediction, labels) * (masks != 0).float()).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_meter.update(prediction, labels, masks)
        total_score = numpy.mean(train_meter.compute_metric(args["metric_name"]))
        print(
            "epoch {:d}/{:d}, training {} {:.4f}".format(
                epoch + 1, args["num_epochs"], args["metric_name"], total_score
            )
        )

    def predict(self, train_mols, return_fps=True):
        if self.model is None:
            raise

        train_loader = self._preprocess(train_mols, batch_size=1)
        self.args["device"] = "cpu"
        self.model.cpu()
        self.model.eval()
        preds = []
        fps = []

        for batch in tqdm(train_loader):
            _, bg, _, _ = batch
            pred, fp = self.regress(self.args, self.model, bg)
            preds.append(pred.detach().numpy())
            fps.append(fp.detach().numpy())

        preds, fps = numpy.vstack(preds), numpy.vstack(fps)
        if return_fps:
            return preds, fps
        else:
            return preds


class GetMPNNFeat:
    # TO-DO: GPUに対応させる
    def __init__(self, method="Simple"):
        self.method = method
        device = "cuda" if torch.cuda.is_available() else "cpu"
        local_pretrained_path = "pretrained_models/MPNN_Alchemy_pre_trained.pth"

        # self.model = model_zoo.chem.load_pretrained('MPNN_Alchemy')
        self.model = MPNNModel(output_dim=12)
        checkpoint = torch.load(local_pretrained_path, map_location=device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        self.mol_graphs = None

        self.cols = [
            "Dipole Moment",
            "Polarizability",
            "HOMO",
            "LUMO",
            "gap",
            "R2",
            "Zero Point Energy",
            "Internal Energy",
            "Internal Energy(298.15K)",
            "Enthalpy(298.15K)",
            "Free Energy(298.15K)",
            "Heat Capacity(298.15K)",
        ]

    def _preprocess(self, mols):
        dataset = DGLDataset(method=self.method)
        labels = numpy.ones(len(mols))
        dataset.add(mols, labels)
        self.mol_graphs = dataset.graphs

    def calculate(self, mols) -> pandas.DataFrame:
        self._preprocess(mols)
        self.model.eval()
        preds = []
        for graph in tqdm(self.mol_graphs):
            h = graph.ndata.pop("n_feat")
            e = graph.edata.pop("e_feat")
            pred, _ = self.model(graph, h, e)
            preds.append(pred.detach().numpy())

        preds = numpy.vstack(preds)

        res = pandas.DataFrame(preds, columns=self.cols)
        return res


if __name__ == "__main__":
    # Example
    smiles = ["CC", "CCC", "COC", "CCCCC", "CC(=O)C"]
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    train_y = numpy.array([1.0262, 1.4163, 0.2626, 0.8022, 0.5952])  # logP
    clf = MPNNRegressor(method="Simple")
    clf.fit(mols, train_y, epochs=10, batch_size=2, metric="rmse")
    pred, fps = clf.predict(mols)
    mse = (numpy.sum((pred - train_y) ** 2)) / len(mols)
    print(mse)

    mpnn_feat = GetMPNNFeat(method="Simple")
    res = mpnn_feat.calculate(mols)
    print(res)
