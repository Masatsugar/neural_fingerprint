import pandas as pd
import rdkit.Chem as Chem
from chemutils import *
from nfp import *
from ecfp import ECFP
from torch.utils.data import DataLoader
from collections import namedtuple
import numpy as np


df_zinc = pd.read_table('./data/train.txt', header=None)
target = pd.read_table('./data/train.logP-SA', header=None)
df_zinc.columns = ["smiles"]
target.columns = ['target']

max_val = 3000
train_idx = 2800

df = pd.concat([df_zinc.iloc[0:max_val, :], target.iloc[0:max_val, :]], axis=1)
train_smiles, test_smiles = df.smiles[0:train_idx], df.smiles[train_idx:]
train_y, test_y = df.target[0:train_idx], df.target[train_idx:]

mols = [Chem.MolFromSmiles(smi) for smi in df.smiles]
graph = mol_to_graph(mols, canonical=False)
train_graph, test_graph = graph[0:train_idx], graph[train_idx:]

trainset = CustomDataset()
testset = CustomDataset()

for smiles, graph, label in zip(train_smiles, train_graph, train_y):
    trainset.add(smiles, graph, label)

for smiles, graph, label in zip(test_smiles, test_graph, test_y):
    testset.add(smiles, graph, label)

train_loader = DataLoader(trainset, batch_size=1, collate_fn=collate_molgraphs)
test_loader = DataLoader(testset, batch_size=1, collate_fn=collate_molgraphs)

# Neural Fingerprint
model = NFP(60, hidden_dim=64, depth=2, nbits=16)
train(model, train_loader, epochs=10)

train_pred, train_fps = eval_pred(model, train_loader)
test_pred, test_fps = eval_pred(model, test_loader)

# NFP + MLP
rf_evalation(train_pred, test_pred, train_y, test_y)

# NFP + Random Forest
rf_evalation(train_fps, test_fps, train_y, test_y)

# features summed over nodes + Random Forest
feats = np.vstack([ECFP(mol).createNodeFeatures().sum(axis=0) for mol in mols])
rf_evalation(feats[0:train_idx], feats[train_idx:], train_y, test_y)

# ECFP
args = namedtuple("args", ('rad', 'nbits'))
args = args(rad=2, nbits=2048)

fps1 = np.vstack(list(map(lambda mol: ECFP(mol, args.rad, args.nbits).calculate(), mols)))
rf_evalation(fps1[0:train_idx], fps1[train_idx:], train_y, test_y, args)

# rdkit morgan fingerprint
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect

fps2 = np.vstack(list(
    map(lambda mol: GetMorganFingerprintAsBitVect(mol, radius=args.rad, nBits=args.nbits, useFeatures=True), mols)))

rf_evalation(fps2[0:train_idx], fps2[train_idx:], train_y, test_y, args)
