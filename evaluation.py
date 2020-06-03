from tqdm import tqdm
import pandas as pd
import rdkit.Chem as Chem
from chemutils import *
from nfp import NFPRegressor
from ecfp import ECFP
from collections import namedtuple
import numpy as np

max_val = 1000
train_idx = 800

df_zinc = pd.read_table('./data/train.txt', header=None)
target = pd.read_table('./data/train.logP-SA', header=None)
df_zinc.columns = ["smiles"]
target.columns = ['target']

df = pd.concat([df_zinc.iloc[0:max_val, :], target.iloc[0:max_val, :]], axis=1)
train_smiles, test_smiles = df.smiles[0:train_idx], df.smiles[train_idx:]
train_y, test_y = df.target[0:train_idx], df.target[train_idx:]

mols = [Chem.MolFromSmiles(smi) for smi in tqdm(df.smiles)]
train_mols, test_mols = mols[0:train_idx], mols[train_idx:]

# Neural Fingerprint
model = NFPRegressor(hidden_dim=64, depth=2, nbits=16)
model.fit(train_mols, train_y, epochs=10, verbose=True)
train_pred, train_fps = model.predict(train_mols, return_fps=True)
test_pred, test_fps = model.predict(test_mols, return_fps=True)

# NFP + MLP
rf_evalation(train_pred, test_pred, train_y, test_y)

# NFP + Random Forest
rf_evalation(train_fps, test_fps, train_y, test_y)

# features summed over nodes + Random Forest
feats = np.vstack([ECFP(mol).createNodeFeatures().sum(axis=0) for mol in tqdm(mols)])
rf_evalation(feats[0:train_idx], feats[train_idx:], train_y, test_y)

# Extended Connectivity Fingerprint
args = namedtuple("args", ('rad', 'nbits'))
args = args(rad=2, nbits=2048)

fps1 = np.vstack(list(map(lambda mol: ECFP(mol, args.rad, args.nbits).calculate(), tqdm(mols))))
rf_evalation(fps1[0:train_idx], fps1[train_idx:], train_y, test_y, args)

# rdkit morgan fingerprint
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect

fps2 = np.vstack(list(
    map(lambda mol: GetMorganFingerprintAsBitVect(
        mol, radius=args.rad, nBits=args.nbits, useFeatures=True), tqdm(mols))))

rf_evalation(fps2[0:train_idx], fps2[train_idx:], train_y, test_y, args)


def mapping_nodes():
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.manifold import TSNE

    fps = np.vstack((train_fps, test_fps))
    label = np.hstack([np.zeros(800), np.ones(200)])

    tsne = TSNE(n_components=2).fit_transform(fps)
    tes = np.c_[tsne, df.target.to_numpy(), label]

    gp = GaussianProcessRegressor()
    gp.fit(train_fps, train_y)
    xmin, xmax = min(tes[:, 0]), max(tes[:, 0])
    ymin, ymax = min(tes[:, 1]), max(tes[:, 1])
    zmin, zmax = min(tes[:, 2]), max(tes[:, 2])

    gp = GaussianProcessRegressor()
    gp.fit(tes[:, 0:2], tes[:, 2])

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.cm as cm

    xx, yy = np.meshgrid(np.linspace(xmin - 1, xmax + 1, 200), np.linspace(ymin - 1, ymax + 1, 200))
    xxyy = np.array([xx.ravel(), yy.ravel()]).T
    z1 = gp.predict(xxyy)
    z1 = z1.reshape(-1, 200)
    # plt.scatter(tes[:, 0], tes[:, 1])
    plt.pcolor(xx, yy, z1, alpha=0.5, cmap=cm.jet, vmin=zmin, vmax=zmax)
    plt.colorbar()
    plt.show()
