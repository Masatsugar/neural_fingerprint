import numpy as np
import pandas as pd
import rdkit.Chem as Chem

from neural_fingerprint.chemutils import rf_evaluation
from neural_fingerprint.models.ecfp import ECFP


def test():
    df_zinc = pd.read_table("./data/train.txt", header=None)
    target = pd.read_table("./data/train.logP-SA", header=None)
    df = pd.concat([df_zinc.iloc[0:max_val, :], target[0:max_val]], axis=1)
    train_y, test_y = df.iloc[0:train_idx, 1], df.iloc[train_idx:, 1]
    mols = [Chem.MolFromSmiles(smi) for smi in df.iloc[:, 0]]

    # features summed over nodes + Random Forest
    print("Node Features summed over nodes + Random Forest")
    feats = np.vstack([ECFP(mol).createNodeFeatures().sum(axis=0) for mol in mols])
    rf_evaluation(feats[0:train_idx], feats[train_idx:], train_y, test_y)

    print("ECFP")
    fps1 = np.vstack(
        list(map(lambda mol: ECFP(mol, radius=radius, nbits=nbits).calculate(), mols))
    )
    rf_evaluation(fps1[0:train_idx], fps1[train_idx:], train_y, test_y)

    # rdkit morgan fingerprint
    print("RDKit Morgan fingerprint")
    from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect

    fps2 = np.vstack(
        list(
            map(
                lambda mol: GetMorganFingerprintAsBitVect(
                    mol, radius=radius, nBits=nbits, useFeatures=True
                ),
                mols,
            )
        )
    )
    rf_evaluation(fps2[0:train_idx], fps2[train_idx:], train_y, test_y)


if __name__ == "__main__":
    max_val = 1000
    train_idx = 800
    radius = 4
    nbits = 2048
    test()
