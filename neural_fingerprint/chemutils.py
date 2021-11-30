import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


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
        atom.GetTotalNumHs(),
    ]

    return features


def rf_evaluation(train_x, test_x, train_y, test_y, args=None):
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

    print(f"r2={r2}, RMSE={rmse}")
    plt.scatter(test_y, test_pred, label="test")
    plt.scatter(train_y, train_pred, label="train", alpha=0.2)
    plt.legend()
    plt.show()


def _convert_error_columns(df_mordred: pd.DataFrame) -> pd.DataFrame:
    import mordred

    masks = df_mordred.applymap(lambda d: isinstance(d, mordred.error.Missing))
    df_mordred = df_mordred[~masks]
    for c in df_mordred.columns:
        if df_mordred[c].dtypes == object:
            df_mordred = df_mordred.drop(c, axis=1)

    return df_mordred


def calc_mordred_desc(mols: list):

    from mordred import Calculator, descriptors

    calc = Calculator(descriptors, ignore_3D=True)
    res = calc.pandas(mols)
    res = _convert_error_columns(res)
    return res
