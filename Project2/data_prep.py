
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_DIR = Path(__file__).resolve().parent / "data"


def prepare_data(filename, seed=42):
    # load the csv wo headers and remove rows with missing values
    data = pd.read_csv(DATA_DIR / filename, header=None).dropna()

    # separate input features and target
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # split into 70% training and 30% rest of the data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=seed)

    # split the data that is remaning equally into validation and testing
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=seed)

    # use only training data to learn the mean and standard deviation
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # normalize validation and testing using the same training statistics
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


if __name__ == "__main__":
    import numpy as np

    for dimension in (1, 2, 3):
        for noise in ("L", "M", "H"):
            filename = f"{dimension}D {noise}.csv"

            X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data(
                filename, seed=42
            )

            total = len(y_train) + len(y_val) + len(y_test)

            # check that the splits have all examples
            assert total == len(pd.read_csv(DATA_DIR / filename, header=None).dropna())

            # check that training inputs have mean 0 and std 1
            print(f"training mean: {X_train.mean(axis=0)}")
            print(f"training std: {X_train.std(axis=0)}")
            print(f"scaler mean: {scaler.mean_}")
            print(f"scaler std: {scaler.scale_}")

            assert np.allclose(X_train.mean(axis=0), 0, atol=1e-10)
            assert np.allclose(X_train.std(axis=0), 1, atol=1e-10)


            # check that validation and test inputs use the training scaler as training set
            X_val_original = scaler.inverse_transform(X_val)
            X_test_original = scaler.inverse_transform(X_test)
            assert np.allclose(scaler.transform(X_val_original), X_val)
            assert np.allclose(scaler.transform(X_test_original), X_test)

            # check that the same seed gives the same split
            result = prepare_data(filename, seed=42)
            assert np.array_equal(X_train, result[0])
            assert np.array_equal(y_train, result[3])

            print(f"\n{filename}")
            print(f"training: {len(y_train)}")
            print(f"validation: {len(y_val)}")
            print(f"testing: {len(y_test)}")
            print(f"training mean: {X_train.mean(axis=0)}")
            print(f"training std: {X_train.std(axis=0)}")
            print("all checks passed")