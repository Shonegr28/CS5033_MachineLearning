"""
FILE: creates linear regression models and trains them. 

"""

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from data_prep import prepare_data


def train_models():
    models = {}

    # loop through the 1d, 2d, and 3d datasets
    for dimension in (1, 2, 3):

        # loop through low, moderate, and high noise levels
        for noise in ("L", "M", "H"):
            filename = f"{dimension}D {noise}.csv"

            print(f"\n{'=' * 50}")
            print(f"dataset: {filename}")
            print(f"{'=' * 50}")

            # create a list to store the 30 runs for this dataset
            models[filename] = []

            # train 30 separate models using different data splits
            for run in range(30):

                # use a different seed for each train/validation/test split
                seed = 42 + run

                # load and normalize the data for this run
                X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data(
                    filename, seed=seed
                )

                # create a new ordinary least squares regression model
                model = LinearRegression(fit_intercept=True)

                # train the model using only the training data
                model.fit(X_train, y_train)

                # store the model and data for later evaluation
                models[filename].append({
                    "model": model,
                    "scaler": scaler,
                    "X_train": X_train,
                    "X_val": X_val,
                    "X_test": X_test,
                    "y_train": y_train,
                    "y_val": y_val,
                    "y_test": y_test,
                })

                print(f"  --model {run + 1:02d}/30 | seed: {seed} | trained |")

            print(f"\ncompleted: {len(models[filename])} models trained")

    return models


def evaluate_model(model, X, y):
    # predict the target values using the fitted model
    y_pred = model.predict(X)

    # calculate the three regression performance metrics
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)

    # return the metrics for this dataset split
    return {"MAE": mae, "MSE": mse, "RMSE": rmse}


if __name__ == "__main__":

    # train all models for the nine datasets
    models = train_models()

    # count the total number of fitted models
    total = sum(len(runs) for runs in models.values())

    print(f"\n{'=' * 50}")
    print("training summary")
    print(f"{'=' * 50}")
    print(f"datasets: {len(models)}")
    print(f"total fitted models: {total}")
    print(f"{'=' * 50}")
