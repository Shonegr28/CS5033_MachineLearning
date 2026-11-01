"""
FILE: each function is used to generate a plots, tables, graphics

"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from linear_regression import train_models, evaluate_model


def metrics_table(runs, filename):
    # store the metrics from all 30 runs for each data split
    results = {
        "Train": {"MAE": [], "MSE": [], "RMSE": []},
        "Validation": {"MAE": [], "MSE": [], "RMSE": []},
        "Test": {"MAE": [], "MSE": [], "RMSE": []},
    }

    # evaluate each fitted model on its own data splits
    for run in runs:
        model = run["model"]

        for split, key in (("Train", "train"), ("Validation", "val"), ("Test", "test")):
            metrics = evaluate_model(model, run[f"X_{key}"], run[f"y_{key}"])

            # collect the three metrics for this run
            for metric in ("MAE", "MSE", "RMSE"):
                results[split][metric].append(metrics[metric])

    # calculate the mean and sample standard deviation across 30 runs
    rows = []
    for split in ("Train", "Validation", "Test"):
        row = [split]

        for metric in ("MAE", "MSE", "RMSE"):
            values = results[split][metric]
            mean = np.mean(values)
            std = np.std(values, ddof=1)
            row.append(f"{mean:.4f} ± {std:.4f}")

        rows.append(row)

    # create a table graphic for the dataset
    fig, ax = plt.subplots(figsize=(9, 2.5))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=["Split", "MAE", "MSE", "RMSE"],
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    ax.set_title(f"{filename} - Performance Metrics (30 Runs)", fontsize=13, pad=15)
    fig.tight_layout()

    # create the plots folder if it does not exist
    # save the table in the folder for its dimension
    dimension = filename.split("D")[0] + "D"
    plots_dir = Path(__file__).resolve().parent / "plots" / dimension
    plots_dir.mkdir(parents=True, exist_ok=True)

    # save the table with the dataset name
    plot_name = filename.replace(".csv", "").replace(" ", "_")
    fig.savefig(plots_dir / f"{plot_name}_metrics.png", dpi=300, bbox_inches="tight")

    plt.close(fig)

    return fig



def scatter_plot_1d(runs, filename):
    # use the first trained model (seed 42) for the plot
    run = runs[0]
    model = run["model"]
    scaler = run["scaler"]

    # convert the normalized inputs back to their original values
    X_train = scaler.inverse_transform(run["X_train"])
    X_val = scaler.inverse_transform(run["X_val"])
    X_test = scaler.inverse_transform(run["X_test"])

    # get the original target values for each split
    y_train = run["y_train"]
    y_val = run["y_val"]
    y_test = run["y_test"]

    # create input values across the full dataset range
    X_all = np.concatenate((X_train, X_val, X_test))
    x_line = np.linspace(X_all.min(), X_all.max(), 200).reshape(-1, 1)

    # normalize the line inputs before making predictions
    x_line_scaled = scaler.transform(x_line)
    y_line = model.predict(x_line_scaled)

    # plot the observations from each data split
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(X_train[:, 0], y_train, label="Train", alpha=0.7)
    ax.scatter(X_val[:, 0], y_val, label="Validation", alpha=0.7)
    ax.scatter(X_test[:, 0], y_test, label="Test", alpha=0.7)

    # plot the regression line fitted using training data only
    ax.plot(x_line[:, 0], y_line, color="black", linewidth=2,
            label="Regression Line")

    ax.set_title(f"{filename} - 1D Linear Regression (Run 1)")
    ax.set_xlabel("X1")
    ax.set_ylabel("Target (y)")
    ax.legend()
    ax.grid(alpha=0.3)

    # save the scatter plot in the plots folder
    # save the scatter plot in the 1d folder
    plots_dir = Path(__file__).resolve().parent / "plots" / "1D"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_name = filename.replace(".csv", "").replace(" ", "_")
    fig.savefig(plots_dir / f"{plot_name}_scatter.png",
                dpi=300, bbox_inches="tight")

    plt.close(fig)



def scatter_plot_2d(runs, filename):
    # use the first fitted model (seed 42)
    run = runs[0]
    model = run["model"]
    scaler = run["scaler"]

    # convert the inputs back to their original scale
    X_train = scaler.inverse_transform(run["X_train"])
    X_val = scaler.inverse_transform(run["X_val"])
    X_test = scaler.inverse_transform(run["X_test"])

    # combine inputs to find the full range for the plane
    X_all = np.vstack((X_train, X_val, X_test))

    # create a grid of x1 and x2 values for the regression plane
    x1 = np.linspace(X_all[:, 0].min(), X_all[:, 0].max(), 30)
    x2 = np.linspace(X_all[:, 1].min(), X_all[:, 1].max(), 30)
    X1, X2 = np.meshgrid(x1, x2)

    # normalize the grid inputs and predict the plane values
    grid = np.column_stack((X1.ravel(), X2.ravel()))
    y_plane = model.predict(scaler.transform(grid)).reshape(X1.shape)

    # create a 3d plot of the observed data
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # plot the train, validation, and test observations
    ax.scatter(X_train[:, 0], X_train[:, 1], run["y_train"],
               label="Train", alpha=0.8)
    ax.scatter(X_val[:, 0], X_val[:, 1], run["y_val"],
               label="Validation", alpha=0.8)
    ax.scatter(X_test[:, 0], X_test[:, 1], run["y_test"],
               label="Test", alpha=0.8)

    # plot the plane fitted using only the training observations
    ax.plot_surface(X1, X2, y_plane, color="gray", alpha=0.3)

    ax.set_title(f"{filename} - 2D Linear Regression (Run 1)")
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("Target (y)")
    ax.legend()

    # save the plot in the 2d folder
    plots_dir = Path(__file__).resolve().parent / "plots" / "2D"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_name = filename.replace(".csv", "").replace(" ", "_")
    fig.savefig(plots_dir / f"{plot_name}_scatter.png",
                dpi=300, bbox_inches="tight")

    plt.close(fig)




if __name__ == "__main__":
    # train all 30 models for each dataset
    models = train_models()

    # create and save a metrics table for each dataset
    for filename, runs in models.items():
        print(f"---------- Creating metrics table: {filename} ----------")
        metrics_table(runs, filename)

    # create scatter plots only for the three 1d datasets
    for filename, runs in models.items():
        if filename.startswith("1D"):
            print(f"---------- Creating scatter plot: {filename} ----------")
            scatter_plot_1d(runs, filename)

    for filename, runs in models.items():
        if filename.startswith("2D"):
            print(f"---------- Creating 2d scatter plot: {filename} ----------")
            scatter_plot_2d(runs, filename)
            
    print("\n---------- all plots saved to plots/ ----------")
