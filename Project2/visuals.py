"""
FILE: each function is used to generate a plots, tables, graphics

"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from linear_regression import train_models, evaluate_model


def metrics_table(runs, filename):
    # The empty dictionary to stores the error scores for all 30 runs of the 3 training, validation, and test.
    results = {
        "Train": {"MAE": [], "MSE": [], "RMSE": []},
        "Validation": {"MAE": [], "MSE": [], "RMSE": []},
        "Test": {"MAE": [], "MSE": [], "RMSE": []},
    }

    # goes through 30 runs and gets the train model
    for run in runs:
        model = run["model"]

        # loops through each of the 3 data
        for split, key in (("Train", "train"), ("Validation", "val"), ("Test", "test")):
            metrics = evaluate_model(model, run[f"X_{key}"], run[f"y_{key}"])

            # collect the scoores and match them to the corresponding split
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
    # use first run with seed 42 for the plot
    run = runs[0]   # run[0] contains the first trained model and its data
    model = run["model"]     # get the fitted linear regression model
    scaler = run["scaler"]  # get the scaler used to normalize the inputs

    # change the scaled X values back to their original values
    X_train = scaler.inverse_transform(run["X_train"]) # training inputs in original units
    X_val = scaler.inverse_transform(run["X_val"]) # validation inputs in original units
    X_test = scaler.inverse_transform(run["X_test"]) # test inputs in original units

    # get the actual y values for training, validation and test data
    y_train = run["y_train"]
    y_val = run["y_val"]
    y_test = run["y_test"]

    # make x values from the smallest to largest input so we can draw the line
    X_all = np.concatenate((X_train, X_val, X_test))
    x_line = np.linspace(X_all.min(), X_all.max(), 200).reshape(-1, 1)

    # scale the x line using the same scaler and predict y values
    x_line_scaled = scaler.transform(x_line)  # scale the new x values before prediction
    y_line = model.predict(x_line_scaled)   # predicted y values for the regression line

    # show train, validation and test points separately on the plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(X_train[:, 0], y_train, label="Train", alpha=0.7)
    ax.scatter(X_val[:, 0], y_val, label="Validation", alpha=0.7)
    ax.scatter(X_test[:, 0], y_test, label="Test", alpha=0.7)

    # scale the x line using the same scaler and predict y values
    ax.plot(x_line[:, 0], y_line, color="black", linewidth=2, 
            label="Regression Line")
    
    # draw the regression line from the model that was trained on training data
    ax.set_title(f"{filename} - 1D Linear Regression (Run 1)")
    ax.set_xlabel("X1")
    ax.set_ylabel("Target (y)")
    ax.legend()
    ax.grid(alpha=0.3)

    # save the plot in the 1D folder
    plots_dir = Path(__file__).resolve().parent / "plots" / "1D"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_name = filename.replace(".csv", "").replace(" ", "_")
    fig.savefig(plots_dir / f"{plot_name}_scatter.png",
                dpi=300, bbox_inches="tight")

    plt.close(fig)



def scatter_plot_2d(runs, filename):
    # use the first model trained with seed 42
    run = runs[0] # runs[0] contains the first trained model and its data
    model = run["model"] # get the fitted linear regression model
    scaler = run["scaler"] # get the scaler used on input features

    # change normalized inputs back to the original scale
    X_train = scaler.inverse_transform(run["X_train"]) 
    X_val = scaler.inverse_transform(run["X_val"])
    X_test = scaler.inverse_transform(run["X_test"])

    # combine all X values to get the minimum and maximum of both features
    X_all = np.vstack((X_train, X_val, X_test))

    # make x1 and x2 grid points to draw the predicted regression plane
    x1 = np.linspace(X_all[:, 0].min(), X_all[:, 0].max(), 30)
    x2 = np.linspace(X_all[:, 1].min(), X_all[:, 1].max(), 30)
    X1, X2 = np.meshgrid(x1, x2)

    # scale every grid point then predict y for the plane
    grid = np.column_stack((X1.ravel(), X2.ravel()))
    y_plane = model.predict(scaler.transform(grid)).reshape(X1.shape)

    # create a 3D graph because there are two X features and one y value
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # plot the actual train, validation and test examples
    ax.scatter(X_train[:, 0], X_train[:, 1], run["y_train"],
               label="Train", alpha=0.8)
    ax.scatter(X_val[:, 0], X_val[:, 1], run["y_val"],
               label="Validation", alpha=0.8)
    ax.scatter(X_test[:, 0], X_test[:, 1], run["y_test"],
               label="Test", alpha=0.8)

    # draw the predicted regression plane using the trained model
    ax.plot_surface(X1, X2, y_plane, color="gray", alpha=0.3)

    ax.set_title(f"{filename} - 2D Linear Regression (Run 1)")
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("Target (y)")
    ax.legend()

    # save the plot 2d folder
    plots_dir = Path(__file__).resolve().parent / "plots" / "2D"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_name = filename.replace(".csv", "").replace(" ", "_")
    fig.savefig(plots_dir / f"{plot_name}_scatter.png",
                dpi=300, bbox_inches="tight")

    plt.close(fig)


def comparison_bar_charts(models):
    # collect test MAE, MSE and RMSE from all 30 runs of each dataset
    results = {}  # store metric values for each dataset


    for filename, runs in models.items():
        results[filename] = {"MAE": [], "MSE": [], "RMSE": []}

        for run in runs:
            metrics = evaluate_model(
                run["model"], run["X_test"], run["y_test"]
            )

            for metric in ("MAE", "MSE", "RMSE"):
                results[filename][metric].append(metrics[metric])

    # make a folder to save the comparison charts
    plots_dir = Path(__file__).resolve().parent / "plots" / "comparisons"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # make two comparisons for each metric, one by noise and one by dimension
    for comparison in ("noise", "dimension"):
        for metric in ("MAE", "MSE", "RMSE"):

            if comparison == "noise":
                groups = ("L", "M", "H")
                labels = ("Low", "Moderate", "High")
                categories = ("1D", "2D", "3D")
                xlabel = "Noise Level"
            else:
                groups = ("1D", "2D", "3D")
                labels = groups
                categories = ("L", "M", "H")
                xlabel = "Dimension"

            fig, ax = plt.subplots(figsize=(8, 5))
            x = np.arange(3)
            width = 0.25

            # plot the mean metric values and use standard deviation for the error bars
            for i, category in enumerate(categories):
                means = []
                stds = []

                for group in groups:
                    if comparison == "noise":
                        filename = f"{category} {group}.csv"
                    else:
                        filename = f"{group} {category}.csv"

                    values = results[filename][metric]
                    means.append(np.mean(values))
                    stds.append(np.std(values, ddof=1))

                if comparison == "noise":
                    bar_label = category
                else:
                    bar_label = {"L": "Low", "M": "Moderate", "H": "High"}[category]

                ax.bar(x + (i - 1) * width, means, width,
                       yerr=stds, capsize=4, label=bar_label)

            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(f"Mean Test {metric}")
            ax.set_title(f"{metric} - {comparison.capitalize()} Comparison (30 Runs)")
            ax.set_ylim(bottom=0)
            ax.legend()
            ax.grid(axis="y", alpha=0.3) # positions for the three bar groups
            ax.set_axisbelow(True)

            fig.tight_layout()
            fig.savefig(plots_dir / f"{comparison}_{metric.lower()}.png",
                        dpi=300, bbox_inches="tight")
            plt.close(fig)


def predicted_vs_actual(runs, filename):
    # use the first fitted model (seed 42)
    run = runs[0] # run[0] contains the first trained model and data for that
    model = run["model"]    # get the fitted linear regression model

    # predicts the y values for validation and test sets
    y_val = run["y_val"]
    y_test = run["y_test"]
    pred_val = model.predict(run["X_val"])
    pred_test = model.predict(run["X_test"])

    # find the lowest and highest actual or predicted y for the reference line
    low = min(y_val.min(), y_test.min(), pred_val.min(), pred_test.min())
    high = max(y_val.max(), y_test.max(), pred_val.max(), pred_test.max())

    # show actual y against predicted y for validation and test data
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_val, pred_val, alpha=0.7, label="Validation Predictions")
    ax.scatter(y_test, pred_test, alpha=0.7, label="Test Predictions")

    # points on this line have the same actual and predicted y values
    ax.plot([low, high], [low, high], "k--", label="Perfect Prediction")

    ax.set_title(f"{filename} - Predicted vs. Actual (Run 1)")
    ax.set_xlabel("Actual y")
    ax.set_ylabel("Predicted y")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    ax.grid(alpha=0.3)

    # save the plot in  1D, 2D or 3D folder
    dimension = filename.split("D")[0] + "D"
    plots_dir = Path(__file__).resolve().parent / "plots" / dimension
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_name = filename.replace(".csv", "").replace(" ", "_")
    fig.savefig(plots_dir / f"{plot_name}_predicted_vs_actual.png",
                dpi=300, bbox_inches="tight")

    plt.close(fig)



def residual_plot(runs, filename):
    # use the first fitted model (seed 42)
    run = runs[0] # does the same thing as the other function
    model = run["model"] # does the same thing as the other function

    # use the model to get predictions for validation and test data
    pred_val = model.predict(run["X_val"])
    pred_test = model.predict(run["X_test"])

    # residual is actual y minus predicted y
    residuals_val = run["y_val"] - pred_val
    residuals_test = run["y_test"] - pred_test

    # plot predicted y against residuals for validation and test
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(pred_val, residuals_val, alpha=0.7, label="Validation Residuals")
    ax.scatter(pred_test, residuals_test, alpha=0.7, label="Test Residuals")

    # zero line shows where there is no prediction error
    ax.axhline(0, color="black", linestyle="--", label="Zero Residual")

    ax.set_title(f"{filename} - Residual Plot (Run 1)")
    ax.set_xlabel("Predicted y")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.legend()
    ax.grid(alpha=0.3)

    # save the plot in its dimension folder
    dimension = filename.split("D")[0] + "D"
    plots_dir = Path(__file__).resolve().parent / "plots" / dimension
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_name = filename.replace(".csv", "").replace(" ", "_")
    fig.savefig(plots_dir / f"{plot_name}_residuals.png",
                dpi=300, bbox_inches="tight")

    plt.close(fig)



def print_regression_equation(runs, filename):
    # does the same thing as the other functions, using the first fitted model
    run = runs[0]
    model = run["model"]
    scaler = run["scaler"]

    # divide each model coefficient by its feature scaling value
    coefficients = model.coef_ / scaler.scale_

    # change the intercept so the equation works with original X values
    intercept = model.intercept_ - np.sum(coefficients * scaler.mean_)  # coefficients in original feature units

    # start the equation with the intercept then add each X coefficient
    equation = f"y = {intercept:.6f}"

    for i, coefficient in enumerate(coefficients):
        equation += f" {coefficient:+.6f}*x{i + 1}"

    print(f"\nDataset: {filename}")
    print(f"Run 1 (seed 42): {equation}")




if __name__ == "__main__":
    # train 30 runs for every dataset
    models = train_models()

    # make a metrics table for each dataset using its 30 runs
    for filename, runs in models.items():
        print(f"---------- Creating metrics table: {filename} ----------")
        metrics_table(runs, filename)

    # 1d scatter plots for the three 1d datasets
    for filename, runs in models.items():
        if filename.startswith("1D"):
            print(f"---------- Creating scatter plot: {filename} ----------")
            scatter_plot_1d(runs, filename)

    # 2d scatter plots for the three 2d datasets
    for filename, runs in models.items():
        if filename.startswith("2D"):
            print(f"---------- Creating 2d scatter plot: {filename} ----------")
            scatter_plot_2d(runs, filename)


    
    # predicted vs actual plots for all 9 datasets
    for filename, runs in models.items():
        print(f"---------- Creating predicted vs actual plot: {filename} ----------")
        predicted_vs_actual(runs, filename)

    # comparison bar charts for all datasets
    print("---------- Creating comparison bar charts ----------")
    comparison_bar_charts(models)



    # all residual plots for all 9 datasets
    for filename, runs in models.items():
        print(f"---------- Creating residual plot: {filename} ----------")
        residual_plot(runs, filename)


    # all 9 datasets regression equations
    for filename, runs in models.items():
        print_regression_equation(runs, filename)

    print("\n---------- all plots saved to plots/ ----------")
