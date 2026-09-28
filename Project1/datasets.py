"""
FILE: functions to load in data from csv files to a format that it can be used to train
"""

import csv
import csv
import os

def load_dataset(file_name, num_features):
    X = []
    Y = []

    class_0 = []
    class_1 = []

    directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(directory, file_name)

    # read data from csv file
    with open(file_path, "r") as file:
        reader = csv.reader(file)

        for row in reader:

            # first num_features columns contain one class-0 example:
            example_0 = []
            for n in range(num_features):
                example_0.append(float(row[n]))

            # last num_features columns contain one class-1 example:
            example_1 = []
            for n in range(num_features, num_features * 2):
                example_1.append(float(row[n]))

            class_0.append(example_0)
            class_1.append(example_1)

    # add all class-0 examples
    for example in class_0:
        X.append(example)
        Y.append(0)

    # add all class-1 examples
    for example in class_1:
        X.append(example)
        Y.append(1)

    return X, Y


# GAUSSIAN 2D DATASETS:
gaussian_2d_wide_X, gaussian_2d_wide_Y = load_dataset(
    "Gaussian 2D Wide.csv", 2
)

gaussian_2d_narrow_X, gaussian_2d_narrow_Y = load_dataset(
    "Gaussian 2D Narrow.csv", 2
)

gaussian_2d_overlap_X, gaussian_2d_overlap_Y = load_dataset(
    "Gaussian 2D Overlap.csv", 2
)


# GAUSSIAN 3D DATASETS:
gaussian_3d_wide_X, gaussian_3d_wide_Y = load_dataset(
    "Gaussian 3D Wide.csv", 3
)

gaussian_3d_narrow_X, gaussian_3d_narrow_Y = load_dataset(
    "Gaussian 3D Narrow.csv", 3
)

gaussian_3d_overlap_X, gaussian_3d_overlap_Y = load_dataset(
    "Gaussian 3D Overlap.csv", 3
)


# MOONS 2D DATASETS:
moons_2d_wide_X, moons_2d_wide_Y = load_dataset(
    "Moons 2D Wide.csv", 2
)

moons_2d_narrow_X, moons_2d_narrow_Y = load_dataset(
    "Moons 2D Narrow.csv", 2
)

moons_2d_overlap_X, moons_2d_overlap_Y = load_dataset(
    "Moons 2D Overlap.csv", 2
)