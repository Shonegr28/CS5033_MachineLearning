"""
FILE: functions to load in data from csv files to a format that it can be used to train
"""

import csv
import csv
import os
import random


TRAIN_SPLIT = 0.80
VALIDATION_SPLIT = 0.20

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



def split_dataset(X, Y):
    random.seed(1)
    # combine examples with their labels so they stay together when shuffled
    data = []

    for m in range(len(X)):
        data.append([X[m], Y[m]])

    # shuffle examples
    random.shuffle(data)

    # determine where training data ends
    train_size = int(len(data) * TRAIN_SPLIT)

    # split into training and validation data
    train_data = data[:train_size]
    validation_data = data[train_size:]

    X_train = []
    Y_train = []
    X_validation = []
    Y_validation = []

    for example in train_data:
        X_train.append(example[0])
        Y_train.append(example[1])

    for example in validation_data:
        X_validation.append(example[0])
        Y_validation.append(example[1])

    return X_train, Y_train, X_validation, Y_validation


# GAUSSIAN 2D DATASETS:
gaussian_2d_wide_X, gaussian_2d_wide_Y = load_dataset("Gaussian 2D Wide.csv", 2)
gaussian_2d_wide_X_train, gaussian_2d_wide_Y_train, gaussian_2d_wide_X_validation, gaussian_2d_wide_Y_validation = split_dataset(gaussian_2d_wide_X, gaussian_2d_wide_Y)

gaussian_2d_narrow_X, gaussian_2d_narrow_Y = load_dataset("Gaussian 2D Narrow.csv", 2)
gaussian_2d_narrow_X_train, gaussian_2d_narrow_Y_train, gaussian_2d_narrow_X_validation, gaussian_2d_narrow_Y_validation = split_dataset(gaussian_2d_narrow_X, gaussian_2d_narrow_Y)

gaussian_2d_overlap_X, gaussian_2d_overlap_Y = load_dataset("Gaussian 2D Overlap.csv", 2)
gaussian_2d_overlap_X_train, gaussian_2d_overlap_Y_train, gaussian_2d_overlap_X_validation, gaussian_2d_overlap_Y_validation = split_dataset(gaussian_2d_overlap_X, gaussian_2d_overlap_Y)


# GAUSSIAN 3D DATASETS:
gaussian_3d_wide_X, gaussian_3d_wide_Y = load_dataset("Gaussian 3D Wide.csv", 3)
gaussian_3d_wide_X_train, gaussian_3d_wide_Y_train, gaussian_3d_wide_X_validation, gaussian_3d_wide_Y_validation = split_dataset(gaussian_3d_wide_X, gaussian_3d_wide_Y)

gaussian_3d_narrow_X, gaussian_3d_narrow_Y = load_dataset("Gaussian 3D Narrow.csv", 3)
gaussian_3d_narrow_X_train, gaussian_3d_narrow_Y_train, gaussian_3d_narrow_X_validation, gaussian_3d_narrow_Y_validation = split_dataset(gaussian_3d_narrow_X, gaussian_3d_narrow_Y)

gaussian_3d_overlap_X, gaussian_3d_overlap_Y = load_dataset("Gaussian 3D Overlap.csv", 3)
gaussian_3d_overlap_X_train, gaussian_3d_overlap_Y_train, gaussian_3d_overlap_X_validation, gaussian_3d_overlap_Y_validation = split_dataset(gaussian_3d_overlap_X, gaussian_3d_overlap_Y)


# MOONS 2D DATASETS:
moons_2d_wide_X, moons_2d_wide_Y = load_dataset("Moons 2D Wide.csv", 2)
moons_2d_wide_X_train, moons_2d_wide_Y_train, moons_2d_wide_X_validation, moons_2d_wide_Y_validation = split_dataset(moons_2d_wide_X, moons_2d_wide_Y)

moons_2d_narrow_X, moons_2d_narrow_Y = load_dataset("Moons 2D Narrow.csv", 2)
moons_2d_narrow_X_train, moons_2d_narrow_Y_train, moons_2d_narrow_X_validation, moons_2d_narrow_Y_validation = split_dataset(moons_2d_narrow_X, moons_2d_narrow_Y)

moons_2d_overlap_X, moons_2d_overlap_Y = load_dataset("Moons 2D Overlap.csv", 2)
moons_2d_overlap_X_train, moons_2d_overlap_Y_train, moons_2d_overlap_X_validation, moons_2d_overlap_Y_validation = split_dataset(moons_2d_overlap_X, moons_2d_overlap_Y)
