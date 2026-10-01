import matplotlib.pyplot as plt
import numpy as np

import os
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

"""
FILE: calculates and plots results from the neural network experiments
"""


def average_results(results):

    # stores averages
    average_training_accuracy = 0.0
    average_validation_accuracy = 0.0
    average_final_cost = 0.0

    # add metrics from every repetition
    for result in results:
        average_training_accuracy += result[0]
        average_validation_accuracy += result[1]
        average_final_cost += result[2]

    # divide by number of repetitions
    average_training_accuracy /= len(results)
    average_validation_accuracy /= len(results)
    average_final_cost /= len(results)


    # number of epochs stored in cost history
    num_epochs = len(results[0][3])

    # initialize average cost for each epoch
    average_costs = []

    for epoch in range(num_epochs):

        total_cost = 0.0

        # add cost at this epoch from every repetition
        for result in results:
            total_cost += result[3][epoch]

        # average cost at this epoch
        average_cost = total_cost / len(results)

        average_costs.append(average_cost)


    return (
        average_training_accuracy,
        average_validation_accuracy,
        average_final_cost,
        average_costs
    )


def print_results(dataset_names, all_results):

    print("\n\n========================================")
    print("AVERAGE RESULTS")
    print("========================================")

    for i in range(len(dataset_names)):

        averages = average_results(all_results[i])

        print("\nDataset:", dataset_names[i])
        print("Average Training Accuracy:", averages[0])
        print("Average Validation Accuracy:", averages[1])
        print("Average Final Cost:", averages[2])


def plot_validation_accuracy_comparison(dataset_names, simple_results, complex_results, file_name):
    simple_accuracies = []
    complex_accuracies = []

    for i in range(len(dataset_names)):
        simple_accuracies.append(average_results(simple_results[i])[1])
        complex_accuracies.append(average_results(complex_results[i])[1])

    x = np.arange(len(dataset_names))
    width = 0.35

    plt.figure(figsize=(14, 7))
    plt.bar(x - width / 2, simple_accuracies, width, label="Simple Network")
    plt.bar(x + width / 2, complex_accuracies, width, label="Complex Network")

    plt.xlabel("Dataset")
    plt.ylabel("Average Validation Accuracy")
    plt.title("Simple vs Complex Network - Average Validation Accuracy")
    plt.xticks(x, dataset_names, rotation=30, ha="right")
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(DIRECTORY, file_name))
    plt.close()


def plot_average_cost(dataset_names, all_results, network_name, file_name):
    plt.figure(figsize=(12, 7))

    for i in range(len(dataset_names)):
        average_costs = average_results(all_results[i])[3]

        # plot every 5th epoch to make rapid oscillations readable
        step = 5
        plt.plot(range(0, len(average_costs), step), average_costs[::step], label=dataset_names[i])

    plt.xlabel("Epoch")
    plt.ylabel("Average Cost")
    plt.title(network_name + " - Average Cost for All Datasets")
    plt.ylim(0, 1.0)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(DIRECTORY, file_name))
    plt.close()


def plot_network_cost_comparison(dataset_name, simple_results, complex_results, file_name):
    simple_costs = average_results(simple_results)[3]
    complex_costs = average_results(complex_results)[3]

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(simple_costs)), simple_costs, label="Simple Network")
    plt.plot(range(len(complex_costs)), complex_costs, label="Complex Network")

    # determine zoom from costs after first 10 epochs
    zoom_costs = simple_costs[10:] + complex_costs[10:]
    y_min = min(zoom_costs)
    y_max = max(zoom_costs)
    padding = (y_max - y_min) * 0.10

    plt.ylim(max(0, y_min - padding), y_max + padding)

    plt.xlabel("Epoch")
    plt.ylabel("Average Cost")
    plt.title(dataset_name + " - Simple vs Complex Network Cost")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(DIRECTORY, file_name))
    plt.close()



import csv


def save_results_csv(dataset_names, all_results, file_name):
    file_path = os.path.join(DIRECTORY, file_name)

    with open(file_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Dataset", "Repetition", "Epoch", "Cost", "Training Accuracy", "Validation Accuracy"])

        for i in range(len(dataset_names)):
            for repetition in range(len(all_results[i])):
                result = all_results[i][repetition]
                training_accuracy = result[0]
                validation_accuracy = result[1]
                costs = result[3]

                for epoch in range(len(costs)):
                    writer.writerow([
                        dataset_names[i], repetition + 1, epoch, costs[epoch],
                        training_accuracy, validation_accuracy
                    ])






def plot_decision_boundary_2d(network, X, Y, dataset_name, file_name):

    # verify that plotted network matches plotted data
    data_predictions = network.predict(X)

    correct = 0
    for m in range(len(X)):
        if data_predictions[m] == Y[m]:
            correct += 1

    # print(dataset_name, "PLOTTED NETWORK ACCURACY:", correct / len(X))

    # get area around all the data so we know where to test the network
    x1_min = min(example[0] for example in X) - 1
    x1_max = max(example[0] for example in X) + 1
    x2_min = min(example[1] for example in X) - 1
    x2_max = max(example[1] for example in X) + 1

    # fill the area with points, then predict every point
    grid = []
    x1 = x1_min
    while x1 <= x1_max:
        x2 = x2_min
        while x2 <= x2_max:
            grid.append([x1, x2])
            x2 += 0.1
        x1 += 0.1

    predictions = network.predict(grid)

    # separate the points by what the network thinks they are
    predicted_0_x1 = []
    predicted_0_x2 = []
    predicted_1_x1 = []
    predicted_1_x2 = []

    for m in range(len(grid)):
        if predictions[m] == 0:
            predicted_0_x1.append(grid[m][0])
            predicted_0_x2.append(grid[m][1])
        else:
            predicted_1_x1.append(grid[m][0])
            predicted_1_x2.append(grid[m][1])

    # these two areas meet at the decision boundary
    plt.scatter(predicted_0_x1, predicted_0_x2, alpha=0.1)
    plt.scatter(predicted_1_x1, predicted_1_x2, alpha=0.1)

    # now put the real examples over the prediction areas
    class_0_x1 = []
    class_0_x2 = []
    class_1_x1 = []
    class_1_x2 = []

    for m in range(len(X)):
        if Y[m] == 0:
            class_0_x1.append(X[m][0])
            class_0_x2.append(X[m][1])
        else:
            class_1_x1.append(X[m][0])
            class_1_x2.append(X[m][1])

    plt.scatter(class_0_x1, class_0_x2, label="Class 0")
    plt.scatter(class_1_x1, class_1_x2, label="Class 1")

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(dataset_name + " Decision Boundary")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(DIRECTORY, file_name))
    plt.close()


def plot_decision_boundary_3d(network, X, Y, dataset_name, file_name):

    # get area around the data so we can test points through the 3D space
    x1_min = min(example[0] for example in X) - 1
    x1_max = max(example[0] for example in X) + 1
    x2_min = min(example[1] for example in X) - 1
    x2_max = max(example[1] for example in X) + 1
    x3_min = min(example[2] for example in X) - 1
    x3_max = max(example[2] for example in X) + 1

    # for each x1,x2 location make a line of points going through x3
    # when prediction changes from 0 to 1, that is where the boundary is
    boundary = []

    x1 = x1_min
    while x1 <= x1_max:
        x2 = x2_min

        while x2 <= x2_max:
            line = []

            x3 = x3_min
            while x3 <= x3_max:
                line.append([x1, x2, x3])
                x3 += 0.2

            predictions = network.predict(line)

            for m in range(1, len(line)):
                if predictions[m] != predictions[m - 1]:
                    boundary.append(line[m])
                    break

            x2 += 0.2
        x1 += 0.2

    # split boundary coordinates so matplotlib can plot them
    boundary_x1 = []
    boundary_x2 = []
    boundary_x3 = []

    for point in boundary:
        boundary_x1.append(point[0])
        boundary_x2.append(point[1])
        boundary_x3.append(point[2])

    # split the real data into the two classes
    class_0_x1 = []
    class_0_x2 = []
    class_0_x3 = []
    class_1_x1 = []
    class_1_x2 = []
    class_1_x3 = []

    for m in range(len(X)):
        if Y[m] == 0:
            class_0_x1.append(X[m][0])
            class_0_x2.append(X[m][1])
            class_0_x3.append(X[m][2])
        else:
            class_1_x1.append(X[m][0])
            class_1_x2.append(X[m][1])
            class_1_x3.append(X[m][2])

    # plot the boundary we found and put the real examples on top
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(boundary_x1, boundary_x2, boundary_x3, s=8, alpha=0.5, label="Decision Boundary")
    ax.scatter(class_0_x1, class_0_x2, class_0_x3, marker="o", label="Class 0")
    ax.scatter(class_1_x1, class_1_x2, class_1_x3, marker="x", label="Class 1")

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")
    ax.set_title(dataset_name + " Decision Boundary")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(DIRECTORY, file_name))
    plt.close()


import sys

def save_terminal_output(file_name):
    sys.stdout = open(file_name, "w")