
import math
import random
from datasets import *
from results import *
import results
import os
"""
FILE: implements a neural network from scratch for this project
"""

"""
WHAT: this class uses lists, arrays, and looping to implement a neural net.
"""
class FeedForwardNeuralNetwork:

    def __init__(self, X, Y, dimensions, learning_rate=0.01, num_iterations=1000):
        # TRAINING DATA
        self.X = X              # X[m] = entire input feature vector for example-m, X[m][n] = input to node-n or feature-n for example-m
        self.Y = Y              # Y[m] = output label for example-m, single integer either 0 or 1 since we are only doing binary classification
        self.m = len(self.X)

        # ARCHITECTURE
        self.dimensions = dimensions        # list where each element is the number of nodes in that layer
        self.num_layers = len(dimensions)   # number of layers including input and output layers

        # HYPERPARAMTERS
        self.learning_rate = learning_rate     # the step size at which optimization algorithm moves toward the minimum of cost function
        self.num_iterations = num_iterations   # the number of epochs the network is trained on.

        # PARAMETERS
        self.W = []     # W[l][prev][n] = in layer-l the weight connecting node-prev in layer l-1 to node-n in layer l.
        self.b = []     # b[l][n] = bias-parameter for node-n in layer-l

        # FORWARD PROPAGATION
        self.Z = []             # Z[l][m][n] = weighted-sum for node-n in layer-l for example-m
        self.A = []             # A[l][m][n] = activation for node-n in layer-l for example-m

        # BACKPROPAGATION
        self.dW = []          # same representation as W, dW[l][prev][n] = gradient of cost with respect to weight W[l][prev][n]
        self.db = []          # same representation as b, db[l][n] = gradient of cost with respect to bias b[l][n]
        self.dZ = []          # same representation as Z, dZ[l][m][n] = gradient of cost with respect to weighted-sum Z[l][m][n]

        # RESULTS
        self.cost = 0       # current cost 
        self.costs = []     # stores cost values during training for each iteration


    def initialize_parameters(self):
        self.W, self.b = [], []
        # iterate all layers and add a empty-list in each paramter to represent that layer
        for l in range(self.num_layers):
            self.W.append([])
            self.b.append([])

        # iterate for every layer except input layer to init weights becaues input layer does not have weights to it
        for l in range (1, self.num_layers):
            # iterate all nodes in preivous layer l-1, and add a empty list for it
            for prev in range(self.dimensions[l-1]):
                self.W[l].append([])
                # for every node in previous layer l-1, init a random weight for every node in cur-layer-l
                for n in range(self.dimensions[l]):
                    weight_init = np.random.randn() * np.sqrt(2.0 / self.dimensions[l-1])
                    self.W[l][prev].append(weight_init)

            # init one bias-param for every node in cur-layer
            for n in range(self.dimensions[l]):
                self.b[l].append(0.0)
        

    def initialize_calculations(self):
        self.Z = []
        self.A = []
        # create Z and A for every layer
        for l in range(self.num_layers):
            self.Z.append([])
            self.A.append([])
            # add an empty list for every layer
            for m in range(self.m):
                self.Z[l].append([])
                self.A[l].append([])

                # add a placeholder 0 for node in cur-layer for cur-example
                for n in range(self.dimensions[l]):
                    self.Z[l][m].append(0.0)
                    self.A[l][m].append(0.0)

        # input layer activations are the input features
        for m in range(self.m):
            for n in range(self.dimensions[0]):
                self.A[0][m][n] = self.X[m][n]

    def initialize_gradients(self):
        self.dW = []
        self.db = []
        self.dZ = []

        # itearte all layers and add a empty list to each gradient type
        for l in range(self.num_layers):
            self.dW.append([])
            self.db.append([])
            self.dZ.append([])

        # iterate all layeres except input layer because it doesnt have griadnets
        for l in range(1, self.num_layers):

            # iterate all nodes in preivous layer l-1 and add a empty list to it
            for prev in range(self.dimensions[l - 1]):
                self.dW[l].append([])
                # iterate all nodes in cur-layer-l and add a init 0.0 value to the preivous layer-list
                for n in range(self.dimensions[l]):
                    self.dW[l][prev].append(0.0)

            # Initialize bias gradients:
            for n in range(self.dimensions[l]):
                self.db[l].append(0.0)

        # iterate all layers
        for l in range(self.num_layers):
            # iterate all examples
            for m in range(self.m):
                # for cur-layer, for each each example add a list
                self.dZ[l].append([])
                # for eveyr node in cur-layer-l, for cur-example add a init value 0.0
                for n in range(self.dimensions[l]):
                    self.dZ[l][m].append(0.0)


    # sigmoid activation function takes in weighted-sum
    def sigmoid(self, Z):
        return 1 / (1 + math.exp(-Z))

    # relu activation function takes in weighted-sum
    def relu(self, Z):
        return max(0, Z)

    # derivative of sigmoid
    def sigmoid_backward(self, Z):
            sigmoid_Z = self.sigmoid(Z)
            return sigmoid_Z * (1 - sigmoid_Z)

    # derivative of relu
    def relu_backward(self, Z):
        if Z > 0:
            return 1
        else:
            return 0


    def forward_propagation(self):
        # set activations of input layer to be the input data
        for m in range(self.m):
            for n in range(self.dimensions[0]):
                self.A[0][m][n] = self.X[m][n]

        # iterate all layers except input-layer
        for l in range(1, self.num_layers):
            # for cur-layer iterate all examples
            for m in range(self.m):
                # for cur-layer cur-example iterate all nodes in cur-layer
                for n in range(self.dimensions[l]):
                    # reset weighted-sum for cur-layer-cur-example-cur-node
                    self.Z[l][m][n] = 0.0

                    # iterate all nodes in preivous-layer l-1 to compute weighed-sum for cur-layer-cur-example-cur-node
                    for prev in range(self.dimensions[l - 1]):
                        self.Z[l][m][n] += self.W[l][prev][n] * self.A[l - 1][m][prev]  # z = w*a[l-1]

                    # add bias param to cur-weighed-sum
                    self.Z[l][m][n] += self.b[l][n]

                    # if cur-layer is a hidden-layer, use relu activation
                    if l < self.num_layers - 1:
                        self.A[l][m][n] = self.relu(self.Z[l][m][n])
                    # if cur-layer is output-layer, use sigmoid activation
                    elif l == len(self.dimensions)-1:
                        self.A[l][m][n] = self.sigmoid(self.Z[l][m][n])

    def compute_cost(self):
        epsilon = 1e-15
        # stores cur cost
        self.cost = 0
        # iterate every example and compute binary cross entropy equation
        for m in range(self.m):
            Y = self.Y[m]   # get label for cur-example
            A = self.A[self.num_layers - 1][m][0]   # get activations of output-layer cur-example, and the only node we have since binary classification
            A = max(epsilon, min(1 - epsilon, A))   # prevent some log(0) math domain error
            # y * log(a) + (1-y)log(1-p)
            self.cost += -(Y * math.log(A) + (1 - Y) * math.log(1 - A))

        # average cost of each example
        self.cost /= self.m
        return self.cost

    def backward_propagation(self):
        # output-layer-indx
        output_l_indx = self.num_layers - 1

        # iterate all examples, and compute dZ for output lauer
        for m in range(self.m):
            # backprop equation for sigmoid output + binary cross-entropy:
            # dZ = A - Y
            # gradient of Z for output-layer for cur-example for one node = activation of output-layer for cur-example for one node - output-label for cur-example
            self.dZ[output_l_indx][m][0] = self.A[output_l_indx][m][0] - self.Y[m]


        # iterate from output-layer and backwards
        for l in range(output_l_indx, 0, -1):
            # reset gradients for cur-layer
            for prev in range(self.dimensions[l - 1]):
                for n in range(self.dimensions[l]):
                    self.dW[l][prev][n] = 0.0

            for n in range(self.dimensions[l]):
                self.db[l][n] = 0.0


            # compute weight gradients by iterate nodes in preivous layer
            for prev in range(self.dimensions[l - 1]):
                # iterate nodes in cur-lauer
                for n in range(self.dimensions[l]):
                    # iterate all examples
                    for m in range(self.m):

                        # Backprop equation: dW = (1/m) * sum(A_previous * dZ)
                        # gradient of W for cur-layer-node-n-cur_prev-node += activation-previous-layer for prev-node for cur-example * gradient of Z of cur-layer cur example cur node in layer l
                        self.dW[l][prev][n] += (self.A[l - 1][m][prev] * self.dZ[l][m][n])
                    self.dW[l][prev][n] /= self.m   # average across example after summation


            # compute bias gradients by iterating through all nodes in cur-layer
            for n in range(self.dimensions[l]):
                # for each node iterate all examples
                for m in range(self.m):
                    # Backprop equation: db = (1/m) * sum(dZ)
                    self.db[l][n] += self.dZ[l][m][n]
                self.db[l][n] /= self.m     # average across examples after summation


            # calculate dZ for previous layer if previous layer is not input layer.
            if l - 1 > 0:
                # iterate all examples
                for m in range(self.m):
                    # iterate all nodes in preivous layer
                    for prev in range(self.dimensions[l - 1]):
                        dA = 0.0

                        # iterate all nodes in cur-layer
                        for n in range(self.dimensions[l]):
                            # Backprop equation: dA_previous = sum(W * dZ_current)
                            dA += self.W[l][prev][n] * self.dZ[l][m][n]
                        # Backprop equation: dZ_previous = dA_previous * g'(Z_previous)
                        self.dZ[l - 1][m][prev] = (dA * self.relu_backward(self.Z[l - 1][m][prev]))

    def update_parameters(self):
        # iterate all layers except input-layer
        for l in range(1, self.num_layers):
            # iterate all nodes nodes in preivous-layer
            for prev in range(self.dimensions[l - 1]):
                # iterate all nodes in cur-layer
                for n in range(self.dimensions[l]):
                    # gradient descent equation: W = W - learning_rate * dW
                    self.W[l][prev][n] -= self.learning_rate * self.dW[l][prev][n]

            # iterate all nodes in cur-layer
            for n in range(self.dimensions[l]):
                # gradient descent equation: b = b - learning_rate * db
                self.b[l][n] -= self.learning_rate * self.db[l][n]

    def train(self):
        # init stuff
        self.initialize_parameters()
        self.initialize_calculations()
        self.initialize_gradients()

        # train network for a number of iterations, for each epoch do thw following
        for epoch in range(self.num_iterations):

            # pass data through the networks
            self.forward_propagation()

            # compute the cost of how well its performing
            self.compute_cost()

            # store cost for cur-iteration
            self.costs.append(self.cost)

            # backprogate and compute gradients
            self.backward_propagation()

            # update weights & bias using gradient descent
            self.update_parameters()

            # print training progress every 100 epochs:
            if epoch % 100 == 0:
                print("Epoch/Iteration:", epoch, "Cost:", self.cost)

        # compute final cost using final updated parameters:
        self.forward_propagation()
        self.compute_cost()

        print("Final Cost:", self.cost)


    def predict(self, X):
        predictions = []

        # save original training data:
        original_X = self.X
        original_m = self.m

        self.X = X
        self.m = len(X)

        # init calculations again because we predicting a new thing
        self.initialize_calculations()

        # pass data forward
        self.forward_propagation()


        output_l_indx = self.num_layers - 1

        # for every example get what the network outputted, and turn that into binary classification prediction.
        for m in range(self.m):
            if self.A[output_l_indx][m][0] >= 0.5:
                predictions.append(1)
            else:
                predictions.append(0)

        # re put original training data:
        self.X = original_X
        self.m = original_m
        self.initialize_calculations()
        return predictions

    def evaluate_accuracy(self, X, Y):
        predictions = self.predict(X)
        correct = 0

        # compare
        for m in range(len(Y)):
            if predictions[m] == Y[m]:
                correct += 1

        # compute accuracy
        accuracy = correct / len(Y)

        return accuracy


def run_gaussian_2d_narrow(dimensions, learning_rate, num_iterations):
    print("\n=========== GAUSSIAN 2D NARROW ===========")

    network = FeedForwardNeuralNetwork(gaussian_2d_narrow_X_train, gaussian_2d_narrow_Y_train, dimensions, learning_rate, num_iterations)
    network.train()

    training_accuracy = network.evaluate_accuracy(gaussian_2d_narrow_X_train, gaussian_2d_narrow_Y_train)
    validation_accuracy = network.evaluate_accuracy(gaussian_2d_narrow_X_validation, gaussian_2d_narrow_Y_validation)

    print("Training Accuracy:", training_accuracy)
    print("Validation Accuracy:", validation_accuracy)

    return training_accuracy, validation_accuracy, network.cost, network.costs, network


def run_gaussian_2d_overlap(dimensions, learning_rate, num_iterations):
    print("\n=========== GAUSSIAN 2D OVERLAP ===========")

    network = FeedForwardNeuralNetwork(gaussian_2d_overlap_X_train, gaussian_2d_overlap_Y_train, dimensions, learning_rate, num_iterations)
    network.train()

    training_accuracy = network.evaluate_accuracy(gaussian_2d_overlap_X_train, gaussian_2d_overlap_Y_train)
    validation_accuracy = network.evaluate_accuracy(gaussian_2d_overlap_X_validation, gaussian_2d_overlap_Y_validation)

    print("Training Accuracy:", training_accuracy)
    print("Validation Accuracy:", validation_accuracy)

    return training_accuracy, validation_accuracy, network.cost, network.costs, network


def run_gaussian_2d_wide(dimensions, learning_rate, num_iterations):
    print("\n=========== GAUSSIAN 2D WIDE ===========")

    network = FeedForwardNeuralNetwork(gaussian_2d_wide_X_train, gaussian_2d_wide_Y_train, dimensions, learning_rate, num_iterations)
    network.train()

    training_accuracy = network.evaluate_accuracy(gaussian_2d_wide_X_train, gaussian_2d_wide_Y_train)
    validation_accuracy = network.evaluate_accuracy(gaussian_2d_wide_X_validation, gaussian_2d_wide_Y_validation)

    print("Training Accuracy:", training_accuracy)
    print("Validation Accuracy:", validation_accuracy)

    return training_accuracy, validation_accuracy, network.cost, network.costs, network


def run_gaussian_3d_narrow(dimensions, learning_rate, num_iterations):
    print("\n=========== GAUSSIAN 3D NARROW ===========")

    network = FeedForwardNeuralNetwork(gaussian_3d_narrow_X_train, gaussian_3d_narrow_Y_train, dimensions, learning_rate, num_iterations)
    network.train()

    training_accuracy = network.evaluate_accuracy(gaussian_3d_narrow_X_train, gaussian_3d_narrow_Y_train)
    validation_accuracy = network.evaluate_accuracy(gaussian_3d_narrow_X_validation, gaussian_3d_narrow_Y_validation)

    print("Training Accuracy:", training_accuracy)
    print("Validation Accuracy:", validation_accuracy)

    return training_accuracy, validation_accuracy, network.cost, network.costs, network


def run_gaussian_3d_overlap(dimensions, learning_rate, num_iterations):
    print("\n=========== GAUSSIAN 3D OVERLAP ===========")

    network = FeedForwardNeuralNetwork(gaussian_3d_overlap_X_train, gaussian_3d_overlap_Y_train, dimensions, learning_rate, num_iterations)
    network.train()

    training_accuracy = network.evaluate_accuracy(gaussian_3d_overlap_X_train, gaussian_3d_overlap_Y_train)
    validation_accuracy = network.evaluate_accuracy(gaussian_3d_overlap_X_validation, gaussian_3d_overlap_Y_validation)

    print("Training Accuracy:", training_accuracy)
    print("Validation Accuracy:", validation_accuracy)

    return training_accuracy, validation_accuracy, network.cost, network.costs, network


def run_gaussian_3d_wide(dimensions, learning_rate, num_iterations):
    print("\n=========== GAUSSIAN 3D WIDE ===========")

    network = FeedForwardNeuralNetwork(gaussian_3d_wide_X_train, gaussian_3d_wide_Y_train, dimensions, learning_rate, num_iterations)
    network.train()

    training_accuracy = network.evaluate_accuracy(gaussian_3d_wide_X_train, gaussian_3d_wide_Y_train)
    validation_accuracy = network.evaluate_accuracy(gaussian_3d_wide_X_validation, gaussian_3d_wide_Y_validation)

    print("Training Accuracy:", training_accuracy)
    print("Validation Accuracy:", validation_accuracy)

    return training_accuracy, validation_accuracy, network.cost, network.costs, network


def run_moon_2d_narrow(dimensions, learning_rate, num_iterations):
    print("\n=========== MOON 2D NARROW ===========")

    network = FeedForwardNeuralNetwork(moons_2d_narrow_X_train, moons_2d_narrow_Y_train, dimensions, learning_rate, num_iterations)
    network.train()

    training_accuracy = network.evaluate_accuracy(moons_2d_narrow_X_train, moons_2d_narrow_Y_train)
    validation_accuracy = network.evaluate_accuracy(moons_2d_narrow_X_validation, moons_2d_narrow_Y_validation)

    print("Training Accuracy:", training_accuracy)
    print("Validation Accuracy:", validation_accuracy)

    return training_accuracy, validation_accuracy, network.cost, network.costs, network


def run_moon_2d_overlap(dimensions, learning_rate, num_iterations):
    print("\n=========== MOON 2D OVERLAP ===========")

    network = FeedForwardNeuralNetwork(moons_2d_overlap_X_train, moons_2d_overlap_Y_train, dimensions, learning_rate, num_iterations)
    network.train()

    training_accuracy = network.evaluate_accuracy(moons_2d_overlap_X_train, moons_2d_overlap_Y_train)
    validation_accuracy = network.evaluate_accuracy(moons_2d_overlap_X_validation, moons_2d_overlap_Y_validation)

    print("Training Accuracy:", training_accuracy)
    print("Validation Accuracy:", validation_accuracy)

    return training_accuracy, validation_accuracy, network.cost, network.costs, network


def run_moon_2d_wide(dimensions, learning_rate, num_iterations):
    print("\n=========== MOON 2D WIDE ===========")

    network = FeedForwardNeuralNetwork(moons_2d_wide_X_train, moons_2d_wide_Y_train, dimensions, learning_rate, num_iterations)
    network.train()

    training_accuracy = network.evaluate_accuracy(moons_2d_wide_X_train, moons_2d_wide_Y_train)
    validation_accuracy = network.evaluate_accuracy(moons_2d_wide_X_validation, moons_2d_wide_Y_validation)

    print("Training Accuracy:", training_accuracy)
    print("Validation Accuracy:", validation_accuracy)

    return training_accuracy, validation_accuracy, network.cost, network.costs, network


# get trained network with highest validation accuracy
def get_best_network(results):
    # start with first result as best
    best_result = results[0]

    # check every repetition
    for result in results:
        # if validation accuracy is higher, make this the best result
        if result[1] > best_result[1]:
            best_result = result

    # return trained network from best result
    return best_result[4]


if __name__ == "__main__":
    root_directory = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(root_directory, "data")
    os.makedirs(data_directory, exist_ok=True)

    save_terminal_output(os.path.join(data_directory, "terminal_summary.txt"))

    # simple network configuration
    simple_dimensions_2d = [2, 3, 1]
    simple_dimensions_3d = [3, 3, 1]
    simple_learning_rate = 0.01
    simple_num_iterations = 3500

    # complex network configuration
    complex_dimensions_2d = [2, 10, 5, 1]
    complex_dimensions_3d = [3, 10, 5, 1]
    complex_learning_rate = 0.01
    complex_num_iterations = 3500

    num_repetitions = 1

    # ============================================================
    # SIMPLE NETWORK
    # ============================================================

    gaussian_2d_narrow_simple = []
    gaussian_2d_overlap_simple = []
    gaussian_2d_wide_simple = []
    gaussian_3d_narrow_simple = []
    gaussian_3d_overlap_simple = []
    gaussian_3d_wide_simple = []
    moon_2d_narrow_simple = []
    moon_2d_overlap_simple = []
    moon_2d_wide_simple = []

    print("\n\n==================================================")
    print("RUNNING SIMPLE NETWORK")
    print("==================================================")

    for repetition in range(num_repetitions):
        print("\n========================================")
        print("SIMPLE NETWORK - REPETITION:", repetition + 1)
        print("========================================")

        gaussian_2d_narrow_simple.append(run_gaussian_2d_narrow(simple_dimensions_2d, simple_learning_rate, simple_num_iterations))
        gaussian_2d_overlap_simple.append(run_gaussian_2d_overlap(simple_dimensions_2d, simple_learning_rate, simple_num_iterations))
        gaussian_2d_wide_simple.append(run_gaussian_2d_wide(simple_dimensions_2d, simple_learning_rate, simple_num_iterations))

        gaussian_3d_narrow_simple.append(run_gaussian_3d_narrow(simple_dimensions_3d, simple_learning_rate, simple_num_iterations))
        gaussian_3d_overlap_simple.append(run_gaussian_3d_overlap(simple_dimensions_3d, simple_learning_rate, simple_num_iterations))
        gaussian_3d_wide_simple.append(run_gaussian_3d_wide(simple_dimensions_3d, simple_learning_rate, simple_num_iterations))

        moon_2d_narrow_simple.append(run_moon_2d_narrow(simple_dimensions_2d, simple_learning_rate, simple_num_iterations))
        moon_2d_overlap_simple.append(run_moon_2d_overlap(simple_dimensions_2d, simple_learning_rate, simple_num_iterations))
        moon_2d_wide_simple.append(run_moon_2d_wide(simple_dimensions_2d, simple_learning_rate, simple_num_iterations))

    # ============================================================
    # COMPLEX NETWORK
    # ============================================================

    gaussian_2d_narrow_complex = []
    gaussian_2d_overlap_complex = []
    gaussian_2d_wide_complex = []
    gaussian_3d_narrow_complex = []
    gaussian_3d_overlap_complex = []
    gaussian_3d_wide_complex = []
    moon_2d_narrow_complex = []
    moon_2d_overlap_complex = []
    moon_2d_wide_complex = []

    print("\n\n==================================================")
    print("RUNNING COMPLEX NETWORK")
    print("==================================================")

    for repetition in range(num_repetitions):
        print("\n========================================")
        print("COMPLEX NETWORK - REPETITION:", repetition + 1)
        print("========================================")

        gaussian_2d_narrow_complex.append(run_gaussian_2d_narrow(complex_dimensions_2d, complex_learning_rate, complex_num_iterations))
        gaussian_2d_overlap_complex.append(run_gaussian_2d_overlap(complex_dimensions_2d, complex_learning_rate, complex_num_iterations))
        gaussian_2d_wide_complex.append(run_gaussian_2d_wide(complex_dimensions_2d, complex_learning_rate, complex_num_iterations))

        gaussian_3d_narrow_complex.append(run_gaussian_3d_narrow(complex_dimensions_3d, complex_learning_rate, complex_num_iterations))
        gaussian_3d_overlap_complex.append(run_gaussian_3d_overlap(complex_dimensions_3d, complex_learning_rate, complex_num_iterations))
        gaussian_3d_wide_complex.append(run_gaussian_3d_wide(complex_dimensions_3d, complex_learning_rate, complex_num_iterations))

        moon_2d_narrow_complex.append(run_moon_2d_narrow(complex_dimensions_2d, complex_learning_rate, complex_num_iterations))
        moon_2d_overlap_complex.append(run_moon_2d_overlap(complex_dimensions_2d, complex_learning_rate, complex_num_iterations))
        moon_2d_wide_complex.append(run_moon_2d_wide(complex_dimensions_2d, complex_learning_rate, complex_num_iterations))

    # names used for printing, CSV files, and graphs
    dataset_names = [
        "Gaussian 2D Narrow", "Gaussian 2D Overlap", "Gaussian 2D Wide",
        "Gaussian 3D Narrow", "Gaussian 3D Overlap", "Gaussian 3D Wide",
        "Moon 2D Narrow", "Moon 2D Overlap", "Moon 2D Wide"
    ]

    # put simple network results together
    simple_results = [
        gaussian_2d_narrow_simple, gaussian_2d_overlap_simple, gaussian_2d_wide_simple,
        gaussian_3d_narrow_simple, gaussian_3d_overlap_simple, gaussian_3d_wide_simple,
        moon_2d_narrow_simple, moon_2d_overlap_simple, moon_2d_wide_simple
    ]

    # put complex network results together
    complex_results = [
        gaussian_2d_narrow_complex, gaussian_2d_overlap_complex, gaussian_2d_wide_complex,
        gaussian_3d_narrow_complex, gaussian_3d_overlap_complex, gaussian_3d_wide_complex,
        moon_2d_narrow_complex, moon_2d_overlap_complex, moon_2d_wide_complex
    ]

    # print results separately
    print("\n\n==================================================")
    print("SIMPLE NETWORK RESULTS")
    print("==================================================")
    print_results(dataset_names, simple_results)

    print("\n\n==================================================")
    print("COMPLEX NETWORK RESULTS")
    print("==================================================")
    print_results(dataset_names, complex_results)


        # ============================================================
    # OUTPUT DIRECTORIES
    # ============================================================

    root_directory = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(root_directory, "data")
    compare_directory = os.path.join(root_directory, "compare_figures")

    os.makedirs(data_directory, exist_ok=True)
    os.makedirs(compare_directory, exist_ok=True)

    plot_validation_accuracy_comparison(dataset_names,simple_results,complex_results,"validation_accuracy.png")


    # ============================================================
    # SAVE CSV DATA
    # ============================================================

    results.DIRECTORY = data_directory

    save_results_csv(dataset_names, simple_results, "results_simple_net_data.csv")
    save_results_csv(dataset_names, complex_results, "results_complex_net_data.csv")


    # ============================================================
    # SAVE COMPARISON FIGURES
    # ============================================================

    results.DIRECTORY = compare_directory

    # overall average cost plots
    plot_average_cost(dataset_names, simple_results, "Simple Network", "simple_average_cost.png")
    plot_average_cost(dataset_names, complex_results, "Complex Network", "complex_average_cost.png")

    # simple vs complex cost for each dataset
    plot_network_cost_comparison("Gaussian 2D Narrow", gaussian_2d_narrow_simple, gaussian_2d_narrow_complex, "G_2D_narrow_cost.png")
    plot_network_cost_comparison("Gaussian 2D Overlap", gaussian_2d_overlap_simple, gaussian_2d_overlap_complex, "G_2D_overlap_cost.png")
    plot_network_cost_comparison("Gaussian 2D Wide", gaussian_2d_wide_simple, gaussian_2d_wide_complex, "G_2D_wide_cost.png")

    plot_network_cost_comparison("Gaussian 3D Narrow", gaussian_3d_narrow_simple, gaussian_3d_narrow_complex, "G_3D_narrow_cost.png")
    plot_network_cost_comparison("Gaussian 3D Overlap", gaussian_3d_overlap_simple, gaussian_3d_overlap_complex, "G_3D_overlap_cost.png")
    plot_network_cost_comparison("Gaussian 3D Wide", gaussian_3d_wide_simple, gaussian_3d_wide_complex, "G_3D_wide_cost.png")

    plot_network_cost_comparison("Moon 2D Narrow", moon_2d_narrow_simple, moon_2d_narrow_complex, "M_2D_narrow_cost.png")
    plot_network_cost_comparison("Moon 2D Overlap", moon_2d_overlap_simple, moon_2d_overlap_complex, "M_2D_overlap_cost.png")
    plot_network_cost_comparison("Moon 2D Wide", moon_2d_wide_simple, moon_2d_wide_complex, "M_2D_wide_cost.png")




    # ============================================================
    # SIMPLE NETWORK DECISION BOUNDARIES
    # ============================================================

    simple_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simple_network_figures")
    os.makedirs(simple_directory, exist_ok=True)
    results.DIRECTORY = simple_directory

    G_2D_narrow = get_best_network(gaussian_2d_narrow_simple)
    G_2D_overlap = get_best_network(gaussian_2d_overlap_simple)
    G_2D_wide = get_best_network(gaussian_2d_wide_simple)
    G_3D_narrow = get_best_network(gaussian_3d_narrow_simple)
    G_3D_overlap = get_best_network(gaussian_3d_overlap_simple)
    G_3D_wide = get_best_network(gaussian_3d_wide_simple)
    M_2D_narrow = get_best_network(moon_2d_narrow_simple)
    M_2D_overlap = get_best_network(moon_2d_overlap_simple)
    M_2D_wide = get_best_network(moon_2d_wide_simple)

    plot_decision_boundary_2d(G_2D_narrow, gaussian_2d_narrow_X_train, gaussian_2d_narrow_Y_train, "Gaussian 2D Narrow - Simple Network", "G_2D_narrow_boundary.png")
    plot_decision_boundary_2d(G_2D_overlap, gaussian_2d_overlap_X_train, gaussian_2d_overlap_Y_train, "Gaussian 2D Overlap - Simple Network", "G_2D_overlap_boundary.png")
    plot_decision_boundary_2d(G_2D_wide, gaussian_2d_wide_X_train, gaussian_2d_wide_Y_train, "Gaussian 2D Wide - Simple Network", "G_2D_wide_boundary.png")

    plot_decision_boundary_3d(G_3D_narrow, gaussian_3d_narrow_X_train, gaussian_3d_narrow_Y_train, "Gaussian 3D Narrow - Simple Network", "G_3D_narrow_boundary.png")
    plot_decision_boundary_3d(G_3D_overlap, gaussian_3d_overlap_X_train, gaussian_3d_overlap_Y_train, "Gaussian 3D Overlap - Simple Network", "G_3D_overlap_boundary.png")
    plot_decision_boundary_3d(G_3D_wide, gaussian_3d_wide_X_train, gaussian_3d_wide_Y_train, "Gaussian 3D Wide - Simple Network", "G_3D_wide_boundary.png")

    plot_decision_boundary_2d(M_2D_narrow, moons_2d_narrow_X_train, moons_2d_narrow_Y_train, "Moon 2D Narrow - Simple Network", "M_2D_narrow_boundary.png")
    plot_decision_boundary_2d(M_2D_overlap, moons_2d_overlap_X_train, moons_2d_overlap_Y_train, "Moon 2D Overlap - Simple Network", "M_2D_overlap_boundary.png")
    plot_decision_boundary_2d(M_2D_wide, moons_2d_wide_X_train, moons_2d_wide_Y_train, "Moon 2D Wide - Simple Network", "M_2D_wide_boundary.png")


    # ============================================================
    # COMPLEX NETWORK DECISION BOUNDARIES
    # ============================================================

    complex_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "complex_network_figures")
    os.makedirs(complex_directory, exist_ok=True)
    results.DIRECTORY = complex_directory

    G_2D_narrow = get_best_network(gaussian_2d_narrow_complex)
    G_2D_overlap = get_best_network(gaussian_2d_overlap_complex)
    G_2D_wide = get_best_network(gaussian_2d_wide_complex)
    G_3D_narrow = get_best_network(gaussian_3d_narrow_complex)
    G_3D_overlap = get_best_network(gaussian_3d_overlap_complex)
    G_3D_wide = get_best_network(gaussian_3d_wide_complex)
    M_2D_narrow = get_best_network(moon_2d_narrow_complex)
    M_2D_overlap = get_best_network(moon_2d_overlap_complex)
    M_2D_wide = get_best_network(moon_2d_wide_complex)

    plot_decision_boundary_2d(G_2D_narrow, gaussian_2d_narrow_X_train, gaussian_2d_narrow_Y_train, "Gaussian 2D Narrow - Complex Network", "G_2D_narrow_boundary.png")
    plot_decision_boundary_2d(G_2D_overlap, gaussian_2d_overlap_X_train, gaussian_2d_overlap_Y_train, "Gaussian 2D Overlap - Complex Network", "G_2D_overlap_boundary.png")
    plot_decision_boundary_2d(G_2D_wide, gaussian_2d_wide_X_train, gaussian_2d_wide_Y_train, "Gaussian 2D Wide - Complex Network", "G_2D_wide_boundary.png")

    plot_decision_boundary_3d(G_3D_narrow, gaussian_3d_narrow_X_train, gaussian_3d_narrow_Y_train, "Gaussian 3D Narrow - Complex Network", "G_3D_narrow_boundary.png")
    plot_decision_boundary_3d(G_3D_overlap, gaussian_3d_overlap_X_train, gaussian_3d_overlap_Y_train, "Gaussian 3D Overlap - Complex Network", "G_3D_overlap_boundary.png")
    plot_decision_boundary_3d(G_3D_wide, gaussian_3d_wide_X_train, gaussian_3d_wide_Y_train, "Gaussian 3D Wide - Complex Network", "G_3D_wide_boundary.png")

    plot_decision_boundary_2d(M_2D_narrow, moons_2d_narrow_X_train, moons_2d_narrow_Y_train, "Moon 2D Narrow - Complex Network", "M_2D_narrow_boundary.png")
    plot_decision_boundary_2d(M_2D_overlap, moons_2d_overlap_X_train, moons_2d_overlap_Y_train, "Moon 2D Overlap - Complex Network", "M_2D_overlap_boundary.png")
    plot_decision_boundary_2d(M_2D_wide, moons_2d_wide_X_train, moons_2d_wide_Y_train, "Moon 2D Wide - Complex Network", "M_2D_wide_boundary.png")