
import math
import random
from datasets import *

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
                    self.W[l][prev].append(random.uniform(-0.1, 0.1))

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


if __name__ == "__main__":

    # Load dataset:
    X = gaussian_2d_narrow_X
    Y = gaussian_2d_narrow_Y

    # Initialize network:
    network = FeedForwardNeuralNetwork(
        X=X,
        Y=Y,
        dimensions=[2, 4, 1],
        learning_rate=0.01,
        num_iterations=1000
    )

    # Train network:
    network.train()

    # Evaluate network performance on training data:
    accuracy = network.evaluate_accuracy(X, Y)

    print("Training Accuracy:", accuracy)