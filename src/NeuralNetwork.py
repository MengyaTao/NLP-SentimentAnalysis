# based on the basic concept of the followings:
# 1) basic neural network
#        - it can deal with multiple hidden layers (architecture)
# 2) sigmoid neuron
# 3) two kinds of cost functions:
#        i. quadratic cost
#        ii. cross-entropy
# 4) regularization function L2:
#        - C = -(1/n)*[yln(a)+(1-y)ln(1-a)] + lambda/(2n)*w^2
# 5) stochastic gradient descent algorithms
#        i. mini-bach size
#        ii. epoch (for iterations)
#        iii. backpropatation - compute partial derivatives
# references:
# 1. Michael Nielsen, 2016, Neural Networks and Deep Learning
# 2. Andrew Ng, Machine Learning, Coursera

################### Part I - Libraries #####################
import numpy as np
import json
import sys
import random
import matplotlib.pyplot as plt

################## Part II - Sigmoid Neurons ################
def sigmoid(z):
    a = 1.0/(1.0+np.exp(-z))
    return a

# the derivative of sigmoid function is used to calculate the
# error term delta
def sigmoid_derivative(z):
    b = sigmoid(z)*(1-sigmoid(z))
    return b

# vectorized the y vector
def vectorized_result(j):
    '''return a 2-unit vector with 1.0 in the j the position and zero
    in the other position. '''
    c = np.zeros((2,1))
    c[j] = 1.0
    return c

################## Part III - Cost Functions ################
# cost function is implemented as a class instead of a function
# obj 1: measure how well an output activation, a, matches the output, y
# obj 2: compute the network's output error, delta
################################
# 1) quadratic cost function (also known as mean square error MSE)
class Cost_QuadraticCost(object):
    @staticmethod
    def function(a, y):
        # C = 1/(2*n)*||y(x)-a||^2
        # np.linalg.norm(x) = sqrt(x1^2+x2^2+...)
        c = 0.5*np.linalg.norm(a-y)**2
        return c
    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a-y) * sigmoid_derivative(z)

class Cost_CrossEntropy(object):
    @staticmethod
    def function(a, y):
        # na.nan_to_num is used to ensure the log of numbers are taken care of
        # replace nan with zero and inf with finite numbers
        c = np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
        return c
    @staticmethod
    def delta(z, a, y):
        return (a-y)

################## Part IV - Neural Network ################
# functions needed are as follows:
#   1) weights and bias initializer
#   2) feedforward function
#   3) stochastic gradient descent algorithms
#   4) backpropagation
#   5) weights and bias updates function
#   6) total cost function with regularization term
#   7) accuracy function

class NeuralNetwork(object):
    def __init__(self, design, cost=Cost_CrossEntropy):
        '''if the design looks like [5,8,8,3], it's a 4-layer network
           the input layer contains 5 neuron, the output layer has 3
           neurons 2 hidder layers, with 8 neurons in each hidden layer.'''
        self.num_of_layers = len(design)
        self.design = design
        self.para_initializer()
        self.cost = cost

    def para_initializer(self):
        '''weights are initialized randomly, and the number of weights
           equals the number of neurons in each layer;
           biases are also initizlied randomly, and the input layer
           (first layer) doesn't have biases.
           the way to generate weights and biases is to avoid saturated
           output neuron problem and less likely to slowdown the learning
           rate. The default parameter (weights and biases) initialization method:
           Gaussian random variables N(0,1)/square root of the # of inputs'''
        self.biases = [np.random.randn(y,1) for y in self.design[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.design[:-1], self.design[1:])]

    def feedforward(self, a):
        ''' return the output of the neural network when the activation
        a as the input'''
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a) + b
            a = sigmoid(z)
        return a

    def stochastic_gradient_descent(self, training_data, iterations, batch_size, learning_rate,
                                    re_lambda=0.0,
                                    validation_data=None,
                                    show_training_cost=False,
                                    show_training_accuracy=False,
                                    show_validation_cost=False,
                                    show_validation_accuracy=False):
        if validation_data:n_data = len(validation_data)
        n = len(training_data)
        training_cost, training_accuracy, validation_cost, validation_accuracy = [],[],[],[]
        for j in xrange(iterations):
            random.shuffle(training_data)
            batches = [
                training_data[k:k + batch_size]
                for k in xrange(0, n, batch_size)]

            for batch in batches:
                self.update_parameter(
                    batch, learning_rate, re_lambda, len(training_data))
            print "iteration %s training complete" % j

            if show_training_cost:
                cost = self.total_cost(training_data, re_lambda)
                training_cost.append(cost)
                print "cost on training data with L2 regularization: %s" %cost
            if show_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)*1.0/n
                training_accuracy.append(accuracy)
                print "accuracy on training data:%s" %accuracy
            if show_validation_cost:
                cost = self.total_cost(validation_data, re_lambda, convert=True)
                validation_cost.append(cost)
                print "cost on validation data with L2 regularization: %s" %cost
            if show_validation_accuracy:
                accuracy = self.accuracy(validation_data)*1.0/n_data
                validation_accuracy.append(accuracy)
                print "accuracy on validation data: %s" %accuracy
            print

        return training_cost, training_accuracy, validation_cost, validation_accuracy

    def update_parameter(self, batch, learning_rate, re_lambda, n):
        '''update the weights and biases parameters by gradient descent
        using backpropagation algorithm to a single batch_size of sample.
        re_lmbda is the regularization parameter'''
        # initiate zeros for temperate change of biases and weights
        new_b = [np.zeros(b.shape) for b in self.biases]
        new_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in batch:
            delta_new_b, delta_new_w = self.backpropagation(x, y)
            new_b = [nb + dnb for nb, dnb in zip(new_b, delta_new_b)]
            new_w = [nw + dnw for nw, dnw in zip(new_w, delta_new_w)]
        self.weights = [(1 - learning_rate * (re_lambda / n)) * w - (learning_rate / len(batch)) * nw
                        for w, nw in zip(self.weights, new_w)]
        self.biases = [b - (learning_rate / len(batch)) * nb
                       for b, nb in zip(self.biases, new_b)]

    def backpropagation(self, x, y):
        '''return (delta_new_b, delta_new_w)
        representing the gradient for the cost function C'''
        new_b = [np.zeros(b.shape) for b in self.biases]
        new_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward, a indicates activation, a=sigmoid(z)
        a = x
        a_s = [x]  # list to store all a vectors, layer by layer
        z_s = []  # list to store all z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a) + b
            z_s.append(z)
            a = sigmoid(z)
            a_s.append(a)

        # backward pass
        # backward pass
        # calculate the error term delta, start from the last layer
        delta = (self.cost).delta(z_s[-1], a_s[-1], y)
        new_b[-1] = delta
        new_w[-1] = np.dot(delta, a_s[-2].transpose())

        for l in xrange(2, self.num_of_layers):
            z = z_s[-l]
            sp = sigmoid_derivative(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            new_b[-l] = delta
            new_w[-l] = np.dot(delta, a_s[-l - 1].transpose())

        return (new_b, new_w)

    def total_cost(self, data, re_lambda, convert=False):
        '''the convert here is reserved compared to the accuracy function.
        the flag convert is set to False if the dataset is the training data'''
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert:
                y = vectorized_result(y)
            cost += self.cost.function(a, y) / len(data)
        cost += 0.5 * (re_lambda / len(data)) * sum(np.linalg.norm(w) ** 2 for w in self.weights)
        return cost

    def accuracy(self, data, convert=False):
        ''' convert is a flag, if the dataset is validation or test data,
        it is set to False; if the dataset is training, it is set to True'''
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       # argmax is used to return the index of the max number
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                       for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    def save(self, filename):
        '''save the neural network to the file'''
        data = {"design": self.design,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

############################## Part V - Make Plots #########################

def plot_training_cost(training_cost, num_iterations, training_cost_xmin):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(training_cost_xmin, num_iterations),
            training_cost[training_cost_xmin:num_iterations],
            color='orange')
    ax.set_xlim([training_cost_xmin, num_iterations])
    ax.grid(True)
    ax.set_xlabel('Iteration')
    ax.set_title('Cost on the training data')
    plt.show()

def plot_training_accuracy(training_accuracy, num_iterations, training_accuracy_xmin):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(training_accuracy_xmin, num_iterations),
            [accuracy for accuracy in training_accuracy[training_accuracy_xmin:num_iterations]],
            color='darkgreen')
    ax.set_xlim([training_accuracy_xmin, num_iterations])
    ax.grid(True)
    ax.set_xlabel('Iteration')
    ax.set_title('Accuracy on the training data')
    plt.show()

def plot_validation_cost(validation_cost, num_iterations, validation_cost_xmin):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(validation_cost_xmin, num_iterations),
            validation_cost[validation_cost_xmin:num_iterations],
            color='blue')
    ax.set_xlim([validation_cost_xmin, num_iterations])
    ax.grid(True)
    ax.set_xlabel('Iteration')
    ax.set_title('Cost on the test data')
    plt.show()

def plot_validation_accuracy(validation_accuracy, num_iterations, validation_accuracy_xmin):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(validation_accuracy_xmin, num_iterations),
            [accuracy for accuracy in validation_accuracy[validation_accuracy_xmin:num_iterations]],
            color='salmon')
    ax.set_xlim([validation_accuracy_xmin, num_iterations])
    ax.grid(True)
    ax.set_xlabel('Iteration')
    ax.set_title('Accuracy on the test data')
    plt.show()

def plot_overlay(training_accuracy, validation_accuracy, num_iterations, xmin):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(xmin, num_iterations),
            training_accuracy,"o-",
            color='chocolate',
            label="Accuracy on the training data")
    ax.plot(np.arange(xmin, num_iterations),
            validation_accuracy, "o-",
            color='green',
            label="Accuracy on the validation data")
    ax.grid(True)
    ax.set_xlim([xmin, num_iterations])
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Accuracy')
    ax.set_ylim([0, 1])
    plt.legend(loc="lower right")
    plt.show()


# loading a neural network
def load(filename):
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = NeuralNetwork(data["design"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net
