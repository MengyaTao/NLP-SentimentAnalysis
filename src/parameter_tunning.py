
"""
plot the relationship between the hyper-parameters and the cost
# plot 1 - learning rate vs. cost (training data)
"""
# Libraries
import json
import random
import sys
import matplotlib.pyplot as plt
import numpy as np
import loading_data as ld
import NeuralNetwork as nn

# Constants
LEARNING_RATES = [0.001, 0.01, 0.1, 1, 10]
RE_LAMBDA = [0.001, 0.01, 0.1, 1, 10]
BATCH_SIZE = [10,100,200,300,400,500,600,700,800,1000]
NEURONS = [10,30,50,100,150,200,500, 1000]
COLORS = ['blue', 'red', 'orange', 'green','purple']
NUM_ITERATIONS = 50
NUM_FEATURES = [10,50, 100,300,600, 1000, 2000, 3000, 5000, 10000]
COLORS_features = ['blue', 'red', 'orange', 'green','purple','black','yellow','cyan','chocolate','magenta']
COLORS_neurons = ['blue', 'red', 'orange', 'green','purple','black','cyan','magenta']

def run_network_eta():
    np.random.seed(12345678)
    training_data, validation_data, test_data = ld.load_data('training_data_clean', 'validation_data_clean',
                                                             'test_data_clean', 300)
    results = []
    for eta in LEARNING_RATES:
        net = nn.NeuralNetwork([300, 30, 2])
        results.append(
            net.stochastic_gradient_descent(training_data, NUM_ITERATIONS, 10, eta, re_lambda=0.0,
                                            validation_data=validation_data, show_training_cost=True))
    f = open("multiple_eta.json", "w")
    json.dump(results, f)
    f.close()

def make_plot_eta():
    f = open("multiple_eta.json", "r")
    results = json.load(f)
    f.close()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for eta, result, color in zip(LEARNING_RATES, results, COLORS):
        training_cost, _, _, _ = result
        ax.plot(np.arange(NUM_ITERATIONS), training_cost, "o-",
                label="$\eta$ = "+str(eta),
                color=color)
    ax.set_xlim([0, NUM_ITERATIONS])
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost')
    plt.legend(loc='upper right')
    plt.show()

def run_network_lambda():
    np.random.seed(12345678)
    training_data, validation_data, test_data = ld.load_data('training_data_clean', 'validation_data_clean',
                                                             'test_data_clean', 300)
    results = []
    for re_lambda in RE_LAMBDA:
        net = nn.NeuralNetwork([300, 30, 2])
        results.append(
            net.stochastic_gradient_descent(training_data, NUM_ITERATIONS, 10, learning_rate=0.1, re_lambda=re_lambda,
                                            validation_data=validation_data, show_validation_accuracy=True))
    f = open("multiple_lambda.json", "w")
    json.dump(results, f)
    f.close()

def make_plot_lambda():
    f = open("multiple_lambda.json", "r")
    results = json.load(f)
    f.close()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for re_lambda, result, color in zip(RE_LAMBDA, results, COLORS):
        _, _, _, validation_accuracy = result
        ax.plot(np.arange(NUM_ITERATIONS), validation_accuracy, "o-",
                label="$\lambda$ = "+str(re_lambda),
                color=color)
    ax.set_xlim([0, NUM_ITERATIONS])
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Accuracy')
    plt.legend(loc='upper right')
    plt.show()

def run_network_feature():
    np.random.seed(12345678)
    results = []
    for num_feature in NUM_FEATURES:
        training_data, validation_data, test_data = ld.load_data('training_data_clean', 'validation_data_clean',
                                                                 'test_data_clean', num_feature)
        net = nn.NeuralNetwork([num_feature, 30, 2])
        results.append(
            net.stochastic_gradient_descent(training_data, NUM_ITERATIONS, 10, learning_rate=0.1, re_lambda=0.1,
                                            validation_data=validation_data, show_validation_accuracy=True))
    f = open("multiple_feature.json", "w")
    json.dump(results, f)
    f.close()

def make_plot_feature():
    f = open("multiple_feature.json", "r")
    results = json.load(f)
    f.close()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for num_feature, result, color in zip(NUM_FEATURES, results, COLORS_features):
        _, _, _, validation_accuracy = result
        ax.plot(np.arange(NUM_ITERATIONS), validation_accuracy, "o-",
                label="$\ numFeatures$ = "+str(num_feature),
                color=color)
    ax.set_xlim([0, NUM_ITERATIONS])
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Accuracy')
    plt.legend(loc='upper right')
    plt.show()


def run_network_neuron():
    np.random.seed(12345678)
    results = []
    for neuron in NEURONS:
        training_data, validation_data, test_data = ld.load_data('training_data_clean', 'validation_data_clean',
                                                                 'test_data_clean', 300)
        net = nn.NeuralNetwork([300, neuron, 2])
        results.append(
            net.stochastic_gradient_descent(training_data, NUM_ITERATIONS, 10, learning_rate=0.1, re_lambda=0.1,
                                            validation_data=validation_data, show_validation_accuracy=True))
    f = open("multiple_neuron.json", "w")
    json.dump(results, f)
    f.close()


def make_plot_neuron():
    f = open("multiple_neuron.json", "r")
    results = json.load(f)
    f.close()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for neuron, result, color in zip(NEURONS, results, COLORS_neurons):
        _, _, _, validation_accuracy = result
        ax.plot(np.arange(NUM_ITERATIONS), validation_accuracy, "o-",
                label="$\ numNeurons$ = " + str(neuron),
                color=color)
    ax.set_xlim([0, NUM_ITERATIONS])
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Accuracy')
    plt.legend(loc='bottom right')
    plt.show()
