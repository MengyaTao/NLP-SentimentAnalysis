import loading_data as ld
import NeuralNetwork as nn
import parameter_tunning as pt
import json

if __name__ == '__main__':
    # 1) tunning the optimal learning rate based on the training cost
    pt.run_network_eta()
    pt.make_plot_eta()  # from the plot, learning rate = 0.1 is the optimal one

    # 2) tunning the optimal regularization term lambda, use learning rate=0.1
    pt.run_network_lambda()
    pt.make_plot_lambda()  # it fluctuates a lot, need more investigation on this, select lambda=0.1 for now

    # 3) tunning the optimal # of features based on the validation data accuracy
    # use learning_rate = 0.1, re_lambda = 0.1
    pt.run_network_feature()
    pt.make_plot_feature()  # feature size = 300 is selected based on the plot figure

    # 4) tunning the optimal # of neurons based on the validation data accuracy
    # use learning_rate = 0.1, re_lambda = 0.1, num_features = 300
    pt.run_network_neuron()
    pt.make_plot_neuron()  # select neuron = 100

    # 5) based on those selected hyperparameters, plot the training accuracy and validation accuracy
    # learning_rate = 0.1, re_lambda = 0.1, batch_size=100, num_feature = 300, neuron = 100
    training_data, validation_data, test_data = ld.load_data('training_data_clean', 'validation_data_clean',
                                                             'test_data_clean', 300)
    net = nn.NeuralNetwork([300, 100, 2])
    training_cost, training_accuracy, validation_cost, validation_accuracy = \
        net.stochastic_gradient_descent(training_data, iterations=50, batch_size=10,
                                        learning_rate=0.1, re_lambda=0.1,
                                        validation_data=validation_data,
                                        show_training_cost=False,
                                        show_training_accuracy=True,
                                        show_validation_cost=False,
                                        show_validation_accuracy=True)

    f = open("accuracy.json", "w")
    json.dump([training_accuracy, validation_accuracy], f)
    f.close()

    f = open("accuracy.json", "r")
    training_accuracy, validation_accuracy = json.load(f)
    f.close()
    nn.plot_overlay(training_accuracy, validation_accuracy, 50, xmin=0)
    # after the plot, we can see that around the 10th iteration, the validation_accuracy stop increasing
    # so we can save the weights and biases around that iteration and use it for test data prediction
    training_data, validation_data, test_data = ld.load_data('training_data_clean', 'validation_data_clean',
                                                             'test_data_clean', 300)
    net = nn.NeuralNetwork([300, 100, 2])
    nn_para = net.stochastic_gradient_descent(training_data, iterations=10, batch_size=10,
                                              learning_rate=0.1, re_lambda=0.1,
                                              validation_data=validation_data,
                                              show_training_accuracy=True,
                                              show_validation_accuracy=True)

    net.save("nn_para")
    net1 = nn.load("nn_para")
    print net1.accuracy(test_data)
    # training_accuracy = 0.88
    # validation_accuracy = 0.779
    # test_accuracy = 765/960 = 0.797


