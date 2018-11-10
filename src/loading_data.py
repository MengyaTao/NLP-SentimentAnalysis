import numpy as np
import pandas as pd
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split

def load_data(train_file, validation_file, test_file, vector_size):
    n_features = vector_size
    train = pd.read_csv(train_file + '.csv',header=0)
    validation = pd.read_csv(validation_file + '.csv', header=0)
    # train, validation = train_test_split(tr_data, train_size=0.8, random_state=44)
    test = pd.read_csv(test_file + '.csv',header=0)

    vectorizer = CountVectorizer(analyzer="word", \
                                 tokenizer=None, \
                                 preprocessor=None, \
                                 stop_words=None, \
                                 max_features=n_features)

    train_data_features = vectorizer.fit_transform(train["Abstract"])
    train_data_features = train_data_features.toarray()
    # print np.shape(train_data_features), (1600, 1000)
    trainY = train["Score"].values
    # print np.shape(trainY), (1600,)

    validation_data_features = vectorizer.transform(validation["Abstract"])
    validation_data_features = validation_data_features.toarray()
    validationY = validation["Score"].values

    test_data_features = vectorizer.transform(test["Abstract"])
    test_data_features = test_data_features.toarray()
    # print np.shape(test_data_features), (2468, 1000)
    testY = test["Score"].values

    training_inputs = [np.reshape(x, (vector_size, 1)) for x in train_data_features]
    training_results = [vectorized_result(y) for y in trainY]
    # print np.shape(training_inputs), (1600, 1000,1)
    # print training_inputs[1]
    # print training_results[1]
    validation_inputs = [np.reshape(x, (vector_size, 1)) for x in validation_data_features]
    # validation_results = [vectorized_result(y) for y in validationY]

    test_inputs = [np.reshape(x, (vector_size, 1)) for x in test_data_features]
    # test_results = [vectorized_result(y) for y in testY]

    training_data = zip(list(training_inputs), training_results)
    validation_data = zip(list(validation_inputs), validation["Score"])
    test_data = zip(list(test_inputs), test["Score"])

    return training_data, validation_data, test_data


def vectorized_result(j):
    e = np.zeros((2, 1))
    e[j] = 1.0
    return e