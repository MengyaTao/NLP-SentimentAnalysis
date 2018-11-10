import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import cross_validation
from sklearn.cross_validation import KFold, cross_val_score
import numpy as np
import csv
import cPickle
import matplotlib.pyplot as plt
import json
import random
import sys

# input the cleaned data
random.seed(12345678)
np.random.seed(12345678)
train = pd.read_csv("training_data_clean.csv", header=0)
test = pd.read_csv("test_data_clean.csv", header=0)

###########Hyper-parameter tuning###############################
results = []  #Store the test results
para = []  #Store the parameters input for hyper-parameter tuning
#for i in range(10, 600, 10): #Test different vector_size for both SVM and Random Forest.
# Also used for tuning n_estimators of Random Forest

########SVM parameters#################
#gammad = [0.0001, 0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 100.0]          #Test gamma
gammad = [0.0009, 0.001, 0.002, 0.005, 0.008, 0.01, 0.02,0.05]       #Test gamma
#cd = [1.0, 3.0, 4.0, 5.0, 6.0, 7.0]                                 #Test parameter C
cd = [0.0001, 0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 100.0, 200.0, 600.0] #Test parameter C

#vector_size = [10, 50, 100, 400, 410, 430, 450, 460, 470, 480, 490, 500] #Test different vector_size

######################################################################
#################### Method 1: Bags of words #########################
######################################################################

for i in range(0, len(cd), 1):
    vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 stop_words = None,   \
                                 max_features = 470)

    train_data_features = vectorizer.fit_transform(train["Abstract"])

    train_data_features = train_data_features.toarray()

    # look at the words in vocabulary
    vocabulary = vectorizer.get_feature_names()

    # Get a bag of words for the test set, and convert to a numpy array
    # for the words that were not seen in the training corpus will be ignored
    # use vectorizer.transform
    test_data_features = vectorizer.transform(test["Abstract"])
    test_data_features = test_data_features.toarray()
    #===============================================================================
    # # print the counts of each word in the vocabulary
    dist = np.sum(train_data_features, axis = 0)
    for feature, count in zip(vocabulary, dist):
        print count, feature

    # ##############################################################################
    # ################### random forest ############################################
    # ##############################################################################
    # # initialize a random forest classifier with 100 trees
    # # increase the n_estimators to 1000, doesn't really increase the prediction accuracy
    forest = RandomForestClassifier(n_estimators = 310)

    # # use the bags of words as features and the score column as the response variable
    # # fit the model
    clf_RF = forest.fit(train_data_features, train["Score"])
    # # Use the random forest to make sentiment label predictions
    result_pred_RF = clf_RF.predict(test_data_features)
    output = pd.DataFrame(data = {"Score": test["Score"], "sentiment":result_pred_RF})

    # # print output
    output.to_csv("bag_of_word_model.csv", index=False)

    print(accuracy_score(test["Score"], result_pred_RF)) #
    print(confusion_matrix(test["Score"], result_pred_RF))

    results.append(accuracy_score(test["Score"], result_pred_RF))
    para.append(i)

    # save the classifier
    with open('forest.pk1', 'wb') as fid:
        cPickle.dump(clf_RF, fid)

    ####################################################################################
    ##################### support vector machine #######################################

    from sklearn import svm

    #Use k-fold method to cross-validate whether overfitting happened
    #k_fold = KFold(len(train_data_features), n_folds=10, shuffle=True, random_state=0)

    clf_SVM = svm.SVC(gamma = 0.001,kernel = 'rbf',C=cd[i])
    model_SVM = clf_SVM.fit(train_data_features, train["Score"])

    result_pred_SVM = model_SVM.predict(test_data_features)
    print("Vector_size= {}".format(cd[i]))

    #Use k-fold method to cross-validate whether overfitting happened
    #print(np.average(cross_val_score(clf_SVM, train_data_features, train["Score"], cv=k_fold, n_jobs=1)))

    print(accuracy_score(test["Score"], result_pred_SVM)) #
    results.append(accuracy_score(test["Score"], result_pred_SVM))

    #Use k-fold method to cross-validate whether overfitting happened
    #results.append(np.average(cross_val_score(clf_SVM, train_data_features, train["Score"], cv=k_fold, n_jobs=1)))
    para.append(cd[i])

############################Plot drawing##############################
f = open("gamma.json", "w")
json.dump(results, f)
f.close()

f = open("gamma2.json", "w")
json.dump(para, f)
f.close()

#for i in range(0, 10, 1):
f = open("gamma.json", "r")
results = json.load(f)
f.close()
f = open("gamma2.json", "r")
para = json.load(f)
f.close()

fig = plt.figure()
ax = fig.add_subplot(111)
print("Plot= {}, {}".format(para, results))
for c, result in zip(para, results):
    #_, _, training_cost, _ = result
    ax.plot(c, result, 'd',
            label="vector size = " + str(c))
ax.set_xlim([-100, 1400])
#ax.set_ylim([0.78, 0.85])
ax.set_xlabel('n_estimators')
ax.set_ylabel('Prediction accuracy of Random Forest')
plt.legend(loc='upper right')
plt.show()


