                                     FALL2016 CS273 Course Project 

Project Title:
------------------------------
Sentiment analysis on anthropogenic climate change in the scientific literature using natural language processing


Project Authors:
------------------------------ 
Mengya Tao & Yingchun Du


Documentation
------------------------------
	This project submissions contain three sub-folders (code, data, figure) and two documents:

	code:
		main_NN.py #run neural networks models and run making plots
		NeuralNetwork.py #hand coded neural network algorithm to a function
		main_RF_SVM.py #run random forest and support vector machine models
		data_preprocessing.py #read in abstract and do data cleaning
		loading_data.py #load datafiles after data_preprocessing.py and transform to bag-of-words
		parameter_tunning #grid search method for tuning hyper-parameters in neural network
		training_data_clean.csv #cleaned training data (after data_preprocessing.py)
		validation_data_clean.csv #cleaned validation data (after data_preprocessing.py)
		test_data_clean.csv #cleaned test data(after data_preprocessing.py)
	
	data: 	
		training_data_5000.csv #raw training data, containing 5000 data points
		validation_data_1000.csv #raw validation data, containing 1000 data points
		test_data_clean.csv #raw test data, containing 960 data points
		training_data_clean.csv #cleaned training data (after data_preprocessing.py)
		validation_data_clean.csv #cleaned validation data (after data_preprocessing.py)
		test_data_clean.csv #cleaned test data(after data_preprocessing.py)
		multiple_eta.json #saved json data for learning rate selection plot
		multiple_feature.json #saved json data for feature selection plot
		multiple_lambda.json #saved json data for regularization term lambda selection plot
		multiple_neuron.json #saved json data for #of neurons selection plot
		nn_para.json #saved json data for best model parameters - weights and biases

	figure:
		accuracy.png
		featureSize.png
		lambda.png
		learning_rate.png
		neurons.png
	
	
	Report_MengyaTao_YingchunDu.pdf

	CS273_Project_Mengya_Yingchun.ppt


Contact
------------------------------
Mengya Tao: mengya@umail.ucsb.edu
Yingchun Du: ydu@umail.ucsb.edu