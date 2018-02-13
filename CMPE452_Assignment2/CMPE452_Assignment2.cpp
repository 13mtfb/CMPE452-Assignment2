// CMPE452_Assignment2.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "wine.h"
#include "backpropagation.h"

using namespace std;

//calculates the absolute error between the desired output vector and the actual output vector
double Error(double target[], double output[], int num_outputs)
{
	double sum = 0.0;
	for (int i = 0; i < num_outputs; ++i)
		sum += abs(target[i] - output[i]);
	return sum;
}

void ShowVector(double *arr, int num)
{
	cout << std::setprecision(4) << fixed;
	for (int i = 0; i < num; ++i)
	{
		if (i != (num - 1))
			cout << arr[i] << "\t";
		else
			cout << arr[i];
		if ( (i+1) % 10 == 0){
			cout<<endl;
		}
	}
	cout << endl;
}

void dataSetStats(std::vector <wine>& w, int numTotal) {

	int numPoints = w.size();
	int numEachQuality[3] = {0,0,0};

	for (int i = 0; i < w.size(); i++) {
		if (w[i].quality == 5) { numEachQuality[0]++; }
		if (w[i].quality == 7) { numEachQuality[1]++; }
		if (w[i].quality == 8) { numEachQuality[2]++; }
	}

	cout << "Number of points in dataset: " << numPoints << " (" << (double(numPoints)) / (double(numTotal)) * 100 << "% of total)" << endl;
	cout << "\t [5]: " << numEachQuality[0] << " [7]: " << numEachQuality[1] << " [8]: " << numEachQuality[2] << endl;

}


int main()
{
	//create a vector of wine structs
	vector <wine> testing;
	vector <wine> training;
	int num_samples = 0;
	int numEachQuality[4] = { 0,0,0,0};
	int countEven[3] = { 0,0,0 };
	bool filled = false;
	//declare an input filestream to read data
	ifstream input;
	//Create an output filestream to write data
	ofstream output("output.txt");
	//open the file stream
	input.open("assignment2data.csv");
	//check for errors opening file
	if (!input) {
		cerr << "Unable to open file" << endl;
		exit(1);
	}
	//iterate through the train file, line-by-line
	for (string line; getline(input, line); ) {
		//create iris struct to push to vector
		wine addData;
		//parse string line
		addData = parseInput(line);

		//push the struct to the apprporiate vector
		if (num_samples % 4 == 0) {
			testing.push_back(addData);
		}//push back every 4th point to testing dataset
		else {
			training.push_back(addData);
		}

		num_samples++;
		//count number of samples at each quality
		numEachQuality[addData.quality - 5]++;
	}

	//close the file stream
	input.close();

	//show statistics about the dataset
	cout << "Total of datapoints: " << num_samples << endl;
	cout << "\t [5]: " << numEachQuality[0] << " [6]: " << numEachQuality[1] << " [7]: " << numEachQuality[2] << " [8]: " << numEachQuality[3] << endl;


	//print stats about each dataset
	cout << "Training DataSet: " << endl;
	dataSetStats(training, num_samples);

	//make a copy of the testing vector for later output
	vector <wine> print;
	print = testing;


	//preprocess the data somehow
	normalize(testing);
	normalize(training);

	//define which features to remove
	//1 - keep, 0 - remove
	int features[11] = { 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0 };

	//remove features above
	removeFeatures(testing, features);
	removeFeatures(training, features);

	//define the number of neurons
	const int num_input = 6;
	const int num_hidden = 5;
	const int num_output = 3;
	const int num_weights = num_input * num_hidden + num_hidden * num_output + num_hidden + num_output;

	//assign train, validation and testing data

	//add backpropagation algorithm
	NeuralNetwork nn(num_input, num_hidden, num_output);
	
	//define weights/biases array
	double weights[num_weights];
	
	//set hidden weights
	for (int i = 0; i < num_weights; i++) {
		if (i < num_input * num_hidden)
			weights[i] = 0.25 + 0.01*i;
		else if (i < num_input * num_hidden + num_hidden)
			weights[i] = -0.5;
		else if (i < num_input * num_hidden + num_hidden + num_hidden * num_output)
			weights[i] = 0.25 + 0.01*i;
		else
			weights[i] = -0.5;
	}


	cout << "Initial random weights and biases are:" << endl;
	ShowVector(weights, num_weights);
	
	cout << "\nLoading neural network weights and biases" << endl;
	nn.SetWeights(weights);

	double learningRate = 1.1;  // learning rate
	double momentum = 0.01; // momentum
	cout << "Setting learning rate (eta) = " << learningRate << " and momentum (alpha) = " << momentum << endl;

	cout << "\nEntering main back-propagation compute-update cycle" << endl;
	cout << "Stopping when sum absolute error <= 0.90 or 2000 iterations\n" << endl;
	int iterations = 0;
	double *yValues = nn.ComputeOutputs(training[0].input); // prime the back-propagation loop
	double error = 3000;//set high error in order to avoid premature loop ending
	while (iterations < 2000 && (error/training.size())> 0.1)
	{
		//cout << "===================================================" << endl;
		//cout << "iteration = " << iterations << endl;
		error = 0;
		for (int i = 0; i < training.size(); i++) {
			nn.UpdateWeights(training[i].output, learningRate, momentum);
			//cout << "Computing new outputs:" << endl;
			yValues = nn.ComputeOutputs(training[i].input);
			//ShowVector(training[i].output, num_output);
			//ShowVector(yValues, num_output);
			//cout << "\nComputing new error" << endl;
			error += Error(training[i].output, yValues, num_output);
		}
		//cout << "Error = " << error / training.size() << endl;
		//cout << "===================================================" << endl;
		++iterations;
	}
	cout << "Number of iterations: " << iterations << endl;
	cout << "Error = " << error / training.size() << endl;
	cout << "\nBest weights and biases found:" << endl;
	double *bestWeights = nn.GetWeights();
	ShowVector(bestWeights, num_weights);

	/////////////////////////////////////////////////////////////////////////////
	////////////////////////Training Complete////////////////////////////////////

	cout << "Testing DataSet: " << endl;
	dataSetStats(testing, num_samples);

	//the output is determined by which of the categorical output neurons have the highest output function
	//possible outputs for quality are 5, 7, 8
	double highest = 0; 
	int highest_index = 0;
	int numClassified = 0;
	int numSuccess = 0;
	int numError = 0;
	double numSuccessEach[3] = { 0, 0, 0 };//records successfull classifications of each testing point for each wine quality
	double numFalsePositive[3] = { 0, 0, 0 };	//records unsuccessfull classifications of each testing point for each quality
	double numFalseNegative[3] = { 0,0,0 }; //records instances where wine was not predicted but was the desired output
	int wineGuess = 0;

	for (int i = 0; i < testing.size(); i++) {
		yValues = nn.ComputeOutputs(testing[i].input);
		//cout << "Sample Number: " << i << endl;
		highest = -1;
		highest_index = -1;
		for (int j = 0; j < num_output; j++) {
			if (yValues[j] > highest) { highest = yValues[j]; highest_index = j; }
		cout << yValues[j] << " ";
		}
		//cout << "\t" << testing[i].quality << endl;
		//currently highest_index holds the quality value index determined by the neural network
		if (highest_index==0){//network is predicting a wine quality of 5
			wineGuess = 5;
		}
		if (highest_index == 1) {//network is predicting a wine quality of 7
			wineGuess = 7;

		}
		if (highest_index == 2) {//network is predicting a wine quality of 8
			wineGuess = 8;
		}
		printWine(print[i]);
		cout << wineGuess << endl;
		if (wineGuess == testing[i].quality) { 
			numSuccess++; 
			//cout << " success" << endl; 
			if (testing[i].quality == 5) { numSuccessEach[0]++; }
			if (testing[i].quality == 7) { numSuccessEach[1]++; }
			if (testing[i].quality == 8) { numSuccessEach[2]++; }
		}
		else { 
			numError++; 
			//cout << " failure" << endl; 
			if (wineGuess == 5) { numFalsePositive[0]++; }
			if (wineGuess == 5) { numFalsePositive[0]++; }
			if (wineGuess == 5) { numFalsePositive[0]++; }
			if (testing[i].quality == 5) { numFalseNegative[0]++; }
			if (testing[i].quality == 7) { numFalseNegative[1]++; }
			if (testing[i].quality == 8) { numFalseNegative[2]++; }
		}
		numClassified++;
	}

	cout << "\nNumber of classifications: " << numClassified << endl;
	cout << "Successfull classifications: " << numSuccess << endl;
	cout << "unSuccessfull classifications: " << numError << endl;
	//Calculate precision and recall for each class
	cout << "Precision: " << endl;
	cout << "[5]: " << numSuccessEach[0] / (numSuccessEach[0] + numFalsePositive[0]) << endl;
	cout << "[7]: " << numSuccessEach[1] / (numSuccessEach[1] + numFalsePositive[1]) << endl;
	cout << "[8]: " << numSuccessEach[2] / (numSuccessEach[2] + numFalsePositive[2]) << endl;
	cout << "Recall: " << endl;
	cout << "[5]: " << numSuccessEach[0] / (numSuccessEach[0] + numFalseNegative[0]) << endl;
	cout << "[7]: " << numSuccessEach[1] / (numSuccessEach[1] + numFalseNegative[1]) << endl;
	cout << "[8]: " << numSuccessEach[2] / (numSuccessEach[2] + numFalseNegative[2]) << endl;

    return 0;
}