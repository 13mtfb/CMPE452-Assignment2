// CMPE452_Assignment2.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "wine.h"
#include "backpropagation.h"

using namespace std;

double Error(double target[], double output[], int num_outputs) // sum absolute error. could put into NeuralNetwork class.
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

int main()
{
	//create a vector of wine structs
	vector <wine> data;
	int num_samples = 0;
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
		//push the struct to the vector
		data.push_back(addData);
		num_samples++;
	}
	//close the file stream
	input.close();

	//preprocess the data somehow
	normalize(data);

	//define the number of neurons
	const int num_input = 11;
	const int num_hidden = 4;
	const int num_output = 4;
	const int num_weights = num_input * num_hidden + num_hidden * num_output + num_hidden + num_output;

	//assign train, validation and testing data

	//add backpropagation algorithm
	NeuralNetwork nn(num_input, num_hidden, num_output);
	
	//define weights/biases array
	double weights[num_weights];
	
	//set hidden weights
	for (int i = 0; i < num_weights; i++) {
		if (i < num_input * num_hidden)
			weights[i] = 0.5;
		else if (i < num_input * num_hidden + num_hidden)
			weights[i] = -0.5;
		else if (i < num_input * num_hidden + num_hidden + num_hidden * num_output)
			weights[i] = 0.5;
		else
			weights[i] = -0.5;
	}


	cout << "Initial random weights and biases are:" << endl;
	ShowVector(weights, num_weights);
	
	cout << "\nLoading neural network weights and biases" << endl;
	nn.SetWeights(weights);

	cout << "\nSetting inputs:" << endl;
	//double xValues[3] = { 1.0, 2.0, 3.0 };
	ShowVector(data[0].input, num_input);

	double * initialOutputs = nn.ComputeOutputs(data[0].input);
	cout << "Initial outputs:" << endl;
	ShowVector(initialOutputs, num_output);
	
	// target (desired) values. note these only make sense for tanh output activation
	//double tValues[2] = { -0.8500, 0.7500 }; 
	cout << "Target outputs to learn are:" << endl;
	ShowVector(data[0].output, num_output);

	double eta = 0.90;  // learning rate - controls the maginitude of the increase in the change in weights. found by trial and error.
	double alpha = 0.04; // momentum - to discourage oscillation. found by trial and error.
	cout << "Setting learning rate (eta) = " << eta << " and momentum (alpha) = " << alpha << endl;

	cout << "\nEntering main back-propagation compute-update cycle" << endl;
	cout << "Stopping when sum absolute error <= 0.01 or 1,000 iterations\n" << endl;
	int iterations = 0;
	int num_training = 1000;
	double *yValues = nn.ComputeOutputs(data[0].input); // prime the back-propagation loop
	double error = Error(data[0].output, yValues, num_output);

	while (iterations < 1000 && error > 0.1)
	{
		cout << "===================================================" << endl;
		cout << "iteration = " << iterations << endl;
		error = 0;
		for (int i = 0; i < num_training; i++) {
			//cout << "Updating weights and biases using back-propagation" << endl;
			nn.UpdateWeights(data[i].output, eta, alpha);
			//cout << "Computing new outputs:" << endl;
			yValues = nn.ComputeOutputs(data[i].input);
			//ShowVector(yValues, num_output);
			//cout << "\nComputing new error" << endl;
			error += Error(data[i].output, yValues, num_output);
		}
		cout << "Error = " << error/num_training << endl;
		cout << "===================================================" << endl;
		++iterations;
	}
	cout << "\nBest weights and biases found:" << endl;
	double *bestWeights = nn.GetWeights();
	ShowVector(bestWeights, num_weights);

	cout << "End Neural Network Back-Propagation demo\n" << endl;

    return 0;
}