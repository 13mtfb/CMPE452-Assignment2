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
	}
	//close the file stream
	input.close();

	//preprocess the data somehow
	normalize(data);

	//define the number of outputs (output neurons)
	int num_outputs = 2;
	const int num_weights = 26;

	//assign train, validation and testing data

	//add backpropagation algorithm
	NeuralNetwork nn(3, 4, 2);
	
	//arbitrary weights and biases
	double weights[num_weights]={
		0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
			-2.0, -6.0, -1.0, -7.0,
			1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
			-2.5, -5.0 };

	cout << "Initial 26 random weights and biases are:" << endl;
	ShowVector(weights, num_weights);
	
	cout << "\nLoading neural network weights and biases" << endl;
	nn.SetWeights(weights);

	cout << "\nSetting inputs:" << endl;
	double xValues[3] = { 1.0, 2.0, 3.0 };
	ShowVector(xValues, 3);

	double * initialOutputs = nn.ComputeOutputs(xValues);
	cout << "Initial outputs:" << endl;
	ShowVector(initialOutputs, 2);
	
	// target (desired) values. note these only make sense for tanh output activation
	double tValues[2] = { -0.8500, 0.7500 }; 
	cout << "Target outputs to learn are:" << endl;
	ShowVector(tValues, 2);

	double eta = 0.90;  // learning rate - controls the maginitude of the increase in the change in weights. found by trial and error.
	double alpha = 0.04; // momentum - to discourage oscillation. found by trial and error.
	cout << "Setting learning rate (eta) = " << eta << " and momentum (alpha) = " << alpha << endl;

	cout << "\nEntering main back-propagation compute-update cycle" << endl;
	cout << "Stopping when sum absolute error <= 0.01 or 1,000 iterations\n" << endl;
	int ctr = 0;
	double *yValues = nn.ComputeOutputs(xValues); // prime the back-propagation loop
	double error = Error(tValues, yValues, num_outputs);

	while (ctr < 1000 && error > 0.01)
	{
		cout << "===================================================" <<endl;
		cout << "iteration = " << ctr << endl;
		cout << "Updating weights and biases using back-propagation" << endl;
		nn.UpdateWeights(tValues, eta, alpha);
		cout << "Computing new outputs:" << endl;
		yValues = nn.ComputeOutputs(xValues);
		ShowVector(yValues, 2);
		cout << "\nComputing new error" << endl;
		error = Error(tValues, yValues, num_outputs);
		cout << "Error = " << error << endl;
		++ctr;
	}
	cout << "===================================================" << endl;
	cout << "\nBest weights and biases found:" << endl;
	double *bestWeights = nn.GetWeights();
	ShowVector(bestWeights, num_weights);

	cout << "End Neural Network Back-Propagation demo\n" << endl;

    return 0;
}