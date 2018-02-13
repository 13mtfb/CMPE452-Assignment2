#include "stdafx.h"
#include "backpropagation.h"

NeuralNetwork::NeuralNetwork(int num_Input, int num_Hidden, int num_Output)
{
	numInput = num_Input;
	numHidden = num_Hidden;
	numOutput = num_Output;

	inputs = new double[numInput];
	ihWeights = MakeMatrix(numInput, numHidden);
	ihSums = new double[numHidden];
	ihBiases = new double[numHidden];
	ihOutputs = new double[numHidden];
	hoWeights = MakeMatrix(numHidden, numOutput);
	hoSums = new double[numOutput];
	hoBiases = new double[numOutput];
	outputs = new double[numOutput];

	oGrads = new double[numOutput];
	hGrads = new double[numHidden];

	ihPrevWeightsDelta = MakeMatrix(numInput, numHidden);
	ihPrevBiasesDelta = new double[numHidden];
	for (int i = 0; i < numHidden; i++)ihPrevBiasesDelta[i] = 0;
	hoPrevWeightsDelta = MakeMatrix(numHidden, numOutput);
	hoPrevBiasesDelta = new double[numOutput];
	for (int i = 0; i < numOutput; i++)hoPrevBiasesDelta[i] = 0;
}

//method uses the backpropagation algorigthm to update the weight/bias values
void NeuralNetwork::UpdateWeights(double *desiredOutput, double learningRate, double momentum)
{
	//find compute output gradients
	//this is the delta value found in the backpropagation notes
	for (int i = 0; i < numOutput; ++i)
	{
		//compute the derivative of sigmoid function
		double derivative = (1 - outputs[i]) * outputs[i]; //f'(a2j) 
		oGrads[i] = derivative * (desiredOutput[i] - outputs[i]); //δj = ej * f'(a2j)
	}

	//find hidden gradients
	for (int i = 0; i < numHidden; ++i)
	{
		double derivative = (1 - ihOutputs[i]) * ihOutputs[i]; // (1 / 1 + exp(-x))'  -- using output value of neuron
		double sum = 0.0;
		for (int j = 0; j < numOutput; ++j)
			sum += oGrads[j] * hoWeights[i][j]; 
		hGrads[i] = derivative * sum;
	}

	//Update input to hidden weights
	for (int i = 0; i < numInput; ++i)
	{
		for (int j = 0; j < numHidden; ++j)
		{
			double delta = learningRate * hGrads[j] * inputs[i]; // compute the new delta
			ihWeights[i][j] += delta; // update
			ihWeights[i][j] += momentum * ihPrevWeightsDelta[i][j]; //use momentum to keep going
		}
	}

	//Update input to hidden biases
	for (int i = 0; i < numHidden; ++i)
	{
		double delta = learningRate * hGrads[i] * 1.0;
		ihBiases[i] += delta;
		ihBiases[i] += momentum * ihPrevBiasesDelta[i];
	}

	//Update hidden to output weights
	for (int i = 0; i < numHidden; ++i)
	{
		for (int j = 0; j < numOutput; ++j)
		{
			double delta = learningRate * oGrads[j] * ihOutputs[i]; 
			hoWeights[i][j] += delta;
			hoWeights[i][j] += momentum * hoPrevWeightsDelta[i][j];
			hoPrevWeightsDelta[i][j] = delta;
		}
	}

	//Update hidden to output biases
	for (int i = 0; i < numOutput; ++i)
	{
		double delta = learningRate * oGrads[i] * 1.0;
		hoBiases[i] += delta;
		hoBiases[i] += momentum * hoPrevBiasesDelta[i];
		hoPrevBiasesDelta[i] = delta;
	}
} // UpdateWeights

void NeuralNetwork::SetWeights(double *weights)
{
	// copy weights and biases in weights[] array to i-h weights, i-h biases, h-o weights, h-o biases
	int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;

	int k = 0; // points into weights param

	for (int i = 0; i < numInput; ++i)
		for (int j = 0; j < numHidden; ++j) {
			ihWeights[i][j] = weights[k++];
			//std::cout << ihWeights[i][j] << std::endl;
		}


	for (int i = 0; i < numHidden; ++i){
		ihBiases[i] = weights[k++];
		//std::cout << ihBiases[i] << std::endl;
	}

	for (int i = 0; i < numHidden; ++i)
		for (int j = 0; j < numOutput; ++j) {
			hoWeights[i][j] = weights[k++];
			//std::cout << hoWeights[i][j] << std::endl;
		}

	for (int i = 0; i < numOutput; ++i){
		hoBiases[i] = weights[k++];
		//std::cout << hoBiases[i] << std::endl;
	}
}

double * NeuralNetwork::GetWeights()
{
	int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
	double *result  = new double[numWeights];
	int k = 0;
	for (int i = 0; i < numInput; ++i)
		for (int j = 0; j < numHidden; ++j)
			result[k++] = ihWeights[i][j];
	for (int i = 0; i < numHidden; ++i)
		result[k++] = ihBiases[i];
	for (int i = 0; i < numHidden; ++i)
		for (int j = 0; j < numOutput; ++j)
			result[k++] = hoWeights[i][j];
	for (int i = 0; i < numOutput; ++i)
		result[k++] = hoBiases[i];
	return result;
}

double ** NeuralNetwork::MakeMatrix(int rows, int columns)
{
	double **result = new double *[rows];
	for (int i = 0; i < rows; i++) {
		result[i] = new double[columns];
		for (int j = 0; j < columns; j++) {
			result[i][j] = 0;					//set array values to 0
		}
	}
	return result;
}

double * NeuralNetwork::ComputeOutputs(double *xValues)
{
	for (int i = 0; i < numHidden; ++i)
		ihSums[i] = 0.0;
	for (int i = 0; i < numOutput; ++i)
		hoSums[i] = 0.0;

	for (int i = 0; i < numInput; ++i) // copy x-values to inputs
		inputs[i] = xValues[i];

	for (int j = 0; j < numHidden; ++j)  // compute input-to-hidden weighted sums
		for (int i = 0; i < numInput; ++i){
			ihSums[j] += inputs[i] * ihWeights[i][j];
		}


	for (int i = 0; i < numHidden; ++i)  // add biases to input-to-hidden sums
		ihSums[i] += ihBiases[i];

	//std::cout << "ihOutputs: ";
	for (int i = 0; i < numHidden; ++i) {   // determine input-to-hidden output
		ihOutputs[i] = SigmoidFunction(ihSums[i]);
		//std::cout << ihOutputs[i] << " ";
	}
	//std::cout << std::endl;


	for (int j = 0; j < numOutput; ++j)   // compute hidden-to-output weighted sums
		for (int i = 0; i < numHidden; ++i)
			hoSums[j] += ihOutputs[i] * hoWeights[i][j];

	for (int i = 0; i < numOutput; ++i)  // add biases to input-to-hidden sums
		hoSums[i] += hoBiases[i];

	for (int i = 0; i < numOutput; ++i)   // determine hidden-to-output result
		outputs[i] = SigmoidFunction(hoSums[i]);

	return outputs;
} // ComputeOutputs