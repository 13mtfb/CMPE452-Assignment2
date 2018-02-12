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

void NeuralNetwork::UpdateWeights(double *tValues, double eta, double alpha)
{
	// 1. compute output gradients
	for (int i = 0; i < numOutput; ++i)
	{
		double derivative = (1 - outputs[i]) * (1 + outputs[i]); // derivative of tanh
		oGrads[i] = derivative * (tValues[i] - outputs[i]);
	}

	// 2. compute hidden gradients
	for (int i = 0; i < numHidden; ++i)
	{
		double derivative = (1 - ihOutputs[i]) * ihOutputs[i]; // (1 / 1 + exp(-x))'  -- using output value of neuron
		double sum = 0.0;
		for (int j = 0; j < numOutput; ++j) // each hidden delta is the sum of numOutput terms
			sum += oGrads[j] * hoWeights[i][j]; // each downstream gradient * outgoing weight
		hGrads[i] = derivative * sum;
	}

	// 3. update input to hidden weights (gradients must be computed right-to-left but weights can be updated in any order
	for (int i = 0; i < numInput; ++i) // 0..2 (3)
	{
		for (int j = 0; j < numHidden; ++j) // 0..3 (4)
		{
			double delta = eta * hGrads[j] * inputs[i]; // compute the new delta
			ihWeights[i][j] += delta; // update
			ihWeights[i][j] += alpha * ihPrevWeightsDelta[i][j]; // add momentum using previous delta. on first pass old value will be 0.0 but that's OK.
		}
	}

	// 3b. update input to hidden biases
	for (int i = 0; i < numHidden; ++i)
	{
		double delta = eta * hGrads[i] * 1.0; // the 1.0 is the constant input for any bias; could leave out
		ihBiases[i] += delta;
		ihBiases[i] += alpha * ihPrevBiasesDelta[i];
	}

	// 4. update hidden to output weights
	for (int i = 0; i < numHidden; ++i)  // 0..3 (4)
	{
		for (int j = 0; j < numOutput; ++j) // 0..1 (2)
		{
			double delta = eta * oGrads[j] * ihOutputs[i];  // see above: ihOutputs are inputs to next layer
			hoWeights[i][j] += delta;
			hoWeights[i][j] += alpha * hoPrevWeightsDelta[i][j];
			hoPrevWeightsDelta[i][j] = delta;
		}
	}

	// 4b. update hidden to output biases
	for (int i = 0; i < numOutput; ++i)
	{
		double delta = eta * oGrads[i] * 1.0;
		hoBiases[i] += delta;
		hoBiases[i] += alpha * hoPrevBiasesDelta[i];
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
		for (int i = 0; i < numInput; ++i)
			ihSums[j] += inputs[i] * ihWeights[i][j];

	for (int i = 0; i < numHidden; ++i)  // add biases to input-to-hidden sums
		ihSums[i] += ihBiases[i];

	for (int i = 0; i < numHidden; ++i)   // determine input-to-hidden output
		ihOutputs[i] = SigmoidFunction(ihSums[i]);

	for (int j = 0; j < numOutput; ++j)   // compute hidden-to-output weighted sums
		for (int i = 0; i < numHidden; ++i)
			hoSums[j] += ihOutputs[i] * hoWeights[i][j];

	for (int i = 0; i < numOutput; ++i)  // add biases to input-to-hidden sums
		hoSums[i] += hoBiases[i];

	for (int i = 0; i < numOutput; ++i)   // determine hidden-to-output result
		outputs[i] = HyperTanFunction(hoSums[i]);

	return outputs;
} // ComputeOutputs