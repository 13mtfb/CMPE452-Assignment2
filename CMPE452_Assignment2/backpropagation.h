#pragma once
#include "stdafx.h"

class NeuralNetwork
{
private:
	int numInput;
	int numHidden;
	int numOutput;

	double *inputs;
	double **ihWeights; // input-to-hidden
	double *ihSums;
	double *ihBiases;
	double *ihOutputs;

	double **hoWeights;  // hidden-to-output
	double *hoSums;
	double *hoBiases;
	double *outputs;

	double *oGrads; // output gradients for back-propagation
	double *hGrads; // hidden gradients for back-propagation

	double **ihPrevWeightsDelta;  // for momentum with back-propagation
	double *ihPrevBiasesDelta;

	double **hoPrevWeightsDelta;
	double *hoPrevBiasesDelta;

	static double StepFunction(double x) // an activation function that isn't compatible with back-propagation bcause it isn't differentiable
	{
		if (x > 0.0) return 1.0;
		else return 0.0;
	}

	static double SigmoidFunction(double x)
	{
		if (x < -45.0) return 0.0;
		else if (x > 45.0) return 1.0;
		else return 1.0 / (1.0 + exp(-x));
	}

	static double HyperTanFunction(double x)
	{
		if (x < -10.0) return -1.0;
		else if (x > 10.0) return 1.0;
		else return tanh(x);
	}

public:
	//constructor definition
	NeuralNetwork(int, int, int);
	// update the weights and biases using back-propagation, with target values, eta (learning rate), alpha (momentum)
	void UpdateWeights(double[], double, double);
	void SetWeights(double *);
	double *GetWeights();
	double *ComputeOutputs(double *);
	double **MakeMatrix(int, int);
};//end neuralNetwork class