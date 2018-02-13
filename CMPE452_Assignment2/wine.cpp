#pragma once
#include "stdafx.h"
#include "wine.h"

//assumes eleven double values separated by a single comma:
//"fixed acidity","volatile acidity","citric acid","residual sugar","chlorides",
//"free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol",
//followed by a string indicating indicating the quality
wine parseInput(std::string line) {

	//iris return
	wine input;

	//substring to parse csv
	std::string subString;
	std::string quality;
	//values for string find
	int firstPos = 0;
	int subPos;
	double value;

	//parse float values
	for (int i = 0; i < 12; i++) {
		subPos = line.find(",", firstPos);
		subString = line.substr(firstPos, subPos);
		firstPos = subPos + 1;

		if (i<11){
			//cast string value to double
			value = atof(subString.c_str());
			//save double value in array
			//can comment out individual lines in order to remove features from the neural network inputs
			//requires a change in the array size in wine.h and the num_inputs variable in main()
			input.input[i] = value;
		}
		//access quality value by removing quotes
		else {
			quality = subString.substr(subString.find('"')+1, 1);
			value = atof(quality.c_str());
			input.quality = value;
				if (value == 5) { input.output[0] = 0.95; }
				else { input.output[0] = 0.05; }
				if (value == 7) { input.output[1] = 0.95; }
				else { input.output[1] = 0.05; }
				if (value == 8) { input.output[2] = 0.95; }
				else { input.output[2] = 0.05; }
		}
	}
	return input;
}

//print a wine structure verbosely
void printWine(wine print) {
	/*std::cout << "fixed acidity: " << print.input[0];
	std::cout << " volatile acidity: " << print.input[1];
	std::cout << " citric acid: " << print.input[2];
	std::cout << " residual sugar " << print.input[3];
	std::cout << " chlorides: " << print.input[4];
	std::cout << " free sulfur dioxide: " << print.input[5];
	std::cout << " total sulfur dioxide: " << print.input[6];
	std::cout << " density: " << print.input[7];
	std::cout << " pH: " << print.input[8];
	std::cout << " sulphates: " << print.input[9];
	std::cout << " alcohol: " << print.input[10];
	std::cout << std::endl;
	std::cout << "quality: " << print.quality << std::endl;\*/

	std::setprecision(2); 
	std::fixed;

	for (int i = 0; i < 11; i++) {
		std::cout << print.input[i] << "\t";
	}
	std::cout << print.quality << "\t";
}

//normalize each input feature on the wine dataset into the range [0 1]
//normalization calculated as follows:
// normalized = x - min(x) / ( max(x) - min(x) )
void normalize(std::vector<wine> &w) {

	double min_element[11];
	double max_element[11];
	double min_quality, max_quality;

	//determine the max and min of the dataset for each feature
	for (int f = 0; f < 11; f++) {
		wine dummy;
		dummy = *std::min_element(w.begin(), w.end(), [&](wine e1, wine e2) {return e1.input[f] < e2.input[f]; });
		min_element[f] = dummy.input[f];
		dummy = *std::max_element(w.begin(), w.end(), [&](wine e1, wine e2) {return e1.input[f] < e2.input[f]; });
		max_element[f] = dummy.input[f];
	}

	//calculate the min and max quality to determine the number of output neurons needed to classify the data
	wine dummy;
	dummy = *std::min_element(w.begin(), w.end(), [&](wine e1, wine e2) {return e1.quality < e2.quality; });
	min_quality = dummy.quality;
	dummy = *std::max_element(w.begin(), w.end(), [&](wine e1, wine e2) {return e1.quality < e2.quality; });
	max_quality = dummy.quality;

	/*
	std::cout << "fixed acidity: " << min_element[0] << " - " << max_element[0] << std::endl;
	std::cout << " volatile acidity: " << min_element[1] << " - " << max_element[1] << std::endl;
	std::cout << " citric acid: " << min_element[2] << " - " << max_element[2] << std::endl;
	std::cout << " residual sugar " << min_element[3] << " - " << max_element[3] << std::endl;
	std::cout << " chlorides: " << min_element[4] << " - " << max_element[4] << std::endl;
	std::cout << " free sulfur dioxide: " << min_element[5] << " - " << max_element[5] << std::endl;
	std::cout << " total sulfur dioxide: " << min_element[6] << " - " << max_element[6] << std::endl;
	std::cout << " density: " << min_element[7] << " - " << max_element[7] << std::endl;
	std::cout << " pH: " << min_element[8] << " - " << max_element[8] << std::endl;
	std::cout << " sulphates: " << min_element[9] << " - " << max_element[9] << std::endl;
	std::cout << " alcohol: " << min_element[10] << " - " << max_element[10] << std::endl;
	std::cout << "quality: " << min_quality << " - " << max_quality << std::endl;
	*/


	//iterate over each element
	for (int e = 0; e < w.size(); e++) {
		//iterate over each feature
		for (int f = 0; f < 11; f++) {
			w[e].input[f] = (w[e].input[f] - min_element[f]) / (max_element[f] - min_element[f]);
		}
	}
}

//function is called with an array of ints which is indicates which features should be kept(1) or removed(0)
//from the feature list
//As arrays are used, but fixed, data is shifted back and 0s are replaced at the end
void removeFeatures(std::vector <wine>& w, int* index) {
	int k;
	double temp[11] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

	for (int i = 0; i < w.size(); i++){
		k = 0;
		for (int j = 0; j < 11; j++) {
			if (index[j] == 1) { temp[k] = w[i].input[j]; k++; }
		}
		for (int j = 0; j < 11; j++) {
			if (j < k) { w[i].input[j] = temp[j];}
			else { w[i].input[j] = 0; }
		}
	}

}