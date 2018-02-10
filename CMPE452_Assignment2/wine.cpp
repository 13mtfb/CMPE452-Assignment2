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
			input.input[i] = value;
		}
		//access quality value by removing quotes
		else {
			quality = subString.substr(subString.find('"')+1, 1);
			value = atof(quality.c_str());
			input.quality = value;
		}
	}
	return input;
}

//print a wine structure verbosely
void printWine(wine print) {
	std::cout << "fixed acidity: " << print.input[0];
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
	std::cout << "quality: " << print.quality << std::endl;
}

//normalize each input feature on the wine dataset into the range [0 1]
//normalization calculated as follows:
// normalized = x - min(x) / ( max(x) - min(x) )
void normalize(std::vector<wine> &w) {

	double min_element[11];
	double max_element[11];

	//determine the max and min of the dataset for each feature
	for (int f = 0; f < 11; f++) {
		wine dummy;
		dummy = *std::min_element(w.begin(), w.end(), [&](wine e1, wine e2) {return e1.input[f] < e2.input[f]; });
		min_element[f] = dummy.input[f];
		dummy = *std::max_element(w.begin(), w.end(), [&](wine e1, wine e2) {return e1.input[f] < e2.input[f]; });
		max_element[f] = dummy.input[f];
	}

	std::cout << min_element[0] << " " << max_element[0] << std::endl;
	std::cout << min_element[10] << " " << max_element[10] << std::endl;


	//iterate over each element
	for (int e = 0; e < w.size(); e++) {
		//iterate over each feature
		for (int f = 0; f < 11; f++) {
			w[e].input[f] = (w[e].input[f] - min_element[f]) / (max_element[f] - min_element[f]);
		}
	}
}