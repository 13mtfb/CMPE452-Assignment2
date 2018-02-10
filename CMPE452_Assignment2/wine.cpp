#pragma once
#include "stdafx.h"

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
	std::cout << " chlorides: " << print.input[3];
	std::cout << " free sulfur dioxide: " << print.input[4];
	std::cout << " total sulfur dioxide: " << print.input[5];
	std::cout << " density: " << print.input[6];
	std::cout << " pH: " << print.input[7];
	std::cout << " sulphates: " << print.input[8];
	std::cout << " alcohol: " << print.input[9];
	std::cout << " fixed acidity: " << print.input[10];
	std::cout << std::endl;
	std::cout << "quality: " << print.quality << std::endl;
}