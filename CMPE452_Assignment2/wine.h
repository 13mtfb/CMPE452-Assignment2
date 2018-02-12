#pragma once
#include "stdafx.h"

//contains structs and functions for the wine dataset

struct wine {
	//input features
	double input[11];
	int quality;
	//desired output
	//output array is structured by quality as follows:
	//5: {1,0,0,0}
	//6: {0,1,0,0}
	//7: {0,0,1,0}
	//8: {0,0,0,1}
	double output[4];
};

//function declarations
wine parseInput(std::string);
void printWine(wine);
void normalize(std::vector <wine>&);