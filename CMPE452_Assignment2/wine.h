#pragma once

//contains structs and functions for the wine dataset

struct wine {
	//input features
	double input[11];
	//desired output
	int quality;
};

//function declarations
wine parseInput(std::string);
void printWine(wine);