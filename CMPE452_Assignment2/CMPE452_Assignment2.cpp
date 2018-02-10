// CMPE452_Assignment2.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

using namespace std;

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


    return 0;
}

