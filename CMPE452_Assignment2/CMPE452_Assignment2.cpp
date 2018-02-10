// CMPE452_Assignment2.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "wine.h"
#include "backpropagation.h"

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

	//preprocess the data somehow
	normalize(data);

	//assign train, validation and testing data

	//add backpropagation algorithm
	Brain b;

	double t_data[] = { 1.0,2.0, 3.0 };
	double t_ddata[] = { -0.25,0.14 };

	b.Initialize(3, 4, 2);

	b.GetDesiredOutput(t_ddata);
	b.GetInput(t_data);


	b.Process();
	b.BP();
	b.Process();
	b.Process();
	b.Process();
	b.Process();
	b.Process();


    return 0;
}

