#include "TestCase.h"
#include "NonGpuAlgorithms/MatrixMultiplication.h"

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>

using namespace std;

int main(int argc, char * argv[]) {
	vector<string> arguments(argv, argv + argc);
	vector<TestCase> allTests;
	vector<string> Output;

	ofstream fileResult("result.txt");

	for (size_t i = 1; i < arguments.size(); i++) {
		string fileName = arguments.at(i);
		
		// build test case graph
		TestCase testCase(fileName);

		// Algorithms go here
		// runNonGpuMatrxMulti(testCase);

		// This is how we'd take time measure.
		// It does NOT need to be hear come final version.
		auto start = chrono::high_resolution_clock::now();
		auto stop = chrono::high_resolution_clock::now();
    	chrono::duration<double, std::milli> time = stop - start;
		double timeTaken = time.count();

		Output.push_back(fileName + " " + to_string(testCase.printSum()) + " " + to_string(timeTaken));
	}

	for (size_t i = 0; i < Output.size(); i++)
	{
		fileResult<<Output[i]<<endl;
	}

	fileResult << flush;
	fileResult.close();

	cout << "done" << endl;
	return 0;
}

/*
Test cases:
	Single source shortest path
	Single destination shortest path
	All pairs shortest path 

	Sparse graph
	Dense graph
*/
