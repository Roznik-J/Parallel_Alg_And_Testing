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
	vector<string> outputFile;

	ofstream fileResult("resultThreads.txt");

	for (size_t i = 1; i < arguments.size(); i++) {
		string fileName = arguments.at(i);
		
		// build test case graph
		TestCase testCase(fileName);

		// Algorithms go here
		auto start = chrono::high_resolution_clock::now();
		
		runNonGpuMatrxMulti(testCase);
		
		auto stop = chrono::high_resolution_clock::now();
    	chrono::duration<double, std::milli> time = stop - start;
		double timeTaken = time.count();
		outputFile.push_back(fileName + ", " + to_string(timeTaken));
	}

	for (size_t i = 0; i < outputFile.size(); i++)
	{
		fileResult<<outputFile[i]<<endl;
	}

	fileResult << flush;
	fileResult.close();

	cout << "Done" << endl;
	return 0;
}
