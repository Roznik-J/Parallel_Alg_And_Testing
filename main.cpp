#include "TestCase.h"
#include "NonGpuAlgorithms/MatrixMultiplication.h"

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>
#include <sstream>

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

		int lnNumTriangles = 0;

		// Algorithms go here
		auto start = chrono::high_resolution_clock::now();

		/*
		fileResult << "fileName" << "," << fileName << ",";
        fileResult << "NumNodes" << "," << testCase.GetNodeSize() << ",";
        fileResult << "NumTriangles" << "," << testCase.GetNumTriangles() << ",";
        fileResult << "ProgramTime" <<","<<lcTimingInformation.at(5) << ",";
        fileResult <<"\n";
		*/
		
		lnNumTriangles = runNonGpuMatrxMulti(testCase);
		auto stop = chrono::high_resolution_clock::now();
    	chrono::duration<double, std::milli> time = stop - start;
		double timeTaken = time.count();
		//outputFile.push_back("fileName" + "," + fileName + "," + "NumNodes" + "," + to_string(testCase.getNodeSize()) + "," + "NumTriangles" + "," + lnNumTriangles + + "," + "ProgramTime" + "," + to_string(timeTaken));
		
		std::stringstream lcStr;
		lcStr << "fileName" << "," << fileName << ",";
        lcStr << "NumNodes" << "," << testCase.getNodeSize() << ",";
        lcStr << "NumTriangles" << "," << lnNumTriangles << ",";
        lcStr << "ProgramTime" <<","<<timeTaken << ",";

		outputFile.push_back(lcStr.str());
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
