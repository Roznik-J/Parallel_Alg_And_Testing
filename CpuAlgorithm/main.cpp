#include "TestCase.hpp"
#include "MatrixMultiplication.hpp"

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>
#include <sstream>
#include <ctime>
#include <iomanip>

using namespace std;

int main(int argc, char * argv[]) {
	vector<string> arguments(argv, argv + argc);
	vector<TestCase> allTests;
	vector<string> outputFile;

	std::time_t now = std::time(nullptr);
    std::tm* timeinfo = std::localtime(&now);
    
    std::ostringstream oss;
    oss << "resultThreads";
    oss << std::setfill('0') << std::setw(2) << timeinfo->tm_mon + 1 << '-'
        << std::setw(2) << timeinfo->tm_mday << '-'
        << std::setw(2) << (timeinfo->tm_year + 1900) % 100 << '_'
        << std::setw(2) << timeinfo->tm_hour << ':'
        << std::setw(2) << timeinfo->tm_min << ":"
        << std::setw(2) << timeinfo->tm_sec;
    oss << ".txt";

	ofstream fileResult(oss.str().c_str());

	for (size_t i = 1; i < arguments.size(); i++) {
		string fileName = arguments.at(i);

		std::cout << "running : " << fileName << std::endl;
		
		// build test case graph
		TestCase testCase(fileName);

		std::cout << "DoneWith : " << fileName << std::endl;

		int lnNumTriangles = 0;

		// Algorithms go here
		auto start = chrono::high_resolution_clock::now();
		
		lnNumTriangles = runNonGpuMatrxMulti(testCase);
		auto stop = chrono::high_resolution_clock::now();
    	chrono::duration<double, std::milli> time = stop - start;
		double timeTaken = time.count();
		
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
