#include "../TestCase.h"
#include <thread>
#include <iostream>
#include <vector>
using namespace std;

void multiplyMatrix(const vector<int>* row, const vector<vector<int>>* matrix2, vector<int>* result) {
    for(size_t i = 0; i < matrix2->size(); i++) {
        int sum = 0;
        for(size_t j= 0; j < row->size(); j++) {
            sum += row->at(j) * matrix2->at(j).at(i);
        }
        result->at(i) = sum;
    }
}

void findMatrixCubed(const vector<vector<int>>& matrix, vector<vector<int>>& result) {
    int nodeNum = matrix.size();
    vector<vector<int>> intermediateResult(nodeNum);
    for(vector<int>& row : intermediateResult) {
        row.resize(nodeNum);
    }

    // A x A
    thread threadList[nodeNum];
    for(int i = 0; i < nodeNum; i++) {
        threadList[i] = thread(multiplyMatrix, &matrix[i], &matrix, &intermediateResult[i]);
    }
    for(int i = 0; i < nodeNum; i++) {
        threadList[i].join();
    }

    // A^2 x A
    for(int i = 0; i < nodeNum; i++) {
        threadList[i] = thread(multiplyMatrix, &intermediateResult[i], &matrix, &result[i]);
    }
    for(int i = 0; i < nodeNum; i++) {
        threadList[i].join();
    }
}

int runNonGpuMatrxMulti(const TestCase& graph) {
	int nodeNum = graph.adjMatrix.size();

    // prep result array
    vector<vector<int>> result(nodeNum);
    for(vector<int>& row : result) {
        row.resize(nodeNum);
    }

    // prep adjacency matrix
    vector<vector<int>> input(nodeNum);
    for(vector<int>& row : input) {
        row.resize(nodeNum);
    }

    findMatrixCubed(graph.adjMatrix, result);

    // sum up diagonals and divide by 6 for true triangle
    // number
    int triangleCount = 0;
    for(size_t i = 0; i < input.size(); i++) {
        triangleCount += result[i][i];
    }

    if((triangleCount % 6) != 0) {
        cout << "WARNING: parallel matrix multiplication triangle count recieved a weird number." << endl;
    }
    return triangleCount / 6;
}
