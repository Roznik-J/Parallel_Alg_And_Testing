#include "../TestCase.h"
#include "../Edge.h"
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
    // REMOVE LATER
    vector<int> testing(16);
    testing.resize(4);
    testing[0] = 0;
    testing[1] = 1;
    testing[2] = 1;
    testing[3] = 1;
    testing[4] = 1;
    testing[5] = 0;
    testing[6] = 1;
    testing[7] = 0;
    testing[8] = 1;
    testing[9] = 1;
    testing[10] = 0;
    testing[11] = 1;
    testing[12] = 1;
    testing[13] = 0;
    testing[14] = 1;
    testing[15] = 0;

    int nodeNum = 4;

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


    // convert testcase's adjacency matrix representation
    // into normal adjacency matrix
    int graphArrayIndex = 0;
    for(size_t i = 0; i < input.size(); i++) {
        for(size_t j = 0; j < input.size(); j++) {
            input[i][j] = testing[graphArrayIndex];
            graphArrayIndex += 1;
        }
    }
    findMatrixCubed(input, result);

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
