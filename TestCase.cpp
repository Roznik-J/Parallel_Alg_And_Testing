#include "TestCase.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

TestCase::TestCase(string fileName) : fileName(fileName) {
    ifstream fileIn(fileName);
    int numEdges;
    float sparsity;

    fileIn >> numNodes;
    fileIn >> numEdges; // currently not useful to me
    fileIn >> sparsity; // currently not useful to me

    adjList.resize(numNodes);
    adjMatrix.resize(numNodes);
    for(int i = 0; i < numNodes; i++) {
    	adjMatrix[i].resize(numNodes, 0);
    }

    int src;
    int dest;
    for(int i = 0; i < numEdges; i++){
        fileIn >> src;
        fileIn >> dest;

        adjList[src].push_back(dest);
        adjList[dest].push_back(src);
        adjMatrix[src][dest] = 1;
        adjMatrix[dest][src] = 1;
    }

    fileIn.close();
}

void TestCase::printList() {
    cout << "From file: " << fileName << " ------" << endl;
    for (size_t i = 0; i < this->adjList.size(); i++) {
        cout << i << ": ";
        for (size_t j = 0; j < this->adjList.at(i).size(); j++) {
            int edge = this->adjList.at(i).at(j);
            if (j == 0) {
                cout << "(" << to_string(edge) << ")";
            }
            else {
                cout << " -> (" << to_string(edge) << ")";
            }
        }
        cout << endl;
    }
}

int TestCase::getNodeSize() {
	return numNodes;
}
