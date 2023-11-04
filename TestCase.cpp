#include "TestCase.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

TestCase::TestCase(string fileName) : fileName(fileName) {
    ifstream fileIn(fileName);

    // first line is the number of verticies in the graph
    int numVerticies;
    fileIn >> numVerticies;
    fileIn >> edgeSum;

    // resize the vector to hold `N` elements
    adjList.resize(numVerticies);

    // second line is whether or not it is an undirected graph (directed: false, undirected: true)
    bool isUndirected;
    fileIn >> std::boolalpha >> isUndirected;
    this->isUndirected = isUndirected;

    // read rest of lines and add edges to the graph
    int src, dest, weight;
    while (fileIn >> src >> dest >> weight) {
        adjList[src].push_back({dest, weight});
        if (isUndirected) {
            adjList[dest].push_back({src, weight});
        }
    }
}

int TestCase::printSum()
{
    return edgeSum;
}

void TestCase::printList() {
    if (isUndirected) {
        cout << endl << "------ Undirected Graph ";
    } else {
        cout << endl << "------ Directed Graph ";
    }
    cout << "from file: " << fileName << " ------" << endl;
    for (size_t i = 0; i < this->adjList.size(); i++) {
        cout << i << ": ";
        for (size_t j = 0; j < this->adjList.at(i).size(); j++) {
            Edge edge = this->adjList.at(i).at(j);
            if (j == 0) {
                cout << "(" << edge.dest << ", " << edge.weight << ")";
            } else {
                cout << " -> (" << edge.dest << ", " << edge.weight << ")";
            }
        }
        cout << endl;
    }
}