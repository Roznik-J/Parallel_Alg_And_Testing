#ifndef TESTCASE_H
#define TESTCASE_H

#include "Edge.h"

#include <vector>
#include <string>

using namespace std;

class TestCase {
    public:
        vector<vector<Edge>> adjList;
        TestCase(string fileName);
        void printList();
        int printSum();
    private:
        string fileName;
        bool isUndirected;
        int edgeSum;
};

#endif