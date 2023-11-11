#ifndef TESTCASEV2_H
#define TESTCASEV2_H

#include "Edge.h"

#include <vector>
#include <string>

using namespace std;

class TestCaseV2 {
    public:
        vector<vector<Edge>> adjList;
        TestCase(string fileName);
        void printList();
        int printSum();

        // We can remove this later
        void RunSquareTest(void);
    private:
        string fileName;
        bool isUndirected;
        int edgeSum;
};

#endif