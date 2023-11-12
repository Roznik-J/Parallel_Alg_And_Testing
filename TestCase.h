#ifndef TESTCASE_H
#define TESTCASE_H

#include <vector>
#include <string>

using namespace std;

class TestCase {
    public:
        vector<vector<int>> adjList;
        vector<vector<int>> adjMatrix;
        TestCase(string fileName);
        void printList();
       	int getNodeSize();
    private:
        string fileName;
        int numNodes;
};

#endif