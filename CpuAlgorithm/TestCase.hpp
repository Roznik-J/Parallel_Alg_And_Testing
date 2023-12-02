#ifndef TESTCASE_HPP
#define TESTCASE_HPP

#include <vector>
#include <string>

using namespace std;

class TestCase {
    public:
        vector<vector<int>> adjList;
        vector<vector<int>> adjMatrix;
        TestCase(const string& fileName);
        void printList();
       	int getNodeSize();
    private:
        string fileName;
        int numNodes;
};

#endif