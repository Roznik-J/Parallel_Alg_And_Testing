//*****************************************************
// Developed by J. Roznik
// 2023-11-11
// Creates Graphs
//*****************************************************

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>
#include <cstdlib>
#include <ctime>
#include <set>

std::pair<int,int> GetValidEdge(int numNodes)
{
    int lnSource = std::rand() % (numNodes - 1);
    int lnDestRange = numNodes - lnSource;
    int lnDest = -1;
    while(lnSource >= lnDest)
    {
        lnDest = std::rand() % lnDestRange + lnSource;
    }
    std::pair<int,int> val = {lnSource,lnDest};
    return(val);
}

void RandomGraphV2(int anNumNodes, float anSparsity)
{
    // Set seed based on current time
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    // Max possible edges given the number of nodes
    int lnMaxNumEdges = static_cast<int>((static_cast<float>(anNumNodes * (anNumNodes - 1))/2));

    int lnMinNumEdges = anNumNodes - 1;

    int lnRange = lnMaxNumEdges - lnMinNumEdges;

    // Number of edges is decided using a random number, and then by the sparsity as a scalar
    int lnNumCandidateEdges = (std::rand() % lnRange)*anSparsity + lnMinNumEdges;

    std::set<std::pair<int,int>> edges;

    for(int lnIdx = 0; lnIdx < lnNumCandidateEdges; lnIdx++)
    {
        std::pair<int,int> lcEdge = GetValidEdge(anNumNodes);
        int prevsize = edges.size();
        edges.insert(lcEdge);
        int currsize = edges.size();
        while(currsize == prevsize)
        {
            lcEdge = GetValidEdge(anNumNodes);
            edges.insert(lcEdge);
            currsize = edges.size();
        }
    }

    // Finished Creating Edges in Graph, now Write to File

    std::string lcFileName;

    if(anSparsity <= 0.5)
    {
        lcFileName = "v2GraphsSparse/" + std::string(std::to_string(anNumNodes)) + ".txt";
    }
    else
    {
        lcFileName = "v2GraphsDense/" + std::string(std::to_string(anNumNodes)) + ".txt";
    }

    std::ofstream file(lcFileName);
	file << anNumNodes << std::endl;
	file << edges.size() << std::endl;
    file << anSparsity << std::endl;

    for(std::pair<int, int> lcObject : edges)
    {
        file << lcObject.first << " " << lcObject.second << std::endl;
    }
    file.flush();
	file.close();

}

// Generates graphs with a set number of triangles
void TriangleGraph(int anNumNodes, int anNumTriangles)
{
    std::set<std::pair<int,int>> edges;
    // Generate a series of edges such that 0->1->2->...->n
    for(int lnIdx = 0; lnIdx < (anNumNodes - 1); lnIdx++)
    {
        std::pair<int, int> lcEdge = {lnIdx, lnIdx+1};
        edges.insert(lcEdge);
    }
    // Add a series of edges which form triangles
    for(int lnIdx = 0; lnIdx < anNumTriangles; lnIdx++)
    {
        std::pair<int, int> lcEdge = {lnIdx, lnIdx+2};
        edges.insert(lcEdge);
    }
    std::string lcFileName;
    lcFileName = "v2GraphsSetTriangles/" + std::string(std::to_string(anNumTriangles)) + ".txt";
    std::ofstream file(lcFileName);
    file << anNumNodes << std::endl;
	file << edges.size() << std::endl;
    file << anNumTriangles << std::endl;
    for(std::pair<int, int> lcObject : edges)
    {
        file << lcObject.first << " " << lcObject.second << std::endl;
    }
    file.flush();
	file.close();
}

int main()
{

    //for(int i = 5; i < 200; i++)
    //{
    //    RandomGraphV2(i,0.5);
    //    RandomGraphV2(i,1);
    //}
    //for(int i = 0; i < 200; i++)
    //{
    //    TriangleGraph(300, i);
    //}
    TriangleGraph(10, 0);
    //for(int i = 100; i <= 10000; i+=100)
    for(int i = 100; i <= 10000; i+=100)
    {
        TriangleGraph(i, i-2);
    }
	return 0;
}