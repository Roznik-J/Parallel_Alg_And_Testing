#include "gtest/gtest.h"
#include "../../../CpuAlgorithm/TestCase.hpp"
#include "../../../CpuAlgorithm/MatrixMultiplication.hpp"
#include <vector>
#include <iostream>
#include <fstream>
#include <map>

struct tsCalTriangleGraph
{
    int mnNumNodes{0};
    int mnNumEdges{0};
    int mnNumActualTriangles{0};
};

std::vector<std::string> gcFileNames;
std::map<std::string, tsCalTriangleGraph> gcCalibratedGraphs;

/**
* Debug purposes only
*/
void printFileNames(void)
{
    std::cout << "vector size " << gcFileNames.size() << std::endl;
    for(uint lnIdx = 0; lnIdx < gcFileNames.size(); lnIdx++)
    {
        std::cout << gcFileNames.at(lnIdx) << std::endl;
    }
}

void GetCalibratedInformation(void)
{
    for(std::string lcFileName : gcFileNames)
    {
        std::ifstream fileIn(lcFileName);
        if(!fileIn)
        {
            throw std::runtime_error("Error Reading File");
        }

        tsCalTriangleGraph lsGraphInfo;

        fileIn >> lsGraphInfo.mnNumNodes;
        fileIn >> lsGraphInfo.mnNumEdges;
        fileIn >> lsGraphInfo.mnNumActualTriangles;

        gcCalibratedGraphs.insert({lcFileName, lsGraphInfo});

        fileIn.close();
    }
}

TEST(InputPartitionCPU, ConstructerDoesNotThrow) 
{
	EXPECT_EQ(gcCalibratedGraphs.size(), 3);
    EXPECT_NO_THROW(TestCase lcTestCase("../testcases/NoTriangles.txt"));
    EXPECT_NO_THROW(TestCase lcTestCase("../testcases/OneTriangle.txt"));
    EXPECT_NO_THROW(TestCase lcTestCase("../testcases/MaxTriangles.txt"));
}

TEST(InputPartitionCPU, NoTriangles)
{
	std::string lcFileName("../testcases/NoTriangles.txt");
    
    TestCase lcTestCase(lcFileName);
    int lnNumTriangles = runNonGpuMatrxMulti(lcTestCase);
    auto lpcEntry = gcCalibratedGraphs.find(lcFileName);

    if(lpcEntry != gcCalibratedGraphs.end())
    {
        auto lpcData = lpcEntry->second;
        EXPECT_EQ(lpcData.mnNumNodes, lcTestCase.getNodeSize());
        EXPECT_EQ(lpcData.mnNumActualTriangles, lnNumTriangles);
    }
    else
    {
        // Fail Test
        EXPECT_EQ(2,0);
    }
}

TEST(InputPartitionCPU, OneTriangle)
{
	std::string lcFileName("../testcases/OneTriangle.txt");
    
    TestCase lcTestCase(lcFileName);
    int lnNumTriangles = runNonGpuMatrxMulti(lcTestCase);
    auto lpcEntry = gcCalibratedGraphs.find(lcFileName);

    if(lpcEntry != gcCalibratedGraphs.end())
    {
        auto lpcData = lpcEntry->second;
        EXPECT_EQ(lpcData.mnNumNodes, lcTestCase.getNodeSize());
        EXPECT_EQ(lpcData.mnNumActualTriangles, lnNumTriangles);
    }
    else
    {
        // Fail Test
        EXPECT_EQ(2,0);
    }
}


TEST(InputPartitionCPU, MaxTriangles)
{
	std::string lcFileName("../testcases/MaxTriangles.txt");
    
    TestCase lcTestCase(lcFileName);
    int lnNumTriangles = runNonGpuMatrxMulti(lcTestCase);
    auto lpcEntry = gcCalibratedGraphs.find(lcFileName);

    if(lpcEntry != gcCalibratedGraphs.end())
    {
        auto lpcData = lpcEntry->second;
        EXPECT_EQ(lpcData.mnNumNodes, lcTestCase.getNodeSize());
        EXPECT_EQ(lpcData.mnNumActualTriangles, lnNumTriangles);
    }
    else
    {
        // Fail Test
        EXPECT_EQ(2,0);
    }
}

int main(int argc, char *argv[]) 
{
    ::testing::InitGoogleTest(&argc, argv);

    if(argc == 1)
    {
        std::cout << "No Files Found: Maybe Rebuild Graphs? " << std::endl;
        return 1;
    }
    gcFileNames.assign(argv + 1, argv + argc);

    GetCalibratedInformation();


    return RUN_ALL_TESTS();
}
