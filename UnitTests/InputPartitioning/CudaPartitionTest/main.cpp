#include "gtest/gtest.h"
#include "../../../CudaAlgorithm/TestCase.hpp"
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

TEST(InputPartitionCuda, ConstructerDoesNotThrow) 
{
	EXPECT_EQ(gcCalibratedGraphs.size(), 3);
	std::string lcFileName1("../testcases/NoTriangles.txt");
	std::string lcFileName2("../testcases/OneTriangle.txt");
	std::string lcFileName3("../testcases/MaxTriangles.txt");
    EXPECT_NO_THROW(TestCase lcTestCase(lcFileName1));
    EXPECT_NO_THROW(TestCase lcTestCase(lcFileName2));
    EXPECT_NO_THROW(TestCase lcTestCase(lcFileName3));
}

TEST(InputPartitionCuda, NoTriangles)
{
    std::string lcFileName("../testcases/NoTriangles.txt");

    TestCase lcTestCase(lcFileName);

    auto lpcEntry = gcCalibratedGraphs.find(lcFileName);

    if(lpcEntry != gcCalibratedGraphs.end())
    {
        auto lpcData = lpcEntry->second;
        EXPECT_EQ(lpcData.mnNumNodes, lcTestCase.GetNodeSize());
        EXPECT_EQ(lpcData.mnNumEdges, lcTestCase.GetNumEdges());
        EXPECT_EQ(lpcData.mnNumActualTriangles, lcTestCase.GetNumTriangles());
    }
    else
    {
        // Fail Test
        EXPECT_EQ(2,0);
    }
}

TEST(InputPartitionCuda, OneTriangle)
{
    std::string lcFileName("../testcases/OneTriangle.txt");

    TestCase lcTestCase(lcFileName);

    auto lpcEntry = gcCalibratedGraphs.find(lcFileName);

    if(lpcEntry != gcCalibratedGraphs.end())
    {
        auto lpcData = lpcEntry->second;
        EXPECT_EQ(lpcData.mnNumNodes, lcTestCase.GetNodeSize());
        EXPECT_EQ(lpcData.mnNumEdges, lcTestCase.GetNumEdges());
        EXPECT_EQ(lpcData.mnNumActualTriangles, lcTestCase.GetNumTriangles());
    }
    else
    {
        // Fail Test
        EXPECT_EQ(2,0);
    }
}

TEST(InputPartitionCuda, MaxTriangles)
{
    std::string lcFileName("../testcases/MaxTriangles.txt");

    TestCase lcTestCase(lcFileName);

    auto lpcEntry = gcCalibratedGraphs.find(lcFileName);

    if(lpcEntry != gcCalibratedGraphs.end())
    {
        auto lpcData = lpcEntry->second;
        EXPECT_EQ(lpcData.mnNumNodes, lcTestCase.GetNodeSize());
        EXPECT_EQ(lpcData.mnNumEdges, lcTestCase.GetNumEdges());
        EXPECT_EQ(lpcData.mnNumActualTriangles, lcTestCase.GetNumTriangles());
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