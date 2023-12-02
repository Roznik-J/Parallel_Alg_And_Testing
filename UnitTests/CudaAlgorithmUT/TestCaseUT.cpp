//*****************************************************
// Developed by J. Roznik
// 2023-11-25
// UT for CudaAlgorithm/TestCase
//*****************************************************

#include "gtest/gtest.h"
#include "../../CudaAlgorithm/TestCase.hpp"
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
int gnNumTestCases;

/**
* Debug purposes only
*/
void printFileNames(void)
{
    std::cout << "vector size " << gcFileNames.size() << std::endl;
    for(int lnIdx = 0; lnIdx < gcFileNames.size(); lnIdx++)
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

TEST(TestCaseUT, ConstructerDoesNotThrow) 
{ 
    std::string lcFileName = gcFileNames.at(0);
    EXPECT_NO_THROW(TestCase lcTestCase(lcFileName));
}

TEST(TestCaseUT, AllCalibratedTrianglesWork)
{
    for(int lnIdx = 0; lnIdx < gnNumTestCases; lnIdx++)
    {
        std::string lcFileName = gcFileNames.at(lnIdx);

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

    gnNumTestCases = gcFileNames.size();

    return RUN_ALL_TESTS();
}