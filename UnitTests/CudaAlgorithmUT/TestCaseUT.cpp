//*****************************************************
// Developed by J. Roznik
// 2023-11-25
// UT for CudaAlgorithm/TestCase
//*****************************************************

#include "gtest/gtest.h"
#include "../../CudaAlgorithm/TestCase.hpp"

TEST (TestCaseUT, ConstructerDoesNotThrow) 
{ 
    std::string lcFileName = "../../GraphsSetTriangles/8.txt";
    EXPECT_NO_THROW(TestCase lcTestCase(lcFileName));
}


int main(int argc, char *argv[]) 
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}