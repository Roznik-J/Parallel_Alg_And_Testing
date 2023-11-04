// graph generator

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>

struct Info {
	int _dest;
	int _weights;
};

void RandomGraph(int vertices, int weight, bool sparse, bool direction)
{
	std::vector<std::vector<Info>> List;
	List.resize(vertices);
	srand(time(NULL));
	int Edges = 0;

	for (int i = 0; i < vertices; i++)
	{
		int numedges = 0;
		int tempEdge = 0;
		if (sparse)
		{
			numedges = (std::rand() % 2);
		}
		else if (!sparse)
		{
			numedges = (std::rand() % vertices) - 1;
		}
		numedges++;
		std::vector<int> v;
		while (tempEdge < numedges)
		{
			int edge = (std::rand() % vertices);
			//std::cout << edge << std::endl;
			std::vector<int>::iterator it = std::find(v.begin(), v.end(), edge);
			int _weights = (std::rand()%19) + weight; //random number from -1 to 9
			if (edge != i &&  it == v.end())
			{//if it is not equal to self and not in vector
				v.push_back(edge);
				List[i].push_back({edge, _weights});
				tempEdge++;
				Edges++;
			}
		}
	}

	std::string temp;

	int padding = 0;
	if (vertices < 10) {
		padding = 1;
	}
	if (weight < 0 && sparse && direction)
	{
		temp = "nSparseFalse/" + std::string(padding, '0').append(std::to_string(vertices)) + ".txt";
	}
	else if (weight < 0 && !sparse && direction)
	{
		temp = "nNotSparseFalse/" + std::string(padding, '0').append(std::to_string(vertices)) + ".txt";
	}
	else if (weight < 0 && sparse && !direction)
	{
		temp = "nSparseTrue/" + std::string(padding, '0').append(std::to_string(vertices)) + ".txt";
	}
	else if (weight < 0 && !sparse && !direction)
	{
		temp = "nNotSparseTrue/" + std::string(padding, '0').append(std::to_string(vertices)) + ".txt";
	}
	else if (weight == 0 && sparse && direction)
	{
		temp = "pSparseFalse/" + std::string(padding, '0').append(std::to_string(vertices)) + ".txt";
	}
	else if (weight == 0 && !sparse && direction)
	{
		temp = "pNotSparseFalse/" + std::string(padding, '0').append(std::to_string(vertices)) + ".txt";
	}
	else if (weight == 0 && sparse && !direction)
	{
		temp = "pSparseTrue/" + std::string(padding, '0').append(std::to_string(vertices)) + ".txt";
	}
	else if (weight == 0 && !sparse && !direction)
	{
		temp = "pNotSparseTrue/" + std::string(padding, '0').append(std::to_string(vertices)) + ".txt";
	}

	// for (int i = 0; i < List.size(); i++)
	// {
	// 	for (size_t j = 0; j < List.at(i).size(); j++)
	// 	{
	// 		if (List.at(i).at(j)._dest == List.at(j).at(i)._dest)
	// 		{
	// 			List[i].erase(j);
	// 		}
	// 	}
	// }

	std::vector<std::vector<Info>> ListDirected;
	ListDirected.resize(vertices);
	if (direction == false)
	{
		int tempList[vertices][vertices][2] = {0};
		for (int i = 0; i < List.size(); i++)
		{
			for (int j = 0; j < List.at(i).size(); j++)
			{
				tempList[i][j][0] = List.at(i).at(j)._dest;
				tempList[i][j][1] = List.at(i).at(j)._weights;
			}
		}

		for (int i = 0; i < vertices; i++)
		{
			for (int j = 0; j < vertices; j++)
			{
				if (tempList[i][j][0] == tempList[j][i][0])
				{break;}
				else if (tempList[i][j][0] == 0 && tempList[i][j][1] == 0)
				{break;}
				else if (tempList[i][j][0] >= vertices && tempList[i][j][1] >= vertices)
				{break;}
				else if (tempList[i][j][0] < 0)
				{break;}
				else 
				{	
					ListDirected[i].push_back({tempList[i][j][0], tempList[i][j][1]});				
					//std::cout<<i<<" "<<tempList[i][j][0]<<" "<<tempList[i][j][1]<<std::endl;
				}
			}
		}
	}
	std::ofstream file(temp);
	file << vertices << std::endl;
	file << Edges << std::endl;
	
	if (direction)
	{
		file << "true" <<std::endl;
	}
	else 
	{
		file << "false" << std::endl;
	}

	for (int i = 0; i < List.size(); i++)
	{
		for (int j = 0; j < List.at(i).size(); j++)
		{
			file << i << " " << List.at(i).at(j)._dest << " " << List.at(i).at(j)._weights << std::endl;
		}
	}
	file.flush();
	file.close();
}

int main()
{

	for (int i = 5; i < 100; i++)
	{
		RandomGraph(i, 0, true, true);
		RandomGraph(i, 0, true, false);
		RandomGraph(i, 0, false, true);
		RandomGraph(i, 0, false, false);
		RandomGraph(i, -9, true, true);
		RandomGraph(i, -9, true, false);
		RandomGraph(i, -9, false, true);
		RandomGraph(i, -9, false, false);
		std::cout<<"Done with: "<<i<<std::endl;
	}
	std::cout<<"Done!"<<std::endl;
	return 0;
}