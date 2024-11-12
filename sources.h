#include<vector>
#ifndef RESEARCHPROJECT_SOURCES_H
#define RESEARCHPROJECT_SOURCES_H
extern int all_distances[100][100];
extern int knapsack_array[1000][2];
extern int partition_array[8000];
class functions {
public:
    std::vector<int> generate_TS(std::vector<std::vector<int>> all_distances);
    std::vector<int> generate_KP(std::vector<std::vector<int>> objects, int max_size);
};



#endif //RESEARCHPROJECT_SOURCES_H
