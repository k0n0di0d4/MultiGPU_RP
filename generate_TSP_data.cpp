#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

std::vector<std::vector<int>> generateTSPData(int num_cities, int min_distance, int max_distance) {
    std::vector<std::vector<int>> data(num_cities, std::vector<int>(num_cities, 0));
    srand(time(NULL)); // Seed the random number generator

    for (int i = 0; i < num_cities; ++i) {
        for (int j = i + 1; j < num_cities; ++j) {
            int distance = rand() % (max_distance - min_distance + 1) + min_distance;
            data[i][j] = distance;
            data[j][i] = distance; // Set the symmetric distance
        }
    }

    return data;
}

int main() {
    int num_cities = 100;
    int min_distance = 5;
    int max_distance = 300;

    std::vector<std::vector<int>> tsp_data = generateTSPData(num_cities, min_distance, max_distance);

    std::cout << "{";
    // Print the generated data
    for (int i = 0; i < num_cities; ++i) {
        std::cout << "{";
        for (int j = 0; j < num_cities; ++j) {
            if(j != num_cities - 1){
                std::cout << tsp_data[i][j] << ", ";
            } else {
                std::cout << tsp_data[i][j];
            }
            
        }
        if(i != num_cities - 1) {
            std::cout << "}, " << std::endl;
        } else {
            std::cout << "}";
        }
        
    }
    std::cout << "}";

    return 0;
}
