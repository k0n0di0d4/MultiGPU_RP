#include <iostream>
#include <random>
#include <stdio.h>
using namespace std;

// weight from 1 to 50
// value from 1 to 100
#define the_size 1000

std::random_device device;
std::mt19937 generator(device());

int main()
{
    std::uniform_int_distribution<int> weight_distribution(1, 50);
    std::uniform_int_distribution<int> value_distribution(1, 100);
    
    int knapsack_array[the_size][2];
    std::cout << "{";
    for(int i = 0; i < the_size; i++) {
        knapsack_array[i][0] = weight_distribution(generator);
        knapsack_array[i][1] = value_distribution(generator);
        if(i == the_size - 1){
            std::cout << "{" << knapsack_array[i][0] << ", " << knapsack_array[i][1] << "}";
            break;
        }
        std::cout << "{" << knapsack_array[i][0] << ", " << knapsack_array[i][1] << "}, ";
    }
    std::cout << "}";
    
    std::cout << sizeof(knapsack_array) / sizeof(int);
    return 0;
}

