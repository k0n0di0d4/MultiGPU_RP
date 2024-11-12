#include <iostream>
#include <random>
#include <stdio.h>
#include <string.h>
std::random_device device;
std::mt19937 generator(device());
using namespace std;
#define size 8000
int main()
{
    std::uniform_int_distribution<int> number_distribution(1, 300);
    int array[size];
    memset(array, 0, sizeof(array));
    printf("{");
    for(int i = 0; i < size; i++) {
        array[i] = number_distribution(generator);
        if(i != size - 1){
            printf("%d, ", array[i]);
        }
        else {
            printf("%d", array[i]);
        }
        if(i % 44 == 0 && i != 0) printf("\n");
    }
    printf("};\n");
    
    return 0;
}