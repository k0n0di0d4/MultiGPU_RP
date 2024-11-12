#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "sources.h"
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <ctime>
#include <omp.h>
#include <chrono>
#include <string>
#include <fstream>
#include <iostream>

// change these for tests B)
//#define pop_size 32768
#define pop_size 65536

#define generations 500
#define tour_size 5
#define citynum 100
#define max_weight 500
#define KP_size 1000
#define PP_size 8000
#define partitions 3
#define p2p_send_intervals 5
#define mutation_prob 0.01


functions func;

void genetic_algorithm(int whichAlgo, int nDevices, int powerCap, int all_distances[][citynum], int knapsack_array[][2], int partition_array[PP_size]);

#define gpuCheckErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


// kernel for setting up curand
__global__ void setup_kernel(curandState *state, unsigned long long *d_seed) {
	unsigned int id = threadIdx.x + (blockIdx.x * blockDim.x);
	curand_init(*d_seed, id, 0, &state[id]);
}


// kernel for doing the GA for traveling salesman problem
__global__ void TS_doGAKernel(curandState * d_states, int *d_TS_Members_DNA, int *d_TS_Members_fitness, int *d_all_distances)
{
	unsigned int id = threadIdx.x + (blockIdx.x * blockDim.x);
	int currentMember = id * citynum;
	// index from 0 to the number of cities
	int citynumbers[citynum];
	for (int i = 0; i < citynum; i++) {
		citynumbers[i] = i;
	}
	int myrand = 0;

	// copy state to local memory for efficiency
	curandState localState = d_states[id];
	
	// shuffle the cities using Fisher-Yates shuffle
	for(int i = citynum - 1; i > 0; i--) {
		
		float myrandf = curand_uniform(&localState);
		
		// max value - min value
		myrandf *= i - 1 + 0 + 0.999999;
		// add min value
		myrandf += 0;

		/*if (id == 22) {
			printf("myrandf is equal to %f\n", myrandf);
		}*/

		// convert the float into int
		myrand = (int)truncf(myrandf);

		int temp = citynumbers[i];

		citynumbers[i] = citynumbers[myrand];
		citynumbers[myrand] = temp;
	}


	// copy state back to global memory for efficiency
	d_states[id] = localState;

	// calculate fitness and initialize the member from the previously randomized array
	int counter = 0;
	for (int i = 0; i < citynum; i++) {
		if (i > 0) {
			counter += d_all_distances[citynumbers[i - 1] * citynum + citynumbers[i]];
		}
		d_TS_Members_DNA[currentMember + i] = citynumbers[i];
	}

	d_TS_Members_fitness[id] = counter;
}


__global__ void KP_doGAKernel(curandState* d_states, int *d_KP_Members_DNA, int *d_KP_Members_fitness, int *d_knapsack_array)
{
	unsigned int id = threadIdx.x + (blockIdx.x * blockDim.x);

	int overall_weight = 0;
	int overall_fitness = 0;
	//int localMember[KP_size];
	//memset(localMember, 0, sizeof(localMember));

	// copy state to local memory for efficiency
	curandState localState = d_states[id];

	while (overall_weight < max_weight) {
		int myrand = 0;
		float myrandf = curand_uniform(&localState);
		// max value - min value
		myrandf *= KP_size - 1 + 0 + 0.999999;
		// add min value
		myrandf += 0;
		myrand = (int)truncf(myrandf);
		if (d_KP_Members_DNA[id * KP_size + myrand] != 1) {
			d_KP_Members_DNA[id * KP_size + myrand] = 1;
			overall_weight += d_knapsack_array[myrand * 2 + 0];
			overall_fitness += d_knapsack_array[myrand * 2 + 1];
		}
	}
	if (overall_weight > max_weight) {
		overall_fitness = 0;
	}
	d_states[id] = localState;
	d_KP_Members_fitness[id] = overall_fitness;
}


__global__ void PP_doGAKernel(curandState* d_states, int *d_PP_Members_DNA, int *d_PP_Members_fitness, int *d_partition_array, int sizePerDevice) {
	unsigned int id = threadIdx.x + (blockIdx.x * blockDim.x);

	curandState localState = d_states[id];
	int myrand = 0;
	int total_per_part[partitions];
	memset(total_per_part, 0, sizeof(total_per_part));

	for (int i = 0; i < PP_size; i++) {
		float myrandf = curand_uniform(&localState);
		// max value - min value
		myrandf *= (partitions - 1) - 0 + 0 + 0.999999;
		// add min value
		myrandf += 0;

		myrand = (int)truncf(myrandf);
		d_PP_Members_DNA[id * PP_size + i] = myrand;
		total_per_part[myrand] += d_partition_array[i];
	}
	d_states[id] = localState;
	int max_value = INT_MIN;
	int min_value = INT_MAX;
	for (int i = 0; i < partitions; i++) {
		if (total_per_part[i] > max_value) {
			max_value = total_per_part[i];
		}
		if (total_per_part[i] < min_value) {
			min_value = total_per_part[i];
		}
	}
	d_PP_Members_fitness[id] = max_value - min_value;
	//printf("%d\n", d_PP_Members_fitness[id]);
}


__global__ void TS_TournamentSelection_Kernel(curandState* d_states, int* d_TS_Members_fitness_Copy, int* chosen_Members) {
	unsigned int id = threadIdx.x + (blockIdx.x * blockDim.x); // 0 - 199 (for population of 1600 and 8 gpus)
	//printf("My id is %d\n", id);
	int myrand1 = 0;
	int myrand2 = 0;

	// copy state to local memory for efficiency
	curandState localState = d_states[id];
	float myrandf1 = curand_uniform(&localState);
	float myrandf2 = curand_uniform(&localState);
	// max value - min value
	myrandf1 *= pop_size - 1 + 0 + 0 + 0.999999;
	myrandf2 *= pop_size - 1 + 0 + 0 + 0.999999;
	// add min value
	myrandf1 += 0;
	myrandf2 += 0;
	// convert the float into int
	myrand1 = (int)truncf(myrandf1);
	myrand2 = (int)truncf(myrandf2);

	d_states[id] = localState;

	// TODO: choose smaller fitness (smaller distance)
	if (d_TS_Members_fitness_Copy[myrand1] < d_TS_Members_fitness_Copy[myrand2]) {
		chosen_Members[id] = myrand1;
	}
	else {
		chosen_Members[id] = myrand2;
	}
}


__global__ void KP_TournamentSelection_Kernel(curandState* d_states, int *d_KP_chosen_Members, int *d_KP_Members_fitness) {
	unsigned int id = threadIdx.x + (blockIdx.x * blockDim.x); // 0 - 1023 (for population of 8192 and 8 gpus)

	int myrand0 = 0;
	int myrand1 = 0;

	// copy state to local memory for efficiency
	curandState localState = d_states[id];
	float myrandf0 = curand_uniform(&localState);
	float myrandf1 = curand_uniform(&localState);
	// max value - min value
	myrandf0 *= pop_size - 2 + 0 + 0.999999;
	myrandf1 *= pop_size - 2 + 0 + 0.999999;
	// add min value
	myrandf0 += 0;
	myrandf1 += 0;
	// convert the float into int
	myrand0 = (int)truncf(myrandf0);
	myrand1 = (int)truncf(myrandf1);

	d_states[id] = localState;

	// TODO: choose bigger now
	if (d_KP_Members_fitness[myrand1] > d_KP_Members_fitness[myrand0]) {
		d_KP_chosen_Members[id] = myrand1;
	}
	else {
		d_KP_chosen_Members[id] = myrand0;
	}
}


__global__ void PP_TournamentSelection_Kernel(curandState* d_states, int *d_PP_chosen_Members, int *d_PP_Members_fitness, int sizePerDevice) {
	
	unsigned int id = threadIdx.x + (blockIdx.x * blockDim.x);

	
		float myrandf[tour_size];
		int myrand[tour_size];

		// copy state to local memory for efficiency
		curandState localState = d_states[id];

		// choose random members for selections and save their id
		for (int i = 0; i < tour_size; i++) {
			myrandf[i] = curand_uniform(&localState);
			myrandf[i] *= pop_size - 2 + 0 + 0.999999;
			myrandf[i] += 0;
			myrand[i] = (int)truncf(myrandf[i]);
		}

		d_states[id] = localState;
		int best_fitness = INT_MAX;
		int best_member = 0;
		for (int i = 0; i < tour_size; i++) {
			if (d_PP_Members_fitness[myrand[i]] < best_fitness) {
				best_fitness = d_PP_Members_fitness[myrand[i]];
				best_member = myrand[i];
			}
		}
		d_PP_chosen_Members[id] = best_member;
	
}


// pretty easy to finish but no time right now
__global__ void rouletteSelectionKernel(curandState* d_states, int d_TS_Members_fitness_Copy[], int chosen_Members[]) {
	//unsigned int id = threadIdx.x + (blockIdx.x * blockDim.x); // 0 - 199 (for population of 1600 and 8 gpus)
}


// d_TS_Members_DNA has only part of dna, handled by the gpu 
// d_TS_Members_DNA_Copy has whole DNA of all members (it's updated)
__global__ void TS_Crossover_Kernel(curandState* d_states, int* d_TS_Members_DNA, int* d_TS_Members_fitness, int* d_all_distances, int* d_TS_Members_DNA_Copy, int* d_TS_chosenMembers) {
	unsigned int id = threadIdx.x + (blockIdx.x * blockDim.x);// = 0-99, for a population of 1600 on 8 GPUs 
	unsigned int second_id = id + (blockDim.x * gridDim.x); // = id + 100, for a population of 1600 on 8 GPUs 
	
	// if the value is already in the original array, mark the index as -1
	// if it's not there mark is as 1
	int missingParts0[citynum];
	int missingParts1[citynum];
	//int size0 = 0, size1 = 0;

	// choose a point of crossover
	int myrand = 0;
	curandState localState = d_states[id];
	float myrandf = curand_uniform(&localState);
	myrandf *= citynum - 0 + 0 + 0.999999;
	myrandf += 0;
	myrand = (int)truncf(myrandf);

	for (int i = 0; i < citynum; i++) {
		missingParts0[i] = 0;
		missingParts1[i] = 0;
	}

	// go until the point of crossover
	for (int i = 0; i < myrand; i++) {
		// copy from the original 
		d_TS_Members_DNA[id * citynum + i] = d_TS_Members_DNA_Copy[d_TS_chosenMembers[id] * citynum + i];
		d_TS_Members_DNA[second_id * citynum + i] = d_TS_Members_DNA_Copy[d_TS_chosenMembers[second_id] * citynum + i];

		// mark missing parts as done
		missingParts0[d_TS_Members_DNA[id * citynum + i]] = -1;
		missingParts1[d_TS_Members_DNA[second_id * citynum + i]] = -1;
	}


	// do the crossover
	// start from the point of break
	for (int i = myrand; i < citynum; i++) {
		for (int j = 0; j < myrand; j++) {
			// check if both werent marked as having duplicates
			if (d_TS_Members_DNA[id * citynum + i] != -1 && d_TS_Members_DNA[second_id * citynum + i] != -1) {
				// if found a duplicate that was here before, give it -1
				if (d_TS_Members_DNA[id * citynum + j] == d_TS_Members_DNA_Copy[d_TS_chosenMembers[second_id] * citynum + i] ) {
					//missingParts0[d_TS_Members_DNA_Copy[d_TS_chosenMembers[second_id]][i]] = 1;
					d_TS_Members_DNA[id * citynum + i] = -1;
				}
				// if found a duplicate that was here before, give it -1
				if (d_TS_Members_DNA[second_id * citynum + j] == d_TS_Members_DNA_Copy[d_TS_chosenMembers[id] * citynum + i]) {
					//missingParts1[d_TS_Members_DNA_Copy[d_TS_chosenMembers[id]][i]] = 1;
					d_TS_Members_DNA[second_id * citynum + i] = -1;
				}
			}
			// if both had duplicates, stop searching
			else {
				break;
			}
			
		}
		// if didn't find a duplicate swap the values between the 1st member and 2nd member
		if (d_TS_Members_DNA[id * citynum + i] != -1) {
			d_TS_Members_DNA[id * citynum + i] = d_TS_Members_DNA_Copy[d_TS_chosenMembers[second_id] * citynum + i];
			missingParts0[d_TS_Members_DNA[id * citynum + i]] = -1;
		}
		if (d_TS_Members_DNA[second_id * citynum + i] != -1) {
			d_TS_Members_DNA[second_id * citynum + i] = d_TS_Members_DNA_Copy[d_TS_chosenMembers[id] * citynum + i];
			missingParts1[d_TS_Members_DNA[second_id * citynum + i]] = -1;
		}
	}

	// go through whole missingParts0 and 1
	// if any was found as missing, go through the normal array 
	// from the point of crossover and put a missing part there
	for (int i = 0; i < citynum; i++) {
		if (missingParts0[i] == 0) {
			for (int j = myrand; j < citynum; j++) {
				if (d_TS_Members_DNA[id * citynum + j] == -1) {
					d_TS_Members_DNA[id * citynum + j] = i;
					break;
				}
			}
		}
		if (missingParts1[i] == 0) {
			for (int j = myrand; j < citynum; j++) {
				if (d_TS_Members_DNA[second_id * citynum + j] == -1) {
					d_TS_Members_DNA[second_id * citynum + j] = i;
					break;
				}
			}
		}
	}

	// count fitness for both Members
	int fitness_counter0 = 0;
	int fitness_counter1 = 0;
	for (int i = 0; i < citynum; i++) {
		if (i > 0) {
			fitness_counter0 += d_all_distances[d_TS_Members_DNA[id * citynum + (i - 1)] * citynum + d_TS_Members_DNA[id * citynum + i]];
			fitness_counter1 += d_all_distances[d_TS_Members_DNA[second_id * citynum + (i - 1)] * citynum + d_TS_Members_DNA[second_id * citynum + i]];
		}
	}

	d_TS_Members_fitness[id] = fitness_counter0;
	d_TS_Members_fitness[second_id] = fitness_counter1;
}


__global__ void KP_Crossover_Kernel(curandState* d_states, int *d_KP_Members_DNA, int *d_KP_Members_fitness, int *d_knapsack_array,
	int *d_KP_Members_DNA_Copy, int *d_KP_chosenMembers, int sizePerDevice) {
	unsigned int id0 = threadIdx.x + (blockIdx.x * blockDim.x);// = 0-99, for a population of 1600 on 8 GPUs 
	unsigned int id1 = id0 + (blockDim.x * gridDim.x); // = id + 100, for a population of 1600 on 8 GPUs 
	if (id0 < sizePerDevice / 2 && id1 < sizePerDevice) {
		// choose a point of crossover
		int myrand = 0;
		// copy to local memory
		curandState localState = d_states[id0];
		float myrandf = curand_uniform(&localState);
		myrandf *= KP_size - 2 + 0 + 0.999999;
		myrandf += 0;
		myrand = (int)truncf(myrandf); // point of crossover
		int weight0 = 0, weight1 = 0; // weight for the first and the second member
		int fitness0 = 0, fitness1 = 0; // fitness for the first and the second member
		int member_under_work0 = d_KP_chosenMembers[id0];
		int member_under_work1 = d_KP_chosenMembers[id1];

		// cudaMemcpyAsync(void *to, void *from, size, cudaMemcpyDeviceToDevice) instead
		for (int i = 0; i < myrand; i++) {
			d_KP_Members_DNA[id0 * KP_size + i] = d_KP_Members_DNA_Copy[member_under_work0 * KP_size + i];
			d_KP_Members_DNA[id1 * KP_size + i] = d_KP_Members_DNA_Copy[member_under_work1 * KP_size + i];

			if (d_KP_Members_DNA[id0 * KP_size + i] == 1) {
				weight0 += d_knapsack_array[i * 2 + 0];
				fitness0 += d_knapsack_array[i * 2 + 1];
			}
			if (d_KP_Members_DNA[id1 * KP_size + i] == 1) {
				weight1 += d_knapsack_array[i * 2 + 0];
				fitness1 += d_knapsack_array[i * 2 + 1];
			}
		}
		for (int i = myrand; i < KP_size; i++) {
			d_KP_Members_DNA[id0 * KP_size + i] = d_KP_Members_DNA_Copy[member_under_work1 * KP_size + i];
			d_KP_Members_DNA[id1 * KP_size + i] = d_KP_Members_DNA_Copy[member_under_work0 * KP_size + i];

			if (d_KP_Members_DNA[id0 * KP_size + i] == 1) {
				weight0 += d_knapsack_array[i * 2 + 0];
				fitness0 += d_knapsack_array[i * 2 + 1];
			}
			if (d_KP_Members_DNA[id1 * KP_size + i] == 1) {
				weight1 += d_knapsack_array[i * 2 + 0];
				fitness1 += d_knapsack_array[i * 2 + 1];
			}
		}

		if (weight0 > max_weight) {
			fitness0 = 0;
		}
		if (weight1 > max_weight) {
			fitness1 = 0;
		}

		d_KP_Members_fitness[id0] = fitness0;
		d_KP_Members_fitness[id1] = fitness1;
		d_states[id0] = localState;
	}
}


__global__ void PP_Crossover_Kernel(curandState* d_states, int *d_PP_Members_DNA, int *d_PP_Members_fitness, int *d_PP_Members_DNA_Copy, int *d_PP_Members_fitness_Copy,
	int *d_partition_array, int *d_PP_chosenMembers, int sizePerDevice) {
	unsigned int id0 = threadIdx.x + (blockIdx.x * blockDim.x);// = 0-499, for an example of population of 1000 per each GPU 
	unsigned int id1 = id0 + (blockDim.x * gridDim.x); // = id + 500, for an example of population of 1000 per each GPU 
	
	if (id0 < sizePerDevice / 2 && id1 < sizePerDevice) {
		// choose a point of crossover
		int myrand = 0;
		// copy to local memory
		curandState localState = d_states[id0];
		float myrandf = curand_uniform(&localState);
		myrandf *= PP_size - 2 + 0 + 0.999999;
		myrandf += 0;
		myrand = (int)truncf(myrandf); // point of crossover
		
		// value of each partition in member0 and member1
		int total_per_part0[partitions];
		int total_per_part1[partitions];
		memset(total_per_part0, 0, sizeof(total_per_part0));
		memset(total_per_part1, 0, sizeof(total_per_part1));

		int member_under_work0 = d_PP_chosenMembers[id0];
		int member_under_work1 = d_PP_chosenMembers[id1];
		int max0 = INT_MIN, max1 = INT_MIN;
		int min0 = INT_MAX, min1 = INT_MAX;

		//printf("Member %d and %d\n", member_under_work0, member_under_work1);

		// check the fitness of to-be offsprings
		// combine the fitness of all parts of each member
		for (int i = 0; i < myrand; i++) {
			total_per_part0[d_PP_Members_DNA_Copy[member_under_work0 * PP_size + i]] += d_partition_array[i];
			total_per_part1[d_PP_Members_DNA_Copy[member_under_work1 * PP_size + i]] += d_partition_array[i];
		}
		for (int i = myrand; i < PP_size; i++) {
			total_per_part0[d_PP_Members_DNA_Copy[member_under_work1 * PP_size + i]] += d_partition_array[i];
			total_per_part1[d_PP_Members_DNA_Copy[member_under_work0 * PP_size + i]] += d_partition_array[i];
		}

		for (int i = 0; i < partitions; i++) {
			if (total_per_part0[i] > max0) {
				max0 = total_per_part0[i];
			}
			if (total_per_part1[i] > max1) {
				max1 = total_per_part1[i];
			}
			if (total_per_part0[i] < min0) {
				min0 = total_per_part0[i];
			}
			if (total_per_part1[i] < min1) {
				min1 = total_per_part1[i];
			}
		}

		int fitness0 = 0;
		int fitness1 = 0;
		if (myrand >= PP_size / 2) {
			// do a crossover
			if (fitness0 <= d_PP_Members_fitness_Copy[member_under_work0]) {
				for (int i = 0; i < myrand; i++) {
					d_PP_Members_DNA[id0 * PP_size + i] = d_PP_Members_DNA_Copy[member_under_work0 * PP_size + i];
				}
				for (int i = myrand; i < PP_size; i++) {
					d_PP_Members_DNA[id0 * PP_size + i] = d_PP_Members_DNA_Copy[member_under_work1 * PP_size + i];
				}
				d_PP_Members_fitness[id0] = max0 - min0;
			}
			else {
				for (int i = 0; i < PP_size; i++) {
					d_PP_Members_DNA[id0 * PP_size + i] = d_PP_Members_DNA_Copy[member_under_work0 * PP_size + i];
				}
				d_PP_Members_fitness[id0] = d_PP_Members_fitness_Copy[member_under_work0];
			}

			if (fitness1 <= d_PP_Members_fitness_Copy[member_under_work1]) {
				for (int i = 0; i < myrand; i++) {
					d_PP_Members_DNA[id1 * PP_size + i] = d_PP_Members_DNA_Copy[member_under_work1 * PP_size + i];
				}
				for (int i = myrand; i < PP_size; i++) {
					d_PP_Members_DNA[id1 * PP_size + i] = d_PP_Members_DNA_Copy[member_under_work0 * PP_size + i];
				}
				d_PP_Members_fitness[id1] = max1 - min1;
			}
			else {
				for (int i = 0; i < PP_size; i++) {
					d_PP_Members_DNA[id1 * PP_size + i] = d_PP_Members_DNA_Copy[member_under_work1 * PP_size + i];
				}
				d_PP_Members_fitness[id1] = d_PP_Members_fitness_Copy[member_under_work1];
			}
		}
		if (myrand < PP_size / 2) {
			if (fitness0 <= d_PP_Members_fitness_Copy[member_under_work1]) {
				for (int i = 0; i < myrand; i++) {
					d_PP_Members_DNA[id0 * PP_size + i] = d_PP_Members_DNA_Copy[member_under_work0 * PP_size + i];
				}
				for (int i = myrand; i < PP_size; i++) {
					d_PP_Members_DNA[id0 * PP_size + i] = d_PP_Members_DNA_Copy[member_under_work1 * PP_size + i];
				}
				d_PP_Members_fitness[id0] = max0 - min0;
			}
			else {
				for (int i = 0; i < PP_size; i++) {
					d_PP_Members_DNA[id0 * PP_size + i] = d_PP_Members_DNA_Copy[member_under_work1 * PP_size + i];

				}
				d_PP_Members_fitness[id0] = d_PP_Members_fitness_Copy[member_under_work1];
			}

			if (fitness1 <= d_PP_Members_fitness_Copy[member_under_work0]) {
				for (int i = 0; i < myrand; i++) {
					d_PP_Members_DNA[id1 * PP_size + i] = d_PP_Members_DNA_Copy[member_under_work1 * PP_size + i];
				}
				for (int i = myrand; i < PP_size; i++) {
					d_PP_Members_DNA[id1 * PP_size + i] = d_PP_Members_DNA_Copy[member_under_work0 * PP_size + i];
				}
				d_PP_Members_fitness[id1] = max1 - min1;
			}
			else {
				for (int i = 0; i < PP_size; i++) {
					d_PP_Members_DNA[id1 * PP_size + i] = d_PP_Members_DNA_Copy[member_under_work0 * PP_size + i];
				}
				d_PP_Members_fitness[id1] = d_PP_Members_fitness_Copy[member_under_work0];
			}
		}

		//printf("%d\n", max0 - min0);
		d_states[id0] = localState;
	}
	
}


__global__ void TS_gather_best_fitness(int* idata, int* odata, int n) {
	// shared array for reduction
	extern __shared__ int sdata[];
	// id in a block
	unsigned int thread_id = threadIdx.x;
	// overall id on the whole device
	unsigned int overall_id = threadIdx.x + blockIdx.x * blockDim.x;

	// each thread loads one element from input data (it's in global memory)
	sdata[thread_id] = idata[overall_id];
	__syncthreads();
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		if (thread_id % (s * 2) == 0) {
			// if is within size of the array
			if (sdata[thread_id + s] < n - 1) {
				// if one is smaller than the other go next
				if (sdata[thread_id] > sdata[thread_id + s]) {
					sdata[thread_id] = sdata[thread_id + s];
				}
			}

			//sdata[thread_id] += sdata[thread_id + s];
		}
		__syncthreads();
	}
	if (thread_id == 0) {
		odata[blockIdx.x] = sdata[0];/*
		printf("Best member has %d fitness\n", sdata[0]);*/
	}
	// reduction in shared memory

}


__global__ void TS_get_best_fitness(int* odata, int currentGPU, int* best, int *gen, int currentGEN) {

	int best_fitness = INT_MAX;

	for (int i = 0; i < sizeof(odata) / sizeof(int); i++) {
		if (best_fitness > odata[i]) {
			best_fitness = odata[i];
		}
	}

	if (*best > best_fitness)
	{
		*best = best_fitness;
		*gen = currentGEN;
	}

	//printf("#TravSal GPU:%d = %d\n", currentGPU, best_fitness);
}


__global__ void KP_gather_best_fitness(int* idata, int* odata, int n) {
	// shared array for reduction
	extern __shared__ int sdata[];
	// id in a block
	unsigned int thread_id = threadIdx.x;
	// overall id on the whole device
	unsigned int overall_id = threadIdx.x + blockIdx.x * blockDim.x;

	// each thread loads one element from input data (it's in global memory)
	sdata[thread_id] = idata[overall_id];
	__syncthreads();
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		if (thread_id % (s * 2) == 0) {
			// if is within size of the array
			if (sdata[thread_id + s] < n - 1) {
				// if one is smaller than the other go next
				if (sdata[thread_id] < sdata[thread_id + s]) {
					sdata[thread_id] = sdata[thread_id + s];
				}
			}

			//sdata[thread_id] += sdata[thread_id + s];
		}
		__syncthreads();
	}
	if (thread_id == 0) {
		odata[blockIdx.x] = sdata[0];/*
		printf("Best member has %d fitness\n", sdata[0]);*/
	}
	// reduction in shared memory

}


__global__ void KP_get_best_fitness(int* odata, int currentGPU, int *best, int *gen, int currentGEN) {

	int best_fitness = INT_MIN;

	for (int i = 0; i < sizeof(odata) / sizeof(int); i++) {
		if (best_fitness < odata[i]) {
			best_fitness = odata[i];
		}
	}
	if (*best < best_fitness)
	{
		*best = best_fitness;
		*gen = currentGEN;
	}

	//printf("##Knapsack GPU:%d = %d\n", currentGPU, best_fitness);
}


__global__ void PP_gather_best_fitness(int* idata, int* odata, int n) {
	// shared array for reduction
	extern __shared__ int sdata[];
	// id in a block
	unsigned int thread_id = threadIdx.x;
	// overall id on the whole device
	unsigned int overall_id = threadIdx.x + blockIdx.x * blockDim.x;

	// each thread loads one element from input data (it's in global memory)
	sdata[thread_id] = idata[overall_id];
	__syncthreads();
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		if (thread_id % (s * 2) == 0) {
			// if is within size of the array
			if (sdata[thread_id + s] < n - 1) {
				// if one is smaller than the other go next
				if (sdata[thread_id] > sdata[thread_id + s]) {
					sdata[thread_id] = sdata[thread_id + s];
				}
			}

			//sdata[thread_id] += sdata[thread_id + s];
		}
		__syncthreads();
	}
	if (thread_id == 0) {
		odata[blockIdx.x] = sdata[0];/*
		printf("Best member has %d fitness\n", sdata[0]);*/
	}
	// reduction in shared memory

}


__global__ void PP_get_best_fitness(int* odata, int currentGPU, int *best, int *gen, int currentGEN) {

	int best_fitness = INT_MAX;

	for (int i = 0; i < sizeof(odata) / sizeof(int); i++) {
		if (best_fitness > odata[i]) {
			best_fitness = odata[i];
		}
	}
	if (*best > best_fitness)
	{
		*best = best_fitness;
		*gen = currentGEN;
	}
	//printf("###Partition GPU:%d = %d\n", currentGPU, best_fitness);
}


__global__ void TS_mutate(curandState *state, int* d_TS_Members_DNA, int* d_TS_Members_fitness, int *d_all_distances) {
	unsigned int id = threadIdx.x + (blockIdx.x * blockDim.x);
	curandState localState = state[id];
	float rand_num = curand_uniform(&localState);
	
	if(rand_num < mutation_prob){
		
		// random indexes for the cities
		int city1 = curand(&localState) % citynum;
		int city2 = curand(&localState) % citynum;
		
		// swap the cities
		int temp = d_TS_Members_DNA[id * citynum + city1];
		d_TS_Members_DNA[id * citynum + city1] = d_TS_Members_DNA[id * citynum + city2];
		d_TS_Members_DNA[id * citynum + city2] = temp;
		
		// calculate fitness
		int fitness_counter = 0;
		for (int i = 0; i < citynum; i++) {
			if (i > 0) {
				fitness_counter += d_all_distances[d_TS_Members_DNA[id * citynum + (i - 1)] * citynum + d_TS_Members_DNA[id * citynum + i]];
			}
		}
		d_TS_Members_fitness[id] = fitness_counter;
	}
	// saving state back to global memory
	state[id] = localState;
}

// TODO: fix the fitness issue, either calculate fitness here each time or improve the fitness function
__global__ void KP_mutate(curandState *state, int* d_KP_Members_DNA, int* d_KP_Members_fitness, int* d_knapsack_array) {
	unsigned int id = threadIdx.x + (blockIdx.x * blockDim.x);
	curandState localState = state[id];
	float rand_num = curand_uniform(&localState);
	
	if(rand_num < mutation_prob){
		int myrand = 0;
		float myrandf = curand_uniform(&localState);
		myrandf *= KP_size - 1 + 0 + 0.999999;
		myrand = (int)truncf(myrandf);
		if (d_KP_Members_DNA[id * KP_size + myrand] != 1) {
			d_KP_Members_DNA[id * KP_size + myrand] = 1;
		}
		else {
			d_KP_Members_DNA[id * KP_size + myrand] = 0;
		}
		
		int overall_weight = 0;
		int overall_fitness = 0;
		for(int i = 0; i < KP_size; i++) {
			if(d_KP_Members_DNA[id * KP_size + i] == 1){
				overall_weight += d_knapsack_array[i * 2 + 0];
				overall_fitness += d_knapsack_array[i * 2  + 1];
			}
			if(overall_weight > max_weight) {
				overall_fitness = 0;
				break;
			}
		}
		d_KP_Members_fitness[id] = overall_fitness;
	}
	// saving state back to global memory
	state[id] = localState;
	
	
	
}


__global__ void PP_mutate(curandState *state, int* d_PP_Members_DNA, int* d_PP_Members_fitness, int* d_partition_array) {
	unsigned int id = threadIdx.x + (blockIdx.x * blockDim.x);
	curandState localState = state[id];
	float rand_num = curand_uniform(&localState);
	
	if(rand_num < mutation_prob) {
		float current_indexf = curand_uniform(&localState);
		current_indexf *= KP_size - 1 + 0 + 0.999999;
		int current_index = (int)truncf(current_indexf);
		
		float random_valuef = curand_uniform(&localState);
		int random_value = (random_valuef < 0.5f) ? 0 : 1;
		// dependent on the current value change randomly to 1 others
		if (d_PP_Members_DNA[id * PP_size + current_index] == 0) {
			d_PP_Members_DNA[id * PP_size + current_index] = (random_value) + 1;
		}
		else if (d_PP_Members_DNA[id * PP_size + current_index] == 1) {
			d_PP_Members_DNA[id * PP_size + current_index] = (random_value) * 2;
		}
		else {
			d_PP_Members_DNA[id * PP_size + current_index] = (random_value);
		}
		
		// setup for fitness recalculation
		int total_per_part[partitions];
		memset(total_per_part, 0, sizeof(total_per_part));
		
		for(int i = 0; i < PP_size; i++){
			total_per_part[d_PP_Members_DNA[id * PP_size + i]] += d_partition_array[i];
		}
		
		// recalculate the fitness
		int max_value = INT_MIN;
		int min_value = INT_MAX;
		for (int i = 0; i < partitions; i++) {
			if (total_per_part[i] > max_value) {
				max_value = total_per_part[i];
			}
			if (total_per_part[i] < min_value) {
				min_value = total_per_part[i];
			}
		}
		d_PP_Members_fitness[id] = max_value - min_value;
	}
	
	// saving state back to global memory
	state[id] = localState;

}


__global__ void TS_calculate_fitness(int* d_TS_Members_DNA, int* d_TS_Members_fitness){
	
}


__global__ void KP_calculate_fitness(int* d_KP_Members_DNA, int* d_KP_Members_fitness){
	
}


__global__ void PP_calculate_fitness(int* d_PP_Members_DNA, int* d_PP_Members_fitness){
}


extern int all_distances[citynum][citynum];
extern int knapsack_array[KP_size][2];
extern int partition_array[PP_size];

int main(int argc, char* argv[])
{
	auto start = std::chrono::high_resolution_clock::now();
	genetic_algorithm(std::stoi(argv[1]), std::stoi(argv[2]), std::stoi(argv[3]), all_distances, knapsack_array, partition_array);
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds> (stop - start);
	int seconds = duration.count() / 1000;
	int milliseconds = duration.count() % 1000;
	//printf("The duration is %d.%d seconds\n", seconds, milliseconds);
	return 0;
}

// Function for using CUDA to run a genetic algorithm in parallel.
void genetic_algorithm(int whichAlgo, int nDevices, int powerCap, int all_distances[][citynum], int knapsack_array[][2], int partition_array[PP_size])
{
	// variable that will hold the number of GPU devices
	//int nDevices = 0;
	// get the number of devices in the system
	//cudaGetDeviceCount(&nDevices);
	// set number of threads to the number of current devices
	omp_set_num_threads(nDevices);

	// initializations
	int sizePerBlock = 128;
	int block_num_per_GPU = 1 + ((pop_size - 1) / (sizePerBlock));
	dim3 numberOfBlocks(block_num_per_GPU);
	//printf("numberOfBlocks is equal to %d\n", numberOfBlocks);
	dim3 threadsPerBlock(sizePerBlock);
	
	dim3 threadsPerBlockHalved = (sizePerBlock / 2);

	//printf("Size of array per device is equal to %d\n", pop_size);
	// initialize curand state for random number generation
	curandState** d_states = new curandState*[nDevices];
	cudaStream_t* streams = new cudaStream_t[nDevices];

	// EVERYONE
	int* odata[nDevices];
	int sizeChunk = pop_size / nDevices;
	int h_Int[nDevices];
	int h_Gen[nDevices];
	if (whichAlgo == 1) {
		for (int i = 0; i < nDevices; i++) {
			h_Int[i] = INT_MIN;
		}
		
	}
	else {
		for (int i = 0; i < nDevices; i++) {
			h_Int[i] = INT_MAX;
		}
	}

	int *d_Int[nDevices];
	int* d_Gen[nDevices];

	// traveling salesman problem
	if (whichAlgo == 0) {
		// FOR TRAVELING SALESMAN
		int* d_TS_Members_DNA[nDevices];
		int* d_all_distances[nDevices];
		int* d_TS_Members_fitness[nDevices];

		int* d_TS_chosenMembers[nDevices];
		int* d_TS_Members_DNA_Copy[nDevices];
		int* d_TS_Members_fitness_Copy[nDevices];

		

		//printf("!!!This is generation number %d!!!\n", 0);
		#pragma omp parallel
		{
			int currentThread = omp_get_thread_num();

			unsigned long long seed = std::time(0) * (currentThread + 1);
			unsigned long long* d_seed;

			// set up a GPU device on which we will be working on
			gpuCheckErrors(cudaSetDevice(currentThread));

			// create a stream for our current context
			gpuCheckErrors(cudaStreamCreate(&streams[currentThread]));

			// create Peer access between two gpus
			if (nDevices > 1) {
				if (currentThread == 0) {
					gpuCheckErrors(cudaDeviceEnablePeerAccess(nDevices - 1, 0));
				}
				else {
					gpuCheckErrors(cudaDeviceEnablePeerAccess(currentThread - 1, 0));
				}
			}

			// allocate memory on the gpu for d_states[currentThread]
			gpuCheckErrors(cudaMallocAsync(&d_states[currentThread], sizeof(curandState) * pop_size, streams[currentThread]));

			// allocate memory on the gpu for d_seed
			gpuCheckErrors(cudaMallocAsync(&d_seed, sizeof(unsigned long long), streams[currentThread]));

			gpuCheckErrors(cudaMemcpyAsync(d_seed, &seed, sizeof(unsigned long long), cudaMemcpyHostToDevice, streams[currentThread]));

			// run setup kernel
			setup_kernel << < numberOfBlocks, threadsPerBlock, 0, streams[currentThread] >> > (d_states[currentThread], d_seed);

			//printf("setup_kernel for device %d finished :D\n", currentThread);

			// cudamallocasyncs

			gpuCheckErrors(cudaMallocAsync((void**)&d_all_distances[currentThread], sizeof(int) * citynum * citynum, streams[currentThread]));

			gpuCheckErrors(cudaMallocAsync((void**)&d_TS_chosenMembers[currentThread], sizeof(int) * pop_size, streams[currentThread]));

			gpuCheckErrors(cudaMallocAsync((void**)&odata[currentThread], sizeof(int) * block_num_per_GPU, streams[currentThread]));

			gpuCheckErrors(cudaDeviceSynchronize());

			// cudamallocs

			gpuCheckErrors(cudaMalloc((void**)&d_Int[currentThread], sizeof(int)));

			gpuCheckErrors(cudaMalloc((void**)&d_Gen[currentThread], sizeof(int)));

			gpuCheckErrors(cudaMalloc((void**)&d_TS_Members_DNA[currentThread], sizeof(int) * pop_size * citynum));

			gpuCheckErrors(cudaMalloc((void**)&d_TS_Members_DNA_Copy[currentThread], sizeof(int) * pop_size * citynum));

			gpuCheckErrors(cudaMalloc((void**)&d_TS_Members_fitness[currentThread], sizeof(int) * pop_size));

			gpuCheckErrors(cudaMalloc((void**)&d_TS_Members_fitness_Copy[currentThread], sizeof(int) * pop_size));

			// cudamemcpys

			gpuCheckErrors(cudaMemcpyAsync(d_all_distances[currentThread], all_distances, sizeof(int) * citynum * citynum, cudaMemcpyHostToDevice, streams[currentThread]));
			
			gpuCheckErrors(cudaMemcpyAsync(d_Int[currentThread], &h_Int[currentThread], sizeof(int), cudaMemcpyHostToDevice, streams[currentThread]));

			TS_doGAKernel << < numberOfBlocks, threadsPerBlock, 0, streams[currentThread] >> > (d_states[currentThread], d_TS_Members_DNA[currentThread], d_TS_Members_fitness[currentThread], d_all_distances[currentThread]);
			
			TS_mutate <<< numberOfBlocks, threadsPerBlock, 0, streams[currentThread] >>> (d_states[currentThread], d_TS_Members_DNA[currentThread], d_TS_Members_fitness[currentThread], d_all_distances[currentThread]);
			
			gpuCheckErrors(cudaMemcpyAsync(d_TS_Members_fitness_Copy[currentThread], d_TS_Members_fitness[currentThread], sizeof(int) * pop_size, cudaMemcpyDeviceToDevice, streams[currentThread]));

			gpuCheckErrors(cudaMemcpyAsync(d_TS_Members_DNA_Copy[currentThread], d_TS_Members_DNA[currentThread], sizeof(int) * pop_size * citynum, cudaMemcpyDeviceToDevice, streams[currentThread]));

			TS_gather_best_fitness << <numberOfBlocks, threadsPerBlock, sizePerBlock * sizeof(int), streams[currentThread] >> > (d_TS_Members_fitness[currentThread], odata[currentThread], pop_size);

			TS_get_best_fitness << < 1, 1, 0, streams[currentThread] >> > (odata[currentThread], currentThread, d_Int[currentThread], d_Gen[currentThread], 0);

			gpuCheckErrors(cudaStreamSynchronize(streams[currentThread]));
		
		}

		for (int i = 1; i <= generations; i++) {
			//printf("!!!This is generation number %d!!!\n", i);
			#pragma omp parallel
			{
				int currentThread = omp_get_thread_num();

				gpuCheckErrors(cudaSetDevice(currentThread));

				TS_TournamentSelection_Kernel << < numberOfBlocks, threadsPerBlock, 0, streams[currentThread] >> > (d_states[currentThread], d_TS_Members_fitness_Copy[currentThread], d_TS_chosenMembers[currentThread]);

				TS_Crossover_Kernel << < numberOfBlocks, threadsPerBlockHalved, 0, streams[currentThread] >> > (d_states[currentThread], d_TS_Members_DNA[currentThread],
					d_TS_Members_fitness[currentThread], d_all_distances[currentThread], d_TS_Members_DNA_Copy[currentThread], d_TS_chosenMembers[currentThread]);
				
				TS_mutate <<< numberOfBlocks, threadsPerBlock, 0, streams[currentThread] >>> (d_states[currentThread], d_TS_Members_DNA[currentThread], d_TS_Members_fitness[currentThread], d_all_distances[currentThread]);
				
				gpuCheckErrors(cudaStreamSynchronize(streams[currentThread]));

				if (i % p2p_send_intervals == 0) {
					int gpuToGetFrom = currentThread - 1;

					if (currentThread == 0) {
						gpuToGetFrom = nDevices - 1;
					}
					gpuCheckErrors(cudaMemcpyAsync(&d_TS_Members_DNA[currentThread][sizeChunk * gpuToGetFrom * citynum], 
						&d_TS_Members_DNA[gpuToGetFrom][sizeChunk * gpuToGetFrom * citynum], 
						sizeof(int) * sizeChunk * citynum / 2, cudaMemcpyDefault, streams[currentThread]));
					gpuCheckErrors(cudaMemcpyAsync(&d_TS_Members_fitness[currentThread][sizeChunk * gpuToGetFrom], 
						&d_TS_Members_fitness[gpuToGetFrom][sizeChunk * gpuToGetFrom], 
						sizeof(int) * sizeChunk / 2, cudaMemcpyDefault, streams[currentThread]));

					gpuCheckErrors(cudaStreamSynchronize(streams[currentThread]));
				}

				TS_gather_best_fitness << < numberOfBlocks, threadsPerBlock, sizePerBlock * sizeof(int), streams[currentThread] >> > (d_TS_Members_fitness[currentThread], odata[currentThread], pop_size);

				TS_get_best_fitness << < 1, 1, 0, streams[currentThread] >> > (odata[currentThread], currentThread, d_Int[currentThread], d_Gen[currentThread], i);

				gpuCheckErrors(cudaStreamSynchronize(streams[currentThread]));

				gpuCheckErrors(cudaMemcpyAsync(d_TS_Members_fitness_Copy[currentThread], d_TS_Members_fitness[currentThread], sizeof(int)* pop_size, cudaMemcpyDeviceToDevice, streams[currentThread]));

				gpuCheckErrors(cudaMemcpyAsync(d_TS_Members_DNA_Copy[currentThread], d_TS_Members_DNA[currentThread], sizeof(int)* pop_size* citynum, cudaMemcpyDeviceToDevice, streams[currentThread]));

				gpuCheckErrors(cudaStreamSynchronize(streams[currentThread]));

			}
		}

		#pragma omp parallel
		{
			int currentThread = omp_get_thread_num();
			gpuCheckErrors(cudaSetDevice(currentThread));
			gpuCheckErrors(cudaMemcpyAsync(&h_Int[currentThread], d_Int[currentThread], sizeof(int), cudaMemcpyDeviceToHost, streams[currentThread]));
			gpuCheckErrors(cudaMemcpyAsync(&h_Gen[currentThread], d_Gen[currentThread], sizeof(int), cudaMemcpyDeviceToHost, streams[currentThread]));
		}
		int best_one = h_Int[0];
		int genZ = h_Gen[0];
		for (int i = 1; i < nDevices; i++) {
			if (h_Int[i] < best_one) {
				best_one = h_Int[i];
				genZ = h_Gen[i];
			}
		}
		//printf("Best fitness of power %d was accomplished in generation %d\n", best_one, genZ);
		
		std::ofstream outFile;
		if (nDevices == 1) {
			if (powerCap == 260) {
				outFile.open("TS1GPU.csv", std::ios::app);
			}
			else if (powerCap == 220) {
				outFile.open("TS1GPU220.csv", std::ios::app);
			}
			else if (powerCap == 180) {
				outFile.open("TS1GPU180.csv", std::ios::app);
			}
			else if (powerCap == 140) {
				outFile.open("TS1GPU140.csv", std::ios::app);
			}
			else if (powerCap == 100) {
				outFile.open("TS1GPU100.csv", std::ios::app);
			}
			
		}
		else if (nDevices == 2) {
			if (powerCap == 260) {
				outFile.open("TS2GPU.csv", std::ios::app);
			}
			else if (powerCap == 220) {
				outFile.open("TS2GPU220.csv", std::ios::app);
			}
			else if (powerCap == 180) {
				outFile.open("TS2GPU180.csv", std::ios::app);
			}
			else if (powerCap == 140) {
				outFile.open("TS2GPU140.csv", std::ios::app);
			}
			else if (powerCap == 100) {
				outFile.open("TS2GPU100.csv", std::ios::app);
			}
		}
		else if (nDevices == 4) {
			if (powerCap == 260) {
				outFile.open("TS4GPU.csv", std::ios::app);
			}
			else if (powerCap == 220) {
				outFile.open("TS4GPU220.csv", std::ios::app);
			}
			else if (powerCap == 180) {
				outFile.open("TS4GPU180.csv", std::ios::app);
			}
			else if (powerCap == 140) {
				outFile.open("TS4GPU140.csv", std::ios::app);
			}
			else if (powerCap == 100) {
				outFile.open("TS4GPU100.csv", std::ios::app);
			}
		}
		else if (nDevices == 8) {
			if (powerCap == 260) {
				outFile.open("TS8GPU.csv", std::ios::app);
			}
			else if (powerCap == 220) {
				outFile.open("TS8GPU220.csv", std::ios::app);
			}
			else if (powerCap == 180) {
				outFile.open("TS8GPU180.csv", std::ios::app);
			}
			else if (powerCap == 140) {
				outFile.open("TS8GPU140.csv", std::ios::app);
			}
			else if (powerCap == 100) {
				outFile.open("TS8GPU100.csv", std::ios::app);
			}
		}

		// Check if the file is open
		if (outFile.is_open()) {
			// Write the variables to the file
			outFile << best_one << "," << genZ << std::endl;
			// Close the file
			outFile.close();
		}
		else {
			std::cerr << "Unable to open file for writing." << std::endl;
		}

		#pragma omp parallel
		{
			int currentThread = omp_get_thread_num();
			gpuCheckErrors(cudaSetDevice(currentThread));
			gpuCheckErrors(cudaFreeAsync(odata[currentThread], streams[currentThread]));
			gpuCheckErrors(cudaFreeAsync(d_states[currentThread], streams[currentThread]));
			gpuCheckErrors(cudaFreeAsync(d_all_distances[currentThread], streams[currentThread]));
			gpuCheckErrors(cudaFreeAsync(d_TS_Members_fitness[currentThread], streams[currentThread]));
			gpuCheckErrors(cudaFreeAsync(d_TS_Members_fitness_Copy[currentThread], streams[currentThread]));
			gpuCheckErrors(cudaFreeAsync(d_TS_Members_DNA[currentThread], streams[currentThread]));
			gpuCheckErrors(cudaFreeAsync(d_TS_Members_DNA_Copy[currentThread], streams[currentThread]));
			gpuCheckErrors(cudaFreeAsync(d_TS_chosenMembers[currentThread], streams[currentThread]));
			gpuCheckErrors(cudaStreamSynchronize(streams[currentThread]));
			gpuCheckErrors(cudaStreamDestroy(streams[currentThread]));

		}
	}
	// knapsack problem
	else if (whichAlgo == 1) {
		// FOR KNAPSACK PROBLEM
		int* d_KP_Members_DNA[nDevices];
		int* d_knapsack_array[nDevices];
		int* d_KP_Members_fitness[nDevices];

		int* d_KP_Members_DNA_Copy[nDevices];
		int* d_KP_Members_fitness_Copy[nDevices];
		int* d_KP_chosenMembers[nDevices];

		//printf("!!!This is generation number %d!!!\n", 0);
		#pragma omp parallel
		{
			int currentThread = omp_get_thread_num();

			unsigned long long seed = std::time(0) * (currentThread + 1);
			unsigned long long* d_seed;

			// set up a GPU device on which we will be working on
			gpuCheckErrors(cudaSetDevice(currentThread));

			// create a stream for our current context
			gpuCheckErrors(cudaStreamCreate(&streams[currentThread]));

			// create Peer access between two gpus
			if (nDevices > 1) {
				if (currentThread == 0) {
					gpuCheckErrors(cudaDeviceEnablePeerAccess(nDevices - 1, 0));
				}
				else {
					gpuCheckErrors(cudaDeviceEnablePeerAccess(currentThread - 1, 0));
				}
			}

			// allocate memory on the gpu for d_states[currentThread]
			gpuCheckErrors(cudaMallocAsync(&d_states[currentThread], sizeof(curandState) * pop_size, streams[currentThread]));

			// allocate memory on the gpu for d_seed
			gpuCheckErrors(cudaMallocAsync(&d_seed, sizeof(unsigned long long), streams[currentThread]));

			gpuCheckErrors(cudaMemcpyAsync(d_seed, &seed, sizeof(unsigned long long), cudaMemcpyHostToDevice, streams[currentThread]));

			// run setup kernel
			setup_kernel << < numberOfBlocks, threadsPerBlock, 0, streams[currentThread] >> > (d_states[currentThread], d_seed);

			//printf("setup_kernel for device %d finished :D\n", currentThread);

			// cudamallocasyncs

			gpuCheckErrors(cudaMallocAsync((void**)&d_knapsack_array[currentThread], sizeof(int) * KP_size * 2, streams[currentThread]));

			gpuCheckErrors(cudaMallocAsync((void**)&d_KP_chosenMembers[currentThread], sizeof(int) * pop_size, streams[currentThread]));

			gpuCheckErrors(cudaMallocAsync((void**)&odata[currentThread], sizeof(int)* block_num_per_GPU, streams[currentThread]));

			gpuCheckErrors(cudaDeviceSynchronize());

			// cudamallocs

			gpuCheckErrors(cudaMalloc((void**)&d_Int[currentThread], sizeof(int)));

			gpuCheckErrors(cudaMalloc((void**)&d_Gen[currentThread], sizeof(int)));

			gpuCheckErrors(cudaMalloc((void**)&d_KP_Members_DNA[currentThread], sizeof(int)* pop_size* KP_size));

			gpuCheckErrors(cudaMalloc((void**)&d_KP_Members_fitness[currentThread], sizeof(int)* pop_size));

			gpuCheckErrors(cudaMalloc((void**)&d_KP_Members_DNA_Copy[currentThread], sizeof(int)* pop_size* KP_size));

			gpuCheckErrors(cudaMalloc((void**)&d_KP_Members_fitness_Copy[currentThread], sizeof(int)* pop_size));

			// cudamemcpys

			gpuCheckErrors(cudaMemcpyAsync(d_Int[currentThread], &h_Int, sizeof(int), cudaMemcpyHostToDevice, streams[currentThread]));

			gpuCheckErrors(cudaMemcpyAsync(d_knapsack_array[currentThread], knapsack_array, sizeof(int)* KP_size * 2, cudaMemcpyHostToDevice, streams[currentThread]));
		
			KP_doGAKernel << < numberOfBlocks, threadsPerBlock, 0, streams[currentThread] >> > (d_states[currentThread], d_KP_Members_DNA[currentThread], d_KP_Members_fitness[currentThread], d_knapsack_array[currentThread]);
			
			KP_mutate <<< numberOfBlocks, threadsPerBlock, 0, streams[currentThread] >>> (d_states[currentThread], d_KP_Members_DNA[currentThread], d_KP_Members_fitness[currentThread], d_knapsack_array[currentThread]);
			
			gpuCheckErrors(cudaMemcpyAsync(d_KP_Members_fitness_Copy[currentThread], d_KP_Members_fitness[currentThread], sizeof(int)* pop_size, cudaMemcpyDeviceToDevice, streams[currentThread]));

			gpuCheckErrors(cudaMemcpyAsync(d_KP_Members_DNA_Copy[currentThread], d_KP_Members_DNA[currentThread], sizeof(int)* pop_size * KP_size, cudaMemcpyDeviceToDevice, streams[currentThread]));

			KP_gather_best_fitness << <numberOfBlocks, threadsPerBlock, sizePerBlock * sizeof(int), streams[currentThread] >> > (d_KP_Members_fitness[currentThread], odata[currentThread], pop_size);

			KP_get_best_fitness << < 1, 1, 0, streams[currentThread] >> > (odata[currentThread], currentThread, d_Int[currentThread], d_Gen[currentThread], 0);

			gpuCheckErrors(cudaStreamSynchronize(streams[currentThread]));
		}

		for (int i = 1; i <= generations; i++) {
			//printf("!!!This is generation number %d!!!\n", i);

			#pragma omp parallel
			{
				int currentThread = omp_get_thread_num();

				gpuCheckErrors(cudaSetDevice(currentThread));

				KP_TournamentSelection_Kernel << <numberOfBlocks, threadsPerBlock, 0, streams[currentThread] >> > (d_states[currentThread], d_KP_chosenMembers[currentThread], d_KP_Members_fitness[currentThread]);

				KP_Crossover_Kernel << <numberOfBlocks, threadsPerBlockHalved, 0, streams[currentThread] >> > (d_states[currentThread], d_KP_Members_DNA[currentThread], d_KP_Members_fitness[currentThread],
					d_knapsack_array[currentThread], d_KP_Members_DNA_Copy[currentThread], d_KP_chosenMembers[currentThread], pop_size);
				
				KP_mutate <<< numberOfBlocks, threadsPerBlock, 0, streams[currentThread] >>> (d_states[currentThread], d_KP_Members_DNA[currentThread], d_KP_Members_fitness[currentThread], d_knapsack_array[currentThread]);
				
				gpuCheckErrors(cudaStreamSynchronize(streams[currentThread]));

				if (i % p2p_send_intervals == 0) {
					int gpuToGetFrom = currentThread - 1;

					if (currentThread == 0) {
						gpuToGetFrom = nDevices - 1;
					}
					gpuCheckErrors(cudaMemcpyAsync(&d_KP_Members_DNA[currentThread][sizeChunk * gpuToGetFrom * KP_size], &d_KP_Members_DNA[gpuToGetFrom][sizeChunk * gpuToGetFrom * KP_size], sizeof(int) * sizeChunk * KP_size / 2, cudaMemcpyDefault, streams[currentThread]));
					gpuCheckErrors(cudaMemcpyAsync(&d_KP_Members_fitness[currentThread][sizeChunk * gpuToGetFrom], &d_KP_Members_fitness[gpuToGetFrom][sizeChunk * gpuToGetFrom], sizeof(int) * sizeChunk / 2, cudaMemcpyDefault, streams[currentThread]));

					gpuCheckErrors(cudaStreamSynchronize(streams[currentThread]));
				}

				KP_gather_best_fitness << <numberOfBlocks, threadsPerBlock, sizePerBlock * sizeof(int), streams[currentThread] >> > (d_KP_Members_fitness[currentThread], odata[currentThread], pop_size);

				KP_get_best_fitness << < 1, 1, 0, streams[currentThread] >> > (odata[currentThread], currentThread, d_Int[currentThread], d_Gen[currentThread], i);

				gpuCheckErrors(cudaStreamSynchronize(streams[currentThread]));

				gpuCheckErrors(cudaMemcpyAsync(d_KP_Members_fitness_Copy[currentThread], d_KP_Members_fitness[currentThread], sizeof(int) * pop_size, cudaMemcpyDeviceToDevice, streams[currentThread]));

				gpuCheckErrors(cudaMemcpyAsync(d_KP_Members_DNA_Copy[currentThread], d_KP_Members_DNA[currentThread], sizeof(int) * pop_size * KP_size, cudaMemcpyDeviceToDevice, streams[currentThread]));

				gpuCheckErrors(cudaStreamSynchronize(streams[currentThread]));

			}
		}

		#pragma omp parallel
		{
			int currentThread = omp_get_thread_num();
			gpuCheckErrors(cudaSetDevice(currentThread));
			gpuCheckErrors(cudaMemcpyAsync(&h_Int[currentThread], d_Int[currentThread], sizeof(int), cudaMemcpyDeviceToHost, streams[currentThread]));
			gpuCheckErrors(cudaMemcpyAsync(&h_Gen[currentThread], d_Gen[currentThread], sizeof(int), cudaMemcpyDeviceToHost, streams[currentThread]));
		}
		int best_one = h_Int[0];
		int genZ = h_Gen[0];
		for (int i = 1; i < nDevices; i++) {
			if (h_Int[i] > best_one) {
				best_one = h_Int[i];
				genZ = h_Gen[i];
			}
		}
		//printf("Best fitness of power %d was accomplished in generation %d\n", best_one, genZ);

		std::ofstream outFile;
		if (nDevices == 1) {
			if (powerCap == 260) {
				outFile.open("KP1GPU.csv", std::ios::app);
			}
			else if (powerCap == 220) {
				outFile.open("KP1GPU220.csv", std::ios::app);
			}
			else if (powerCap == 180) {
				outFile.open("KP1GPU180.csv", std::ios::app);
			}
			else if (powerCap == 140) {
				outFile.open("KP1GPU140.csv", std::ios::app);
			}
			else if (powerCap == 100) {
				outFile.open("KP1GPU100.csv", std::ios::app);
			}
		}
		else if (nDevices == 2) {
			if (powerCap == 260) {
				outFile.open("KP2GPU.csv", std::ios::app);
			}
			else if (powerCap == 220) {
				outFile.open("KP2GPU220.csv", std::ios::app);
			}
			else if (powerCap == 180) {
				outFile.open("KP2GPU180.csv", std::ios::app);
			}
			else if (powerCap == 140) {
				outFile.open("KP2GPU140.csv", std::ios::app);
			}
			else if (powerCap == 100) {
				outFile.open("KP2GPU100.csv", std::ios::app);
			}
		}
		else if (nDevices == 4) {
			if (powerCap == 260) {
				outFile.open("KP4GPU.csv", std::ios::app);
			}
			else if (powerCap == 220) {
				outFile.open("KP4GPU220.csv", std::ios::app);
			}
			else if (powerCap == 180) {
				outFile.open("KP4GPU180.csv", std::ios::app);
			}
			else if (powerCap == 140) {
				outFile.open("KP4GPU140.csv", std::ios::app);
			}
			else if (powerCap == 100) {
				outFile.open("KP4GPU100.csv", std::ios::app);
			}
		}
		else if (nDevices == 8) {
			if (powerCap == 260) {
				outFile.open("KP8GPU.csv", std::ios::app);
			}
			else if (powerCap == 220) {
				outFile.open("KP8GPU220.csv", std::ios::app);
			}
			else if (powerCap == 180) {
				outFile.open("KP8GPU180.csv", std::ios::app);
			}
			else if (powerCap == 140) {
				outFile.open("KP8GPU140.csv", std::ios::app);
			}
			else if (powerCap == 100) {
				outFile.open("KP8GPU100.csv", std::ios::app);
			}
		}

		// Check if the file is open
		if (outFile.is_open()) {
			// Write the variables to the file
			outFile << best_one << "," << genZ << std::endl;
			// Close the file
			outFile.close();
		}
		else {
			std::cerr << "Unable to open file for writing." << std::endl;
		}


		#pragma omp parallel
		{
			int currentThread = omp_get_thread_num();
			gpuCheckErrors(cudaSetDevice(currentThread));
			gpuCheckErrors(cudaFreeAsync(odata[currentThread], streams[currentThread]));
			gpuCheckErrors(cudaFreeAsync(d_states[currentThread], streams[currentThread]));
			gpuCheckErrors(cudaFreeAsync(d_knapsack_array[currentThread], streams[currentThread]));
			gpuCheckErrors(cudaFreeAsync(d_KP_Members_fitness[currentThread], streams[currentThread]));
			gpuCheckErrors(cudaFreeAsync(d_KP_Members_fitness_Copy[currentThread], streams[currentThread]));
			gpuCheckErrors(cudaFreeAsync(d_KP_Members_DNA[currentThread], streams[currentThread]));
			gpuCheckErrors(cudaFreeAsync(d_KP_Members_DNA_Copy[currentThread], streams[currentThread]));
			gpuCheckErrors(cudaFreeAsync(d_KP_chosenMembers[currentThread], streams[currentThread]));
			gpuCheckErrors(cudaStreamSynchronize(streams[currentThread]));
			gpuCheckErrors(cudaStreamDestroy(streams[currentThread]));
		}

	}

	// partition problem 
	else if (whichAlgo == 2) {
		// FOR PARTITION PROBLEM
		int* d_PP_Members_DNA[nDevices];
		int* d_PP_Members_fitness[nDevices];
		int* d_partition_array[nDevices];

		int* d_PP_Members_DNA_Copy[nDevices];
		int* d_PP_Members_fitness_Copy[nDevices];
		int* d_PP_chosenMembers[nDevices];

		//printf("!!!This is generation number %d!!!\n", 0);
		#pragma omp parallel
		{
			int currentThread = omp_get_thread_num();

			unsigned long long seed = std::time(0) * (currentThread + 1);
			unsigned long long* d_seed;

			// set up a GPU device on which we will be working on
			gpuCheckErrors(cudaSetDevice(currentThread));

			// create a stream for our current context
			gpuCheckErrors(cudaStreamCreate(&streams[currentThread]));

			// create Peer access between two gpus
			if (nDevices > 1) {
				if (currentThread == 0) {
					gpuCheckErrors(cudaDeviceEnablePeerAccess(nDevices - 1, 0));
				}
				else {
					gpuCheckErrors(cudaDeviceEnablePeerAccess(currentThread - 1, 0));
				}
			}
			

			// allocate memory on the gpu for d_states[currentThread]
			gpuCheckErrors(cudaMallocAsync(&d_states[currentThread], sizeof(curandState)* pop_size, streams[currentThread]));

			// allocate memory on the gpu for d_seed
			gpuCheckErrors(cudaMallocAsync(&d_seed, sizeof(unsigned long long), streams[currentThread]));

			gpuCheckErrors(cudaMemcpyAsync(d_seed, &seed, sizeof(unsigned long long), cudaMemcpyHostToDevice, streams[currentThread]));

			// run setup kernel
			setup_kernel << < numberOfBlocks, threadsPerBlock, 0, streams[currentThread] >> > (d_states[currentThread], d_seed);

			//printf("setup_kernel for device %d finished :D\n", currentThread);

			// cudamallocasyncs

			gpuCheckErrors(cudaMallocAsync((void**)&d_partition_array[currentThread], sizeof(int)* PP_size, streams[currentThread]));

			gpuCheckErrors(cudaMallocAsync((void**)&d_PP_chosenMembers[currentThread], sizeof(int)* pop_size, streams[currentThread]));

			gpuCheckErrors(cudaMallocAsync((void**)&odata[currentThread], sizeof(int)* block_num_per_GPU, streams[currentThread]));

			gpuCheckErrors(cudaDeviceSynchronize());

			// cudamallocs

			gpuCheckErrors(cudaMalloc((void**)&d_Int[currentThread], sizeof(int)));

			gpuCheckErrors(cudaMalloc((void**)&d_Gen[currentThread], sizeof(int)));

			gpuCheckErrors(cudaMalloc((void**)&d_PP_Members_DNA[currentThread], sizeof(int) * pop_size * PP_size));

			gpuCheckErrors(cudaMalloc((void**)&d_PP_Members_DNA_Copy[currentThread], sizeof(int) * pop_size * PP_size));

			gpuCheckErrors(cudaMalloc((void**)&d_PP_Members_fitness[currentThread], sizeof(int)* pop_size));

			gpuCheckErrors(cudaMalloc((void**)&d_PP_Members_fitness_Copy[currentThread], sizeof(int)* pop_size));

			//cudamemcpys

			gpuCheckErrors(cudaMemcpyAsync(d_Int[currentThread], &h_Int, sizeof(int), cudaMemcpyHostToDevice, streams[currentThread]));

			gpuCheckErrors(cudaMemcpyAsync(d_partition_array[currentThread], partition_array, sizeof(int)* PP_size, cudaMemcpyHostToDevice, streams[currentThread]));
			
			PP_doGAKernel << < numberOfBlocks, threadsPerBlock, 0, streams[currentThread] >> > (d_states[currentThread], d_PP_Members_DNA[currentThread], d_PP_Members_fitness[currentThread], d_partition_array[currentThread], pop_size);

			PP_mutate <<< numberOfBlocks, threadsPerBlock, 0, streams[currentThread] >>> (d_states[currentThread], d_PP_Members_DNA[currentThread], d_PP_Members_fitness[currentThread], d_partition_array[currentThread]);

			gpuCheckErrors(cudaMemcpyAsync(d_PP_Members_fitness_Copy[currentThread], d_PP_Members_fitness[currentThread], sizeof(int)* pop_size, cudaMemcpyDeviceToDevice, streams[currentThread]));

			gpuCheckErrors(cudaMemcpyAsync(d_PP_Members_DNA_Copy[currentThread], d_PP_Members_DNA[currentThread], sizeof(int)* pop_size* PP_size, cudaMemcpyDeviceToDevice, streams[currentThread]));

			PP_gather_best_fitness << <numberOfBlocks, threadsPerBlock, sizePerBlock * sizeof(int), streams[currentThread] >> > (d_PP_Members_fitness[currentThread], odata[currentThread], pop_size);

			PP_get_best_fitness << < 1, 1, 0, streams[currentThread] >> > (odata[currentThread], currentThread, d_Int[currentThread], d_Gen[currentThread], 0);

			gpuCheckErrors(cudaStreamSynchronize(streams[currentThread]));

		}

		for (int i = 1; i <= generations; i++) {
			//printf("!!!This is generation number %d!!!\n", i);

			#pragma omp parallel
			{
				int currentThread = omp_get_thread_num();

				gpuCheckErrors(cudaSetDevice(currentThread));

				PP_TournamentSelection_Kernel << < numberOfBlocks, threadsPerBlock, 0, streams[currentThread] >> > (d_states[currentThread], d_PP_chosenMembers[currentThread], d_PP_Members_fitness[currentThread], pop_size);

				PP_Crossover_Kernel << < numberOfBlocks, threadsPerBlockHalved, 0, streams[currentThread] >> > (d_states[currentThread], d_PP_Members_DNA[currentThread],
					d_PP_Members_fitness[currentThread], d_PP_Members_DNA_Copy[currentThread], d_PP_Members_fitness_Copy[currentThread], d_partition_array[currentThread], d_PP_chosenMembers[currentThread], pop_size);
				
				PP_mutate <<< numberOfBlocks, threadsPerBlock, 0, streams[currentThread] >>> (d_states[currentThread], d_PP_Members_DNA[currentThread], d_PP_Members_fitness[currentThread], d_partition_array[currentThread]);
				
				gpuCheckErrors(cudaStreamSynchronize(streams[currentThread]));

				if (i % p2p_send_intervals == 0) {
					int gpuToGetFrom = currentThread - 1;

					if (currentThread == 0) {
						gpuToGetFrom = nDevices - 1;
					}

					gpuCheckErrors(cudaMemcpyAsync(&d_PP_Members_DNA[currentThread][sizeChunk * gpuToGetFrom * PP_size], &d_PP_Members_DNA[gpuToGetFrom][sizeChunk * gpuToGetFrom * PP_size], sizeof(int) * sizeChunk * PP_size / 2, cudaMemcpyDefault, streams[currentThread]));
					gpuCheckErrors(cudaMemcpyAsync(&d_PP_Members_fitness[currentThread][sizeChunk * gpuToGetFrom], &d_PP_Members_fitness[gpuToGetFrom][sizeChunk * gpuToGetFrom], sizeof(int) * sizeChunk / 2, cudaMemcpyDefault, streams[currentThread]));

					gpuCheckErrors(cudaStreamSynchronize(streams[currentThread]));
				}

				PP_gather_best_fitness << <numberOfBlocks, threadsPerBlock, sizePerBlock * sizeof(int), streams[currentThread] >> > (d_PP_Members_fitness[currentThread], odata[currentThread], pop_size);

				PP_get_best_fitness << < 1, 1, 0, streams[currentThread] >> > (odata[currentThread], currentThread, d_Int[currentThread], d_Gen[currentThread], i);

				gpuCheckErrors(cudaStreamSynchronize(streams[currentThread]));

				gpuCheckErrors(cudaMemcpyAsync(d_PP_Members_fitness_Copy[currentThread], d_PP_Members_fitness[currentThread], sizeof(int) * pop_size, cudaMemcpyDeviceToDevice, streams[currentThread]));

				gpuCheckErrors(cudaMemcpyAsync(d_PP_Members_DNA_Copy[currentThread], d_PP_Members_DNA[currentThread], sizeof(int) * pop_size * PP_size, cudaMemcpyDeviceToDevice, streams[currentThread]));

				gpuCheckErrors(cudaStreamSynchronize(streams[currentThread]));
			}
		}

		#pragma omp parallel
		{
			int currentThread = omp_get_thread_num();
			gpuCheckErrors(cudaSetDevice(currentThread));
			gpuCheckErrors(cudaMemcpyAsync(&h_Int[currentThread], d_Int[currentThread], sizeof(int), cudaMemcpyDeviceToHost, streams[currentThread]));
			gpuCheckErrors(cudaMemcpyAsync(&h_Gen[currentThread], d_Gen[currentThread], sizeof(int), cudaMemcpyDeviceToHost, streams[currentThread]));
		}
		int best_one = h_Int[0];
		int genZ = h_Gen[0];
		for (int i = 1; i < nDevices; i++) {
			if (h_Int[i] < best_one) {
				best_one = h_Int[i];
				genZ = h_Gen[i];
			}
		}
		//printf("Best fitness of power %d was accomplished in generation %d\n", best_one, genZ);

		std::ofstream outFile;
		if (nDevices == 1) {
			if (powerCap == 260) {
				outFile.open("PP1GPU.csv", std::ios::app);
			}
			else if (powerCap == 220) {
				outFile.open("PP1GPU220.csv", std::ios::app);
			}
			else if (powerCap == 180) {
				outFile.open("PP1GPU180.csv", std::ios::app);
			}
			else if (powerCap == 140) {
				outFile.open("PP1GPU140.csv", std::ios::app);
			}
			else if (powerCap == 100) {
				outFile.open("PP1GPU100.csv", std::ios::app);
			}
			
		}
		else if (nDevices == 2) {
			if (powerCap == 260) {
				outFile.open("PP2GPU.csv", std::ios::app);
			}
			else if (powerCap == 220) {
				outFile.open("PP2GPU220.csv", std::ios::app);
			}
			else if (powerCap == 180) {
				outFile.open("PP2GPU180.csv", std::ios::app);
			}
			else if (powerCap == 140) {
				outFile.open("PP2GPU140.csv", std::ios::app);
			}
			else if (powerCap == 100) {
				outFile.open("PP2GPU100.csv", std::ios::app);
			}
		}
		else if (nDevices == 4) {
			if (powerCap == 260) {
				outFile.open("PP4GPU.csv", std::ios::app);
			}
			else if (powerCap == 220) {
				outFile.open("PP4GPU220.csv", std::ios::app);
			}
			else if (powerCap == 180) {
				outFile.open("PP4GPU180.csv", std::ios::app);
			}
			else if (powerCap == 140) {
				outFile.open("PP4GPU140.csv", std::ios::app);
			}
			else if (powerCap == 100) {
				outFile.open("PP4GPU100.csv", std::ios::app);
			}
		}
		else if (nDevices == 8) {
			if (powerCap == 260) {
				outFile.open("PP8GPU.csv", std::ios::app);
			}
			else if (powerCap == 220) {
				outFile.open("PP8GPU220.csv", std::ios::app);
			}
			else if (powerCap == 180) {
				outFile.open("PP8GPU180.csv", std::ios::app);
			}
			else if (powerCap == 140) {
				outFile.open("PP8GPU140.csv", std::ios::app);
			}
			else if (powerCap == 100) {
				outFile.open("PP8GPU100.csv", std::ios::app);
			}
		}

		// Check if the file is open
		if (outFile.is_open()) {
			// Write the variables to the file
			outFile << best_one << "," << genZ << std::endl;
			// Close the file
			outFile.close();
		}
		else {
			std::cerr << "Unable to open file for writing." << std::endl;
		}


		#pragma omp parallel
		{
			int currentThread = omp_get_thread_num();
			gpuCheckErrors(cudaSetDevice(currentThread));
			gpuCheckErrors(cudaFreeAsync(d_states[currentThread], streams[currentThread]));
			gpuCheckErrors(cudaFreeAsync(odata[currentThread], streams[currentThread]));
			gpuCheckErrors(cudaFreeAsync(d_partition_array[currentThread], streams[currentThread]));
			gpuCheckErrors(cudaFreeAsync(d_PP_Members_fitness[currentThread], streams[currentThread]));
			gpuCheckErrors(cudaFreeAsync(d_PP_Members_fitness_Copy[currentThread], streams[currentThread]));
			gpuCheckErrors(cudaFreeAsync(d_PP_Members_DNA[currentThread], streams[currentThread]));
			gpuCheckErrors(cudaFreeAsync(d_PP_Members_DNA_Copy[currentThread], streams[currentThread]));
			gpuCheckErrors(cudaFreeAsync(d_PP_chosenMembers[currentThread], streams[currentThread]));
			gpuCheckErrors(cudaStreamSynchronize(streams[currentThread]));
			gpuCheckErrors(cudaStreamDestroy(streams[currentThread]));
		}

	}

}
