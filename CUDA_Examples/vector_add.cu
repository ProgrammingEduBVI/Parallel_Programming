/*
 * CUDA exmaple: Vector additions in parallel. Code based on NVIDIA's CUDA tutorial..
 *
 * Author: Wei Wang <wei.wang@utsa.edu>
 */

#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>

#include "cuda_error_chk.h"

__global__ void add(int *a, int *b, int *c) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	c[index] = a[index] + b[index];

	return;
}

#define N (2048*2048)
#define THREADS_PER_BLOCK 512
int main(void) {
	int *a, *b, *c; // host copies of a, b, c
	int *d_a, *d_b, *d_c; // device copies of a, b, c
	int size = N * sizeof(int);
	int i;
	struct timeval start, end;

	// Allocate space for a, b, c on CPU
	a = (int*)malloc(size);
	b = (int*)malloc(size);
	c = (int*)malloc(size);
        // Setup input values
	for(i = 0; i < N; i++){
		a[i] = 1; b[i] = 2;
	}

	gettimeofday(&start, NULL);
	// Allocate space for device copies of a, b, c
	gpuErrchk(cudaMalloc((void **)&d_a, size));
	gpuErrchk(cudaMalloc((void **)&d_b, size));
	gpuErrchk(cudaMalloc((void **)&d_c, size));
		// Copy inputs to device
	gpuErrchk(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice));

	// rest of the code on next page
	// continue from previous page
	// Launch add() kernel on GPU with N blocks
	add<<<N/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(d_a,d_b, d_c);
	gpuErrchk( cudaPeekAtLastError() );

	// Copy result back to host
	gpuErrchk(cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost));
	// Cleanup
	gpuErrchk(cudaFree(d_a)); 
	gpuErrchk(cudaFree(d_b)); 
	gpuErrchk(cudaFree(d_c));
	gettimeofday(&end, NULL);

	// verfiy results
	for(i = 0; i < N; i++){
		if(c[i] != 3){
			printf("Result incorrect\n");
			return 1;
		}
	}
	printf("Results are correct.\n");
	printf("Execution time is %lf s\n", ((end.tv_sec + ((double)(end.tv_usec))/1000000)
		  - (start.tv_sec + ((double)(start.tv_usec))/1000000)));


	return 0;
}
