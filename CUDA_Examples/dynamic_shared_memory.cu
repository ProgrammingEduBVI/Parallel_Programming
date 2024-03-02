/*
 * CUDA example: All reduction with dynamically allocated shared memory.
 *
 * Author: Wei Wang <wei.wang@utsa.edu>
 */

#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>

#include "cuda_error_chk.h"

#define N (2048*2048)
#define THREADS_PER_BLOCK 512

__global__ void all_reduce(int *in, int *out) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
	
        // allocate an array in shared memory
	extern __shared__ int temp[];
	
        // copy on value into shared memory
        temp[threadIdx.x] = in[index];
        __syncthreads(); // barrier
	
        // sum all values up
        int sum = 0;
        for(int i = 0; i < blockDim.x; i++)
                sum += temp[i];
        // output the sum to out array
        out[index] = sum;

	return;
}


int main(void) {
        int *in, * out; // host copies of in and out
	int *d_in, *d_out; // device copies of in and out
	int size = N * sizeof(int);
	int i;
	struct timeval start, end;

	// Allocate space for a, b, c on CPU
	in = (int*)malloc(size);
	out = (int*)malloc(size);
	
        // Setup input values
	for(i = 0; i < N; i++){
		in[i] = 1;
		out[i] = 0;;
	}

	gettimeofday(&start, NULL);
	
	// Allocate space for device copies of in and out
	gpuErrchk(cudaMalloc((void **)&d_in, size));
	gpuErrchk(cudaMalloc((void **)&d_out, size));
	
	// Copy inputs to device
	gpuErrchk(cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_out, out, size, cudaMemcpyHostToDevice));
	
	// rest of the code on next page
	// continue from previous page
	// Launch add() kernel on GPU with N blocks
	all_reduce<<<N/THREADS_PER_BLOCK,THREADS_PER_BLOCK,
		THREADS_PER_BLOCK*sizeof(int)>>>(d_in, d_out);
	gpuErrchk( cudaPeekAtLastError() );
	
	// Copy result back to host
	gpuErrchk(cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost));
	
	// Cleanup
	gpuErrchk(cudaFree(d_in)); 
	gpuErrchk(cudaFree(d_out));
	gettimeofday(&end, NULL);

	// verfiy results
	for(i = 0; i < N; i++){
		if(out[i] != THREADS_PER_BLOCK){
			printf("Result incorrect\n");
			return 1;
		}
	}
	printf("Results are correct.\n");
	printf("Execution time is %lf s\n", ((end.tv_sec + ((double)(end.tv_usec))/1000000)
		  - (start.tv_sec + ((double)(start.tv_usec))/1000000)));


	return 0;
}
