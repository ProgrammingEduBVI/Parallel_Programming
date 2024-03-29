// This is the REAL "hello world" for CUDA!
// It takes the string "Hello ", prints it, then passes it to CUDA with an array
// of offsets. Then the offsets are added in parallel to produce the string "World!"
// By Ingemar Ragnemalm 2010
 
#include <stdio.h>

#include "cuda_error_chk.h"

const int N = 7;
const int blocksize = 7;

__global__
void hello(char *a, int *b)
{
 a[threadIdx.x] += b[threadIdx.x];
}

int main()
{
 char a[N] = "Hello ";
 int b[N] = {15, 10, 6, 0, -11, 1, 0};

 char *ad;
 int *bd;
 const int csize = N*sizeof(char);
 const int isize = N*sizeof(int);

 printf("%s", a);

 gpuErrchk(cudaMalloc( (void**)&ad, csize ));
 gpuErrchk(cudaMalloc( (void**)&bd, isize ));
 gpuErrchk(cudaMemcpy( ad, a, csize, cudaMemcpyHostToDevice ));
 gpuErrchk(cudaMemcpy( bd, b, isize, cudaMemcpyHostToDevice ));

 dim3 dimBlock( blocksize, 1 );
 dim3 dimGrid( 1, 1 );
 hello<<<dimGrid, dimBlock>>>(ad, bd);
 gpuErrchk( cudaPeekAtLastError() );

 gpuErrchk(cudaMemcpy( a, ad, csize, cudaMemcpyDeviceToHost ));
 gpuErrchk(cudaFree( ad ));
 gpuErrchk(cudaFree( bd ));

 printf("%s\n", a);
 return EXIT_SUCCESS;
}
