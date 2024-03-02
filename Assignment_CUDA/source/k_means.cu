
/*
 * Skeleton function for Parallel Computing Class, 
 * Assignment: K-Means Algorithm (CUDA)
 *
 * To students: You should finish the implementation of k_means algorithm.
 *              You should add device functions/kernels to perform k_means on 
 *              GPU. The "k_means" function in this file is just an interface
 *              for passing in basic parameters needed.. You need to add GPU 
 *              kernels and launch them in the "k_means" function.
 *
 *              Note that the "k_means" function has two input parameters for
 *              block count and thread count per block. Please use these two
 *              parameters when launching your kernels.
 * 
 * Author:
 *     Wei Wang <wei.wang@utsa.edu>
 */
#include <stdio.h>
#include <float.h>
#include <math.h>

#include "k_means.h"


/*
 * k_means: k_means clustering algorithm implementation.
 *
 * Input parameters:
 *     struct point p[]: array of data points
 *     int m           : number of data points in p[]
 *     int k           : number of clusters to find
 *     int iters       : number of clustering iterations to run
 *     int block_cnt   : number of blocks to use
 *     int threads_per_block: number of threads per block
 *
 * Output parameters:   
 *     struct point u[]: array of cluster centers
 *     int c[]         : cluster id for each data points
 */
void k_means(struct point p[MAX_POINTS], 
	     int m, 
	     int k,
	     int iters,
	     struct point u[MAX_CENTERS],
	     int c[MAX_POINTS],
	     int block_cnt,
	     int threads_per_block)
{
	int j;
	
	/* randomly initialized the centers */
	for(j = 0; j < k; j++)
		u[j] = random_center(p);
	/*
	 * To students: This function is not a GPU kernel. You need to add your
	 * own kernel functions in this file. Block count and thread count per
	 * count are provided as parameters to this functions. Please use these
	 * two parameters when launching your kernels.
	 */
	
  	return;
}
