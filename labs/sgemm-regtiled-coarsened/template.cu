#include <cstdio>
#include <cstdlib>

#include "template.hu"

#define TILE_SZ_A 128
#define TILE_SZ_B 16
#define TILE_SZ_RATIO (TILE_SZ_A/TILE_SZ_B)

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float* C) {

  /********************************************************************
  *
  * Compute C = A x B
  *   where A is a (m x k) matrix
  *   where B is a (k x n) matrix
  *   where C is a (m x n) matrix
  *
  * Use register and shared memory tiling and thread coarsening
  *
  * NOTE: A and C are column major, B is row major
  *
  ********************************************************************/

  // Macros for accessing flattened matrices
  #define A(row,col) A[(row) + (col)*m]
  #define B(row,col) B[(row)*n + (col)]
  #define C(row,col) C[(row) + (col)*m]

  // INSERT KERNEL CODE HERE

}

void basicSgemm(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
    if ((transa != 'N') && (transa != 'n')) {
	printf("unsupported value of 'transa'\n");
    	return;
    }

    if ((transb != 'T') && (transb != 't')) {
	printf("unsupported value of 'transb'\n");
	return;
    }

    if ((alpha - 1.0f > 1e-10) || (alpha - 1.0f < -1e-10)) {
	printf("unsupported value of alpha\n");
	return;
    }

    if ((beta - 0.0f > 1e-10) || (beta - 0.0f < -1e-10)) {
	printf("unsupported value of beta\n");
	return;
    }

    // Initialize thread block and kernel grid dimensions ---------------------

    //INSERT CODE HERE

    // Invoke CUDA kernel -----------------------------------------------------

    //INSERT CODE HERE

}

