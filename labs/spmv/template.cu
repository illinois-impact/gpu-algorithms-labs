#include <wb.h>
#include <chrono>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void spmvCSRKernel(float *out, int *matCols, int *matRows,
                              float *matData, float *vec, int dim) {
  //@@ insert spmv kernel for csr format

	  unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < dim) {

    float result = 0.0f;
    unsigned int start = matRows[row];
    unsigned int end = matRows[row + 1];

    for (int elemIdx = start; elemIdx < end; ++elemIdx) {
      unsigned int colIdx = matCols[elemIdx];
      result += matData[elemIdx] * vec[colIdx];
    }

    out[row] = result;
  }

}

__global__ void spmvJDSKernel(float *out, int *matColStart, int *matCols,
                              int *matRowPerm, int *matRows,
                              float *matData, float *vec, int dim) {
  //@@ insert spmv kernel for jds format
	  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < dim) {

    unsigned int row = matRowPerm[idx];
    float result = 0.0f;
    unsigned int rowNNZ = matRows[idx];
    for (unsigned int nzIdx = 0; nzIdx < rowNNZ; ++nzIdx) {
      unsigned int elemIdx = matColStart[nzIdx] + idx;
      unsigned int colIdx = matCols[elemIdx];
      result += matData[elemIdx] * vec[colIdx];
    }
    out[row] = result;
  }

}

static void spmvCSR(float *out, int *matCols, int *matRows, float *matData,
                    float *vec, int dim) {

  //@@ invoke spmv kernel for csr format

	  const unsigned int THREADS_PER_BLOCK = 512;
  const unsigned int numBlocks = (dim - 1) / THREADS_PER_BLOCK + 1;
  spmvCSRKernel<<<numBlocks, THREADS_PER_BLOCK>>>(out, matCols, matRows,
                                                  matData, vec, dim);

}

static void spmvJDS(float *out, int *matColStart, int *matCols,
                    int *matRowPerm, int *matRows, float *matData,
                    float *vec, int dim) {

  //@@ invoke spmv kernel for jds format

	  const unsigned int THREADS_PER_BLOCK = 512;
  const unsigned int numBlocks = (dim - 1) / THREADS_PER_BLOCK + 1;
  spmvJDSKernel<<<numBlocks, THREADS_PER_BLOCK>>>(
      out, matColStart, matCols, matRowPerm, matRows, matData, vec, dim);

}

int main(int argc, char **argv) {
  wbArg_t args;
  bool usingJDSQ;
  int *hostCSRCols;
  int *hostCSRRows;
  float *hostCSRData;
  int *hostJDSColStart;
  int *hostJDSCols;
  int *hostJDSRowPerm;
  int *hostJDSRows;
  float *hostJDSData;
  float *hostVector;
  float *hostOutput;
  int *deviceCSRCols;
  int *deviceCSRRows;
  float *deviceCSRData;
  int *deviceJDSColStart;
  int *deviceJDSCols;
  int *deviceJDSRowPerm;
  int *deviceJDSRows;
  float *deviceJDSData;
  float *deviceVector;
  float *deviceOutput;
  int dim, ncols, nrows, ndata;
  int maxRowNNZ;

  args = wbArg_read(argc, argv);

  printf("Importing data and creating memory on host\n");
  usingJDSQ = wbImport_flag(wbArg_getInputFile(args, 0)) == 1;
  hostCSRCols =
      (int *)wbImport(wbArg_getInputFile(args, 1), &ncols, "Integer");
  hostCSRRows =
      (int *)wbImport(wbArg_getInputFile(args, 2), &nrows, "Integer");
  hostCSRData =
      (float *)wbImport(wbArg_getInputFile(args, 3), &ndata, "Real");
  hostVector =
      (float *)wbImport(wbArg_getInputFile(args, 4), &dim, "Real");

  hostOutput = (float *)malloc(sizeof(float) * dim);

  if (usingJDSQ) {
    CSRToJDS(dim, hostCSRRows, hostCSRCols, hostCSRData, &hostJDSRowPerm,
             &hostJDSRows, &hostJDSColStart, &hostJDSCols, &hostJDSData);
    maxRowNNZ = hostJDSRows[0];
  }

  printf("Allocating GPU memory.\n");
  if (usingJDSQ) {
    cudaMalloc((void **)&deviceJDSColStart, sizeof(int) * maxRowNNZ);
    cudaMalloc((void **)&deviceJDSCols, sizeof(int) * ndata);
    cudaMalloc((void **)&deviceJDSRowPerm, sizeof(int) * dim);
    cudaMalloc((void **)&deviceJDSRows, sizeof(int) * dim);
    cudaMalloc((void **)&deviceJDSData, sizeof(float) * ndata);
  } else {
    cudaMalloc((void **)&deviceCSRCols, sizeof(int) * ncols);
    cudaMalloc((void **)&deviceCSRRows, sizeof(int) * nrows);
    cudaMalloc((void **)&deviceCSRData, sizeof(float) * ndata);
  }
  cudaMalloc((void **)&deviceVector, sizeof(float) * dim);
  cudaMalloc((void **)&deviceOutput, sizeof(float) * dim);

  printf("Copying input memory to the GPU.\n");
  if (usingJDSQ) {
    cudaMemcpy(deviceJDSColStart, hostJDSColStart, sizeof(int) * maxRowNNZ,
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceJDSCols, hostJDSCols, sizeof(int) * ndata,
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceJDSRowPerm, hostJDSRowPerm, sizeof(int) * dim,
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceJDSRows, hostJDSRows, sizeof(int) * dim,
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceJDSData, hostJDSData, sizeof(float) * ndata,
               cudaMemcpyHostToDevice);
  } else {
    cudaMemcpy(deviceCSRCols, hostCSRCols, sizeof(int) * ncols,
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceCSRRows, hostCSRRows, sizeof(int) * nrows,
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceCSRData, hostCSRData, sizeof(float) * ndata,
               cudaMemcpyHostToDevice);
  }
  cudaMemcpy(deviceVector, hostVector, sizeof(float) * dim,
             cudaMemcpyHostToDevice);

  typedef std::chrono::high_resolution_clock Clock;
  typedef std::chrono::duration<double> Duration;
  printf("Performing CUDA computation\n");
  auto start = Clock::now();
  if (usingJDSQ) {
    spmvJDS(deviceOutput, deviceJDSColStart, deviceJDSCols,
            deviceJDSRowPerm, deviceJDSRows, deviceJDSData, deviceVector,
            dim);
  } else {
    spmvCSR(deviceOutput, deviceCSRCols, deviceCSRRows, deviceCSRData,
            deviceVector, dim);
  }
  cudaDeviceSynchronize();
  Duration elapsed = Clock::now() - start;
  std::cout << elapsed.count() << " seconds\n";

  printf("Copying output memory to the CPU\n");
  cudaMemcpy(hostOutput, deviceOutput, sizeof(float) * dim,
             cudaMemcpyDeviceToHost);

  printf("Freeing GPU Memory\n");
  cudaFree(deviceCSRCols);
  cudaFree(deviceCSRRows);
  cudaFree(deviceCSRData);
  cudaFree(deviceVector);
  cudaFree(deviceOutput);
  if (usingJDSQ) {
    cudaFree(deviceJDSColStart);
    cudaFree(deviceJDSCols);
    cudaFree(deviceJDSRowPerm);
    cudaFree(deviceJDSRows);
    cudaFree(deviceJDSData);
  }

  wbSolution(args, hostOutput, dim);

  printf("ncols: %d, nrows: %d, ndata: %d, dim: %d\n",ncols,nrows,ndata,dim);
  return 0;

  free(hostCSRCols);
  free(hostCSRRows);
  free(hostCSRData);
  free(hostVector);
  free(hostOutput);
  if (usingJDSQ) {
    free(hostJDSColStart);
    free(hostJDSCols);
    free(hostJDSRowPerm);
    free(hostJDSRows);
    free(hostJDSData);
  }

  return 0;
}
