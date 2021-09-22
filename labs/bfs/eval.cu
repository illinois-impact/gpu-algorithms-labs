/*! \brief File containing evaluation and grading code

This file contains evaluation and grading code.
This code is in a separate file so that a known-good version can be used for
automatic grading and students should not need to modify it.
*/

#include "helper.hpp"
#include "template.hu"

namespace gpu_algorithms_labs_evaluation {

enum Mode { CPU = 1, GPU_GLOBAL_QUEUE, GPU_BLOCK_QUEUE, GPU_WARP_QUEUE };

void cpu_queueing(unsigned int *nodePtrs, unsigned int *nodeNeighbors,
                 unsigned int *nodeVisited, unsigned int *currLevelNodes,
                 unsigned int *nextLevelNodes, unsigned int *numCurrLevelNodes,
                 unsigned int *numNextLevelNodes) {

  // Loop over all nodes in the current level
  for (unsigned int idx = 0; idx < *numCurrLevelNodes; ++idx) {
    unsigned int node = currLevelNodes[idx];
    // Loop over all neighbors of the node
    for (unsigned int nbrIdx = nodePtrs[node]; nbrIdx < nodePtrs[node + 1];
         ++nbrIdx) {
      unsigned int neighbor = nodeNeighbors[nbrIdx];
      // If the neighbor hasn't been visited yet
      if (!nodeVisited[neighbor]) {
        // Mark it and add it to the queue
        nodeVisited[neighbor] = 1;
        nextLevelNodes[*numNextLevelNodes] = neighbor;
        ++(*numNextLevelNodes);
      }
    }
  }
}




static int eval(const int numNodes, const int maxNeighborsPerNode, Mode mode) {

  // Initialize host variables
  // ----------------------------------------------

  // Variables
  unsigned int *nodePtrs_h;
  unsigned int *nodeNeighbors_h;
  unsigned int *totalNeighbors_h;
  unsigned int *nodeVisited_h;
  unsigned int *nodeVisited_ref;
  unsigned int *currLevelNodes_h;
  unsigned int *nextLevelNodes_h;
  unsigned int *nextLevelNodes_ref;
  unsigned int *numCurrLevelNodes_h;
  unsigned int *numNextLevelNodes_h;
  unsigned int *numNextLevelNodes_ref;
  // GPU
  unsigned int *nodePtrs_d;
  unsigned int *nodeNeighbors_d;
  unsigned int *nodeVisited_d;
  unsigned int *currLevelNodes_d;
  unsigned int *nextLevelNodes_d;
  unsigned int *numCurrLevelNodes_d;
  unsigned int *numNextLevelNodes_d;

  ///////////////////////////////////////////////////////

  timer_start("Generating test data");

  //pupulating arrays
  setupProblem(numNodes, maxNeighborsPerNode, &nodePtrs_h, &nodeNeighbors_h,
               &totalNeighbors_h, &nodeVisited_h, &nodeVisited_ref,
               &currLevelNodes_h, &nextLevelNodes_h, &nextLevelNodes_ref,
               &numCurrLevelNodes_h, &numNextLevelNodes_h,
               &numNextLevelNodes_ref);
  //calculating reference solution
  compute(numNodes, nodePtrs_h, nodeNeighbors_h, nodeVisited_h, nodeVisited_ref,
          currLevelNodes_h, nextLevelNodes_ref, numCurrLevelNodes_h,
          numNextLevelNodes_ref);

  timer_stop(); // Generating test data

  timer_start("Allocating GPU memory.");

  // Allocate device variables
  // ----------------------------------------------

  if (mode != CPU) {

    CUDA_RUNTIME(cudaMalloc((void **)&nodePtrs_d,
                            (numNodes + 1) * sizeof(unsigned int)));

    CUDA_RUNTIME(
        cudaMalloc((void **)&nodeVisited_d, numNodes * sizeof(unsigned int)));

    CUDA_RUNTIME(cudaMalloc((void **)&nodeNeighbors_d,
                            (*totalNeighbors_h) * sizeof(unsigned int)));

    CUDA_RUNTIME(
        cudaMalloc((void **)&numCurrLevelNodes_d, sizeof(unsigned int)));

    CUDA_RUNTIME(cudaMalloc((void **)&currLevelNodes_d,
                            (*numCurrLevelNodes_h) * sizeof(unsigned int)));

    CUDA_RUNTIME(
        cudaMalloc((void **)&numNextLevelNodes_d, sizeof(unsigned int)));

    CUDA_RUNTIME(cudaMalloc((void **)&nextLevelNodes_d,
                            (numNodes) * sizeof(unsigned int)));
  }

  timer_stop(); // Allocating GPU memory

  timer_start("Copying inputs to the GPU.");

  if (mode != CPU) {

    CUDA_RUNTIME(cudaMemcpy(nodePtrs_d, nodePtrs_h,
                            (numNodes + 1) * sizeof(unsigned int),
                            cudaMemcpyHostToDevice));

    CUDA_RUNTIME(cudaMemcpy(nodeVisited_d, nodeVisited_h,
                            numNodes * sizeof(unsigned int),
                            cudaMemcpyHostToDevice));

    CUDA_RUNTIME(cudaMemcpy(nodeNeighbors_d, nodeNeighbors_h,
                            (*totalNeighbors_h) * sizeof(unsigned int),
                            cudaMemcpyHostToDevice));

    CUDA_RUNTIME(cudaMemcpy(numCurrLevelNodes_d, numCurrLevelNodes_h,
                            sizeof(unsigned int), cudaMemcpyHostToDevice));

    CUDA_RUNTIME(cudaMemcpy(currLevelNodes_d, currLevelNodes_h,
                            (*numCurrLevelNodes_h) * sizeof(unsigned int),
                            cudaMemcpyHostToDevice));

    CUDA_RUNTIME(
        cudaMemset(nextLevelNodes_d, 0, (numNodes) * sizeof(unsigned int)));

    CUDA_RUNTIME(cudaMemset(numNextLevelNodes_d, 0, sizeof(unsigned int)));
  }

  timer_stop();

  timer_start("Performing bfs");

  if (mode == CPU) {
    cpu_queueing(nodePtrs_h, nodeNeighbors_h, nodeVisited_h, currLevelNodes_h,
                nextLevelNodes_h, numCurrLevelNodes_h, numNextLevelNodes_h);
  } else if (mode == GPU_GLOBAL_QUEUE) {
    gpu_global_queueing(nodePtrs_d, nodeNeighbors_d, nodeVisited_d,
                       currLevelNodes_d, nextLevelNodes_d, numCurrLevelNodes_d,
                       numNextLevelNodes_d);
  } else if (mode == GPU_BLOCK_QUEUE) {
    gpu_block_queueing(nodePtrs_d, nodeNeighbors_d, nodeVisited_d,
                      currLevelNodes_d, nextLevelNodes_d, numCurrLevelNodes_d,
                      numNextLevelNodes_d);
  } else if (mode == GPU_WARP_QUEUE) {
    gpu_warp_queueing(nodePtrs_d, nodeNeighbors_d, nodeVisited_d,
                     currLevelNodes_d, nextLevelNodes_d, numCurrLevelNodes_d,
                     numNextLevelNodes_d);
  } else {
    // printf("Invalid mode!\n");
    // exit(0);
  }

  if (mode != CPU) {
    CUDA_RUNTIME(cudaDeviceSynchronize());
  }
  timer_stop();

  timer_start("Copying output to the CPU");

  if (mode != CPU) {

    CUDA_RUNTIME(cudaMemcpy(numNextLevelNodes_h, numNextLevelNodes_d,
                            sizeof(unsigned int), cudaMemcpyDeviceToHost));

    CUDA_RUNTIME(cudaMemcpy(nextLevelNodes_h, nextLevelNodes_d,
                            numNodes * sizeof(unsigned int),
                            cudaMemcpyDeviceToHost));
  }
  // CUDA_RUNTIME(cudaDeviceSynchronize());
  timer_stop();

  // verify with provided implementation
  timer_start("Verifying results");
  verify(numNextLevelNodes_ref, nextLevelNodes_ref, numNextLevelNodes_h,
         nextLevelNodes_h);
  timer_stop();

  free(nodePtrs_h);
  free(nodeNeighbors_h);
  free(totalNeighbors_h);
  free(nodeVisited_h);
  free(nodeVisited_ref);
  free(currLevelNodes_h);
  free(nextLevelNodes_h);
  free(nextLevelNodes_ref);
  free(numCurrLevelNodes_h);
  free(numNextLevelNodes_h);
  free(numNextLevelNodes_ref);

  if (mode != CPU) {
    CUDA_RUNTIME(cudaFree(nodePtrs_d));
    CUDA_RUNTIME(cudaFree(nodeVisited_d));
    CUDA_RUNTIME(cudaFree(nodeNeighbors_d));
    CUDA_RUNTIME(cudaFree(numCurrLevelNodes_d));
    CUDA_RUNTIME(cudaFree(currLevelNodes_d));
    CUDA_RUNTIME(cudaFree(numNextLevelNodes_d));
    CUDA_RUNTIME(cudaFree(nextLevelNodes_d));
  }

  return 0;
}

TEST_CASE("GQ", "[GPU_GLOBAL_QUEUE]") {

  SECTION("1023, 2, GPU_GLOBAL_QUEUE") { eval(1023, 2, GPU_GLOBAL_QUEUE); }
  SECTION("2047, 4, GPU_GLOBAL_QUEUE") { eval(2047, 4, GPU_GLOBAL_QUEUE); }
  SECTION("4095, 8, GPU_GLOBAL_QUEUE") { eval(4095, 8, GPU_GLOBAL_QUEUE); }
}

TEST_CASE("BQ", "[GPU_BLOCK_QUEUE]") {

  SECTION("1023, 2, GPU_GLOBAL_QUEUE") { eval(1023, 2, GPU_BLOCK_QUEUE); }
  SECTION("2047, 4, GPU_GLOBAL_QUEUE") { eval(2047, 4, GPU_BLOCK_QUEUE); }
  SECTION("4095, 8, GPU_GLOBAL_QUEUE") { eval(4095, 8, GPU_BLOCK_QUEUE); }
}

TEST_CASE("WQ", "[bfs]") {

  SECTION("1023, 2, GPU_GLOBAL_QUEUE") { eval(1023, 2, GPU_WARP_QUEUE); }
  SECTION("2047, 4, GPU_GLOBAL_QUEUE") { eval(2047, 4, GPU_WARP_QUEUE); }
  SECTION("4095, 8, GPU_GLOBAL_QUEUE") { eval(4095, 8, GPU_WARP_QUEUE); }
}
} // namespace gpu_algorithms_labs_evaluation
