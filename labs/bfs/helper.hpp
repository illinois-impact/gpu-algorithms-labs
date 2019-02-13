#pragma once

#define CATCH_CONFIG_CPP11_TO_STRING
#define CATCH_CONFIG_COLOUR_ANSI
#define CATCH_CONFIG_MAIN

#include "common/catch.hpp"
#include "common/fmt.hpp"
#include "common/utils.hpp"

#include "assert.h"
#include "stdint.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"

#include <algorithm>
#include <random>
#include <string>
#include <chrono>

#include <cuda.h>

void compute(unsigned int numNodes, unsigned int *nodePtrs,
             unsigned int *nodeNeighbors, unsigned int *nodeVisited,
             unsigned int *nodeVisited_ref, unsigned int *currLevelNodes,
             unsigned int *nextLevelNodes_ref, unsigned int *numCurrLevelNodes,
             unsigned int *numNextLevelNodes_ref) {

  // Compute reference out
  // Loop over all nodes in the curent level
  for (unsigned int idx = 0; idx < *numCurrLevelNodes; ++idx) {
    unsigned int node = currLevelNodes[idx];
    // Loop over all neighbors of the node
    for (unsigned int nbrIdx = nodePtrs[node]; nbrIdx < nodePtrs[node + 1];
         ++nbrIdx) {
      unsigned int neighbor = nodeNeighbors[nbrIdx];
      // If the neighbor hasn't been visited yet
      if (!nodeVisited_ref[neighbor]) {
        // Mark it and add it to the queue
        nodeVisited_ref[neighbor] = 1;
        nextLevelNodes_ref[*numNextLevelNodes_ref] = neighbor;
        ++*numNextLevelNodes_ref;
      }
    }
  }
}

static bool verify(unsigned int *numNextLevelNodes_ref,
                   unsigned int *NextLevelNodes_ref,
                   unsigned int *numNextLevelNodes_solution,
                   unsigned int *NextLevelNodes_solution) {

  INFO("Frontier size did not match");
  REQUIRE(*numNextLevelNodes_ref == *numNextLevelNodes_solution);

  std::sort(NextLevelNodes_solution,
            NextLevelNodes_solution + (*numNextLevelNodes_solution));
  std::sort(NextLevelNodes_ref, NextLevelNodes_ref + (*numNextLevelNodes_ref));

  for (int i = 0; i < *numNextLevelNodes_ref; i++) {

    INFO("Sorted frontier did not match solution");
    REQUIRE(NextLevelNodes_ref[i] == NextLevelNodes_solution[i]);
  }
  return true;
}

static std::chrono::time_point<std::chrono::high_resolution_clock> now() {
  return std::chrono::high_resolution_clock::now();
}

#define CUDA_RUNTIME(stmt) checkCuda(stmt, __FILE__, __LINE__);
void checkCuda(cudaError_t result, const char *file, const int line) {
  if (result != cudaSuccess) {
    LOG(critical,
        std::string(fmt::format("{}@{}: CUDA Runtime Error: {}\n", file, line,
                                cudaGetErrorString(result))));
    exit(-1);
  }
}

void setupProblem(unsigned int numNodes, unsigned int maxNeighborsPerNode,
                  unsigned int **nodePtrs_h, unsigned int **nodeNeighbors_h,
                  unsigned int **totalNeighbors_h, unsigned int **nodeVisited_h,
                  unsigned int **nodeVisited_ref,
                  unsigned int **currLevelNodes_h,
                  unsigned int **nextLevelNodes_h,
                  unsigned int **nextLevelNodes_ref,
                  unsigned int **numCurrLevelNodes_h,
                  unsigned int **numNextLevelNodes_h,
                  unsigned int **numNextLevelNodes_ref) {

  // Initialize node pointers
  *nodePtrs_h = (unsigned int *)malloc((numNodes + 1) * sizeof(unsigned int));
  *nodeVisited_h = (unsigned int *)malloc(numNodes * sizeof(unsigned int));
  *nodeVisited_ref = (unsigned int *)malloc(numNodes * sizeof(unsigned int));
  (*nodePtrs_h)[0] = 0;
  for (unsigned int node = 0; node < numNodes; ++node) {
    const unsigned int numNeighbors = rand() % (maxNeighborsPerNode + 1);
    (*nodePtrs_h)[node + 1] = (*nodePtrs_h)[node] + numNeighbors;
    (*nodeVisited_h)[node] = (*nodeVisited_ref)[node] = 0;
  }

  // Initialize neighbors
  *totalNeighbors_h = (unsigned int *)malloc(sizeof(unsigned int));
  **totalNeighbors_h = (*nodePtrs_h)[numNodes];
  *nodeNeighbors_h =
      (unsigned int *)malloc((**totalNeighbors_h) * sizeof(unsigned int));
  for (unsigned int neighborIdx = 0; neighborIdx < **totalNeighbors_h;
       ++neighborIdx) {
    (*nodeNeighbors_h)[neighborIdx] = rand() % numNodes;
  }

  // Initialize current level
  *numCurrLevelNodes_h = (unsigned int *)malloc(sizeof(unsigned int));
  **numCurrLevelNodes_h = numNodes / 10; // Let level contain 10% of all nodes
  *currLevelNodes_h =
      (unsigned int *)malloc((**numCurrLevelNodes_h) * sizeof(unsigned int));
  for (unsigned int idx = 0; idx < **numCurrLevelNodes_h; ++idx) {
    // Find a node that's not visited yet
    unsigned node;
    do {
      node = rand() % numNodes;
    } while ((*nodeVisited_h)[node]);
    (*currLevelNodes_h)[idx] = node;
    (*nodeVisited_h)[node] = (*nodeVisited_ref)[node] = 1;
  }

  // Prepare next level containers (i.e. output variables)
  *numNextLevelNodes_h = (unsigned int *)malloc(sizeof(unsigned int));
  **numNextLevelNodes_h = 0;
  *numNextLevelNodes_ref = (unsigned int *)malloc(sizeof(unsigned int));
  **numNextLevelNodes_ref = 0;
  *nextLevelNodes_h = (unsigned int *)malloc((numNodes) * sizeof(unsigned int));
  *nextLevelNodes_ref =
      (unsigned int *)malloc((numNodes) * sizeof(unsigned int));
}
