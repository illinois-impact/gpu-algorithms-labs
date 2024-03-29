#include <cstdio>
#include <cstdlib>
#include <stdio.h>

#include "template.hu"

__global__ static void kernel_tc(uint64_t *__restrict__ triangleCounts, //!< per-edge triangle counts
                                 const uint32_t *const edgeSrc,         //!< node ids for edge srcs
                                 const uint32_t *const edgeDst,         //!< node ids for edge dsts
                                 const uint32_t *const rowPtr,          //!< source node offsets in edgeDst
                                 const size_t numEdges                  //!< how many edges to count triangles for
) {

  // Determine the source and destination node for the edge

  // Use the row pointer array to determine the start and end of the neighbor list in the column index array

  // Determine how many elements of those two arrays are common
}

uint64_t count_triangles(const pangolin::COOView<uint32_t> view, const int mode) {
  //@@ create a pangolin::Vector (uint64_t) to hold per-edge triangle counts
  // Pangolin is backed by CUDA so you do not need to explicitly copy data between host and device.
  // You may find pangolin::Vector::data() function useful to get a pointer for your kernel to use.

  uint64_t total = 0;

  dim3 dimBlock(512);
  //@@ calculate the number of blocks needed
  // dim3 dimGrid (ceil(number of non-zeros / dimBlock.x))

  if (mode == 1) {

    //@@ launch the linear search kernel here
    // kernel_tc<<<dimGrid, dimBlock>>>(...)

  } else if (2 == mode) {

    //@@ launch the hybrid search kernel here
    // your_kernel_name_goes_here<<<dimGrid, dimBlock>>>(...)

  } else {
    assert("Unexpected mode");
    return uint64_t(-1);
  }

  //@@ do a global reduction (on CPU or GPU) to produce the final triangle count
  return total;
}
