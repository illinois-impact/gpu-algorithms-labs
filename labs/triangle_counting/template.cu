#include <cstdio>
#include <cstdlib>
#include <stdio.h>

#include "solution.hu"


uint64_t count_triangles(const pangolin::COOView<uint32_t> view, const int mode) {
  if (mode == 1) {

    // REQUIRED
    

    //@@ create a pangolin::Vector to hold per-edge triangle counts

    //@@ launch the linear search kernel here

    //@@ do a global reduction to produce the final triangle count
    uint64_t total = 0;
    return total;

  } else if (2 == mode) {

    // OPTIONAL
    
    uint64_t total = 0;
    return total;
  }
  else {
    assert("Unexpected mode");
    return uint64_t(-1);
  }
}