#pragma once

#define CATCH_CONFIG_CPP11_TO_STRING
#define CATCH_CONFIG_COLOUR_ANSI
#define CATCH_CONFIG_MAIN

#include <fmt/format.h>

#include "common/catch.hpp"
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

inline bool endsWith (const std::string &fullString, const std::string &ending) {
  if (fullString.length() >= ending.length()) {
      return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
  } else {
      return false;
  }
}
