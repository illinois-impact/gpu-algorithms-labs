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

static bool verify(const std::vector<float> &ref_sol, const std::vector<float> &sol) {
    for (size_t i = 0; i < ref_sol.size(); i++) {
        INFO("Merge results differ from reference solution.");
        REQUIRE(ref_sol[i] == sol[i]);
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


void generate_input(std::vector<float> &A_h, const std::pair<float, float> &A_range, std::vector<float> &B_h, const std::pair<float, float> &B_range) {
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<float> dis_A(A_range.first, A_range.second);
    std::uniform_real_distribution<float> dis_B(B_range.first, B_range.second);

    std::generate(A_h.begin(), A_h.end(), std::bind(dis_A, std::ref(gen)));
    std::generate(B_h.begin(), B_h.end(), std::bind(dis_B, std::ref(gen)));

    std::sort(A_h.begin(), A_h.end());
    std::sort(B_h.begin(), B_h.end());
}

static void print_vector(const std::vector<float> vec) {
    for (const auto &elem : vec) {
        std::cout << elem << ' ';
    }
    std::cout << "" << '\n';
}
