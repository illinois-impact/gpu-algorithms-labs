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
#include <vector>

#include <cuda.h>

__host__ __device__ uint32_t outInvariant(uint32_t inValue) {
  return inValue * inValue;
}

__host__ __device__ uint32_t outDependent(uint32_t value, int inIdx, int outIdx) {
  if (inIdx == outIdx) {
    return 2 * value;
  }
  if (inIdx > outIdx) {
    return value / (inIdx - outIdx);
  }
  return value / (outIdx - inIdx);
}

static std::vector<uint32_t> generate_input(int len) {
  static std::random_device rd;  // Will be used to obtain a seed for the random number engine
  static std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  static std::uniform_int_distribution<> dis(1, 6);

  std::vector<uint32_t> res(len);
  std::generate(res.begin(), res.end(), [&]() {
    uint32_t r = 0;
    do {
      r = static_cast<uint32_t>(dis(gen));
    } while (r <= 0);
    return r;
  });

  return res;
}

static std::vector<uint32_t> compute_output(const std::vector<uint32_t>& inputValues, uint32_t num_out) {
  std::vector<uint32_t> output(num_out);
  std::fill_n(output.begin(), num_out, 0);
  for (int inIdx = 0; inIdx < inputValues.size(); ++inIdx) {
    uint32_t intermediate = outInvariant(inputValues[inIdx]);
    for (int outIdx = 0; outIdx < num_out; ++outIdx) {
      output[outIdx] += outDependent(intermediate, inIdx, outIdx);
    }
  }
  return output;
}

static bool verify(const std::vector<uint32_t>& expected, const std::vector<uint32_t>& actual) {
  // INFO("Verifying the output");
  SECTION("the expected and actual sizes must match") {
    REQUIRE(expected.size() == actual.size());
  }

  SECTION("the results must match") {
    for (int ii = 0; ii < expected.size(); ii++) {
      INFO("the results did not match at index " << ii);
      REQUIRE(expected[ii] == actual[ii]);
    }
  }
  return true;
}
