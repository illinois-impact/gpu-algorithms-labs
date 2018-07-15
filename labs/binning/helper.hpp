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

/// For simplicity, fix #bins=1024 so scan can use a single block and no
/// padding
static const int NUM_BINS = 1024;

enum class Mode : int { CPUNormal = 1, GPUNormal, GPUCutoff, GPUBinnedCPUPreprocessing, GPUBinnedGPUPreprocessing };

static std::string mode_name(Mode mode) {
  switch (mode) {
    case Mode::CPUNormal:
      return "CPUNormal";
    case Mode::GPUNormal:
      return "GPUNormal";
    case Mode::GPUCutoff:
      return "GPUCutoff";
    case Mode::GPUBinnedCPUPreprocessing:
      return "GPUBinnedCPUPreprocessing";
    case Mode::GPUBinnedGPUPreprocessing:
      return "GPUBinnedGPUPreprocessing";
    default:
      return "Unknown Mode";
  }
}
static std::vector<float> generate_input(int len, int max) {
  static std::random_device rd;  // Will be used to obtain a seed for the random number engine
  static std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  static std::uniform_real_distribution<> dis(1.0f, 2.0f);

  std::vector<float> res(len);
  std::generate(res.begin(), res.end(), [&]() {
    float r = 0;
    do {
      r = static_cast<float>(dis(gen)) * max;
    } while (r == 0);
    return r;
  });

  return res;
}

static std::vector<float> compute_output(const std::vector<float>& inputValues, const std::vector<float>& inputPositions, int numInputs,
                                         int gridSize) {
  std::vector<float> output(gridSize);
  std::fill_n(output.begin(), gridSize, 0);
  for (int inIdx = 0; inIdx < numInputs; ++inIdx) {
    const auto input = inputValues[inIdx];
    for (int outIdx = 0; outIdx < gridSize; ++outIdx) {
      const float dist = inputPositions[inIdx] - static_cast<float>(outIdx);
      if (dist == 0) {
        continue;
      }
      output[outIdx] += (input * input) / (dist * dist);
    }
  }
  return output;
}

static bool verify(const std::vector<float>& expected, const std::vector<float>& actual) {
  // INFO("Verifying the output");
  SECTION("the expected and actual sizes must match") {
    REQUIRE(expected.size() == actual.size());
  }

  SECTION("the results must match") {
    for (int ii = 0; ii < expected.size(); ii++) {
      INFO("the results did not match at index " << ii);
      REQUIRE(expected[ii] == Approx(actual[ii]).epsilon(0.1));
    }
  }
  return true;
}
