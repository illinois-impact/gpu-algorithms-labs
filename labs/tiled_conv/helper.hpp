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

#include "range.hpp"
#include "shape.hpp"

/*********************************************************************/
/* Random number generator                                           */
/* https://en.wikipedia.org/wiki/Xorshift                            */
/* xorshift32                                                        */
/*********************************************************************/

static uint_fast32_t rng_uint32(uint_fast32_t *rng_state) {
  uint_fast32_t local = *rng_state;
  local ^= local << 13; // a
  local ^= local >> 17; // b
  local ^= local << 5;  // c
  *rng_state = local;
  return local;
}

static uint_fast32_t *rng_new_state(uint_fast32_t seed) {
  uint64_t *rng_state = new uint64_t;
  *rng_state          = seed;
  return rng_state;
}

static uint_fast32_t *rng_new_state() {
  return rng_new_state(88172645463325252LL);
}

static float rng_float(uint_fast32_t *state) {
  uint_fast32_t rnd = rng_uint32(state);
  const auto r      = static_cast<float>(rnd) / static_cast<float>(UINT_FAST32_MAX);
  if (std::isfinite(r)) {
    return r;
  }
  return rng_float(state);
}

static void generate_data(float *x, const shape &xdims) {
  const auto rng_state = rng_new_state();

  for (const auto ii : range(0, xdims.flattened_length())) {
    x[ii] = rng_float(rng_state);
  }

  delete rng_state;
}

// generate convolution filter
static void generate_convfilters(float *conv, const shape &convdim) {
  // Set convolution filter values to 1
  std::fill(conv, conv + convdim.flattened_length(), 1);
}

static bool verify(const float *expected, const float *actual, const shape& dims) {

  SECTION("the results must match") {
    for (size_t n = 0; n < dims.num; ++n) {
      for (size_t d = 0; d < dims.depth; ++d) {
        for (size_t h = 0; h < dims.height; ++h) {
          for (size_t w = 0; w < dims.width; ++w) {
            size_t ii = w + dims.width * (h + dims.height * (d + dims.depth * (n)));
            INFO("the results did not match at [" << n << "," << d << "," << h << "," << w << "] (index " << ii << ')');
            REQUIRE(expected[ii] == actual[ii]);
          }      
        }    
      }      
    }
  }
  return true;
}

static std::chrono::time_point<std::chrono::high_resolution_clock> now() {
  return std::chrono::high_resolution_clock::now();
}