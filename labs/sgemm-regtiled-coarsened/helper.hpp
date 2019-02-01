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

static int rng_int(uint_fast32_t *state) {
  uint_fast32_t rnd = rng_uint32(state);
  return static_cast<int>(rnd);
}

static bool verify(const int *Anext, const int *A0, const int nx, const int ny, const int nz) {

    #define _Anext(xi, yi, zi) Anext[(zi) * (ny * nx) + (yi) * nx + (xi)]
    #define _A0(xi, yi, zi)    A0[(zi) * (ny * nx) + (yi) * nx + (xi)]

  SECTION("the results must match") {

    for (size_t z = 1; z < nz-1; ++z) {
      for (size_t y = 1; y < ny-1; ++y) {
        for (size_t x = 1; x < nx-1; ++x) {
          const int expected = _A0(x,     y,     z + 1) +
                               _A0(x,     y,     z - 1) +
                               _A0(x,     y + 1, z    ) +
                               _A0(x,     y - 1, z    ) +
                               _A0(x + 1, y,     z    ) +
                               _A0(x - 1, y,     z    ) -
                           6 * _A0(x,     y,     z    );
          INFO("the results did not match at [" << x << "," << y << "," << z << "]");
          REQUIRE(expected == _Anext(x, y, z));
        }    
      }      
    }
  }
  return true;

  #undef _Anext
  #undef _A0
}

static std::chrono::time_point<std::chrono::high_resolution_clock> now() {
  return std::chrono::high_resolution_clock::now();
}

#define CUDA_RUNTIME(stmt) checkCuda(stmt, __FILE__, __LINE__);
void checkCuda(cudaError_t result, const char *file, const int line)
{
	if (result != cudaSuccess)
	{
		LOG(critical, std::string(fmt::format("{}@{}: CUDA Runtime Error: {}\n", file, line,
			cudaGetErrorString(result))));
		exit(-1);
	}
}