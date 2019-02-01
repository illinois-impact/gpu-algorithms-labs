/*! \brief File containing evaluation and grading code

This file contains evaluation and grading code.
This code is in a separate file so that a known-good version can be used for
automatic grading and students should not need to modify it.
*/

#include "helper.hpp"

#include "template.hu"


namespace gpu_algorithms_labs_evaluation {

static void generate_data(float *x, const size_t n) {
    const auto rng_state = rng_new_state();
  
    for (size_t ii = 0 ; ii < n; ++ii) {
      x[ii] = rng_float(rng_state);
    }
  
    delete rng_state;
  }

  void verify(const float *A, const float *B, const float *C, size_t m, size_t k,
    size_t n) {
  
    for(size_t row = 0; row < m; ++row) {
      for(size_t col = 0; col < n; ++col) {
        float sum = 0;
        for(size_t i = 0; i < k; ++i) {
          sum += A[row + i*m]*B[i*n + col];
        }
        float relativeError = (sum - C[row + col*m])/sum;
        
        INFO("the results were not close enough at C[" << row << "," << col << "], expected " << sum << " got " << C[row + col*m] );
        REQUIRE(std::abs(relativeError) < 1e-6);
      }
    }
  
  }



static int eval(const size_t matArow, const size_t matAcol, const size_t matBcol) {

    const size_t matBrow = matAcol;
  
    // Generate model
    const auto conf_info = std::string("sgemm[<") + std::to_string(matArow) + "," + 
                                                    std::to_string(matAcol) + ">x<" + 
                                                    std::to_string(matBrow) + "," + 
                                                    std::to_string(matBcol) + ">]";
    INFO("Running "  << conf_info);
  
    
    const size_t aSz = matArow * matAcol;
    const size_t bSz = matBrow * matBcol;
    const size_t cSz = matArow * matBcol;
  
    // generate input data
    timer_start("Generating test data");
    std::vector<float> hostA(aSz);
    std::vector<float> hostB(bSz);
    std::vector<float> hostC(cSz);
    generate_data(hostA.data(), hostA.size());
    generate_data(hostB.data(), hostB.size());
    timer_stop();
  
    timer_start("Allocating GPU memory.");
    float *deviceA = nullptr, *deviceB = nullptr, *deviceC = nullptr;
    CUDA_RUNTIME(cudaMalloc((void **)&deviceA, aSz * sizeof(float)));
    CUDA_RUNTIME(cudaMalloc((void **)&deviceB, bSz * sizeof(float)));
    CUDA_RUNTIME(cudaMalloc((void **)&deviceC, cSz * sizeof(float)));
    timer_stop();
  
    timer_start("Copying inputs to the GPU.");
    CUDA_RUNTIME(cudaMemcpy(deviceA, hostA.data(), aSz * sizeof(float), cudaMemcpyDefault));
    CUDA_RUNTIME(cudaMemcpy(deviceB, hostB.data(), bSz * sizeof(float), cudaMemcpyDefault));
    CUDA_RUNTIME(cudaDeviceSynchronize());
    timer_stop();
  
    //////////////////////////////////////////
    // GPU Stencil Computation
    //////////////////////////////////////////
    timer_start("Performing GPU sgemm");
    basicSgemm('N', 'T', matArow, matBcol, matBrow, 1.0f, deviceA, matArow, deviceB, matBrow, 0.0f, deviceC, matBrow);
    CUDA_RUNTIME(cudaDeviceSynchronize());
    timer_stop();
  
    timer_start("Copying output to the CPU");
    CUDA_RUNTIME(cudaMemcpy(hostC.data(), deviceC, cSz * sizeof(float), cudaMemcpyDefault));
    CUDA_RUNTIME(cudaDeviceSynchronize());
    timer_stop();
  
    // verify with provided implementation
    timer_start("Verifying results");
    verify(hostA.data(), hostB.data(), hostC.data(), matArow, matAcol, matBcol);
    timer_stop();
  
    CUDA_RUNTIME(cudaFree(deviceA));
    CUDA_RUNTIME(cudaFree(deviceB));
    CUDA_RUNTIME(cudaFree(deviceC));
  
    return 0;
}
  
  
  TEST_CASE("sgemm", "[sgemm]") {
    SECTION("[dims:32,32,32]") {
      eval(32,32,32);
    }
    SECTION("[dims:30,30,30]") {
      eval(30,30,30);
    }
    SECTION("[dims:29,29,29]") {
      eval(29,29,29);
    }
    SECTION("[dims:31,31,31]") {
      eval(31,31,31);
    }
    SECTION("[dims:128,128,13]") {
      eval(128,128,13);
    }
    SECTION("[dims:13,128,128]") {
      eval(13,128,128);
    }
    SECTION("[dims:128,13,128]") {
      eval(128,13,128);
    }
    SECTION("[dims:1,1,1]") {
      eval(1,1,1);
    }
    SECTION("[dims:512,512,64]") {
      eval(512,512,64);
    }
    SECTION("[dims:256,256,256]") {
        eval(256,256,256);
    }
  }
} // namespace gpu_algorithms_labs_evaluation