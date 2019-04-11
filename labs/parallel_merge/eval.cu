/*! \brief File containing evaluation and grading code

This file contains evaluation and grading code.
This code is in a separate file so that a known-good version can be used for
automatic grading and students should not need to modify it.
*/

#include "helper.hpp"
#include "template.hu"

namespace gpu_algorithms_labs_evaluation {

enum Mode { CPU = 1, GPU_MERGE, GPU_TILED_MERGE, GPU_TILED_MERGE_CIRCULAR};

void cpu_merge(const std::vector<float> &A, const std::vector<float> &B, std::vector<float> &C) {
    int i = 0, j = 0, k = 0;
    while ((i < A.size()) && (j < B.size())) {
        if (A[i] <= B[j]) {
            C[k++] = A[i++];
        } else {
            C[k++] = B[j++];
        }
    }

    if (i == A.size()) {
        std::copy(B.begin()+j, B.end(), C.begin()+k);
    } else {
        std::copy(A.begin()+i, A.end(), C.begin()+k);
    }
}

void merge_sequential_host(float* A, int A_len, float* B, int B_len, float* C) {
    int i = 0, j = 0, k = 0;

    while ((i < A_len) && (j < B_len)) {
        C[k++] = A[i] <= B[j] ? A[i++] : B[j++];
    }

    if (i == A_len) {
        while (j < B_len) {
            C[k++] = B[j++];
        }
    } else {
        while (i < A_len) {
            C[k++] = A[i++];
        }
    }
}

static int eval(const int A_len, std::pair<float, float> A_range,
                const int B_len, std::pair<float, float> B_range,
                Mode mode) {
    // Initialize variables
    // ----------------------------------------------
    const int A_byteCnt = A_len * sizeof(float);
    const int B_byteCnt = B_len * sizeof(float);
    const int output_len = A_len + B_len;
    const int output_byteCnt = A_byteCnt + B_byteCnt;

    std::vector<float> A_h(A_len);
    std::vector<float> B_h(B_len);
    std::vector<float> output(output_len);
    std::vector<float> solution(output_len);

    //GPU
    float* A_d; // Input array
    float* B_d; // Input array
    float* output_d; // Output array

    ///////////////////////////////////////////////////////

    timer_start("Generating test data and reference solution");
    generate_input(A_h, A_range, B_h, B_range);
    std::merge(A_h.begin(), A_h.end(), B_h.begin(), B_h.end(), solution.begin());
    timer_stop(); // Generating test data

    timer_start("Allocating GPU memory");
    if (mode != CPU) {
        CUDA_RUNTIME(cudaMalloc((void**)&A_d, A_byteCnt));
        CUDA_RUNTIME(cudaMalloc((void**)&B_d, B_byteCnt));
        CUDA_RUNTIME(cudaMalloc((void**)&output_d, output_byteCnt));
    }
    timer_stop();

    timer_start("Copying inputs to the GPU");
    if (mode != CPU) {
        CUDA_RUNTIME(cudaMemcpy(A_d, A_h.data(), A_byteCnt, cudaMemcpyHostToDevice));
        CUDA_RUNTIME(cudaMemcpy(B_d, B_h.data(), B_byteCnt, cudaMemcpyHostToDevice));
    }
    timer_stop();

    timer_start("Performing parallel merge");
    if (mode == CPU) {
        // cpu_merge(A_h, B_h, output);
        merge_sequential_host(A_h.data(), A_h.size(), B_h.data(), B_h.size(), output.data());
    } else if (mode == GPU_MERGE) {
        gpu_basic_merge(A_d, A_len, B_d, B_len, output_d);
    } else if (mode == GPU_TILED_MERGE) {
        gpu_tiled_merge(A_d, A_len, B_d, B_len, output_d);
    } else if (mode == GPU_TILED_MERGE_CIRCULAR) {
        gpu_circular_buffer_merge(A_d, A_len, B_d, B_len, output_d);
    }
    timer_stop();

    timer_start("Copying output to host");
    if (mode != CPU) {
        CUDA_RUNTIME(cudaMemcpy(output.data(), output_d, output_byteCnt, cudaMemcpyDeviceToHost));
    }
    timer_stop();

    timer_start("Verifying results");
    verify(solution, output);
    timer_stop();

    if (mode != CPU) {
        CUDA_RUNTIME(cudaFree(A_d));
        CUDA_RUNTIME(cudaFree(B_d));
        CUDA_RUNTIME(cudaFree(output_d));
    }

    return 0;
}

std::pair<float, float> range0(0.0, 100.0);
std::pair<float, float> range1(50.0, 150.0);
std::pair<float, float> range2(100.0, 200.0);
std::pair<float, float> range3(0.0, 1000000.0);

TEST_CASE("required", "[GPU_MERGE]") {
    SECTION("1023, range0, 1023, range0, GPU_MERGE") { eval(1023, range0, 1023, range0, GPU_MERGE); }
    SECTION("4095, range0, 5000, range1, GPU_MERGE") { eval(4095, range0, 5000, range1, GPU_MERGE); }
    SECTION("1023, range0, 1025, range2, GPU_MERGE") { eval(1023, range0, 1025, range2, GPU_MERGE); }
    SECTION("20470, range0, 2047, range3, GPU_MERGE") { eval(20470, range0, 2047, range3, GPU_MERGE); }
    SECTION("1234567, range3, 1357911, range3, GPU_MERGE") { eval(1234567, range3, 1357911, range3, GPU_MERGE); }
    SECTION("1357911, range3, 1234567, range3, GPU_MERGE") { eval(1357911, range3, 1234567, range3, GPU_MERGE); }

    SECTION("1023, range0, 1023, range0, GPU_TILED_MERGE") { eval(1023, range0, 1023, range0, GPU_TILED_MERGE); }
    SECTION("4095, range0, 5000, range1, GPU_TILED_MERGE") { eval(4095, range0, 5000, range1, GPU_TILED_MERGE); }
    SECTION("1023, range0, 1025, range2, GPU_TILED_MERGE") { eval(1023, range0, 1025, range2, GPU_TILED_MERGE); }
    SECTION("20470, range0, 2047, range3, GPU_TILED_MERGE") { eval(20470, range0, 2047, range3, GPU_TILED_MERGE); }
    SECTION("1234567, range3, 1357911, range3, GPU_TILED_MERGE") { eval(1234567, range3, 1357911, range3, GPU_TILED_MERGE); }
    SECTION("1357911, range3, 1234567, range3, GPU_TILED_MERGE") { eval(1357911, range3, 1234567, range3, GPU_TILED_MERGE); }
}

TEST_CASE("extra", "[GPU_TILED_MERGE_CIRCULAR]") {
    SECTION("1023, range0, 1023, range0, GPU_TILED_MERGE_CIRCULAR") { eval(1023, range0, 1023, range0, GPU_TILED_MERGE_CIRCULAR); }
    SECTION("4095, range0, 5000, range1, GPU_TILED_MERGE_CIRCULAR") { eval(4095, range0, 5000, range1, GPU_TILED_MERGE_CIRCULAR); }
    SECTION("1023, range0, 1025, range2, GPU_TILED_MERGE_CIRCULAR") { eval(1023, range0, 1025, range2, GPU_TILED_MERGE_CIRCULAR); }
    SECTION("20470, range0, 2047, range3, GPU_TILED_MERGE_CIRCULAR") { eval(20470, range0, 2047, range3, GPU_TILED_MERGE_CIRCULAR); }
    SECTION("1234567, range3, 1357911, range3, GPU_TILED_MERGE_CIRCULAR") { eval(1234567, range3, 1357911, range3, GPU_TILED_MERGE_CIRCULAR); }
    SECTION("1357911, range3, 1234567, range3, GPU_TILED_MERGE_CIRCULAR") { eval(1357911, range3, 1234567, range3, GPU_TILED_MERGE_CIRCULAR); }
}



} // namespace gpu_algorithms_labs_evaluation
