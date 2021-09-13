#include "helper.hpp"


/******************************************************************************
 GPU main computation kernels
*******************************************************************************/

//
// parallelization scheme
//
// For this lab, you are not writing (nor should you modify) the kernel
// launch code.  All three kernels are parallelized over outputs, with 
// one thread per grid point, so use CUDA variables to obtain each 
// thread's index.
// 
// Feel free to restructure things after you have passed all tests.
// Keep in mind, however, that coarsening in 1D will cost in terms of
// coalescing (which may matter less with newer GPUs with caches), but
// separating a thread's grid points reduces overlap in terms of which
// bins are needed for its grid points.  You may enjoy exploring the 
// performance space a bit.
//

//
// definition of inputs for kernels
// 
// grid_size -- number of grid points; coordinates are 0 to grid_size - 1
// num_in    -- number of input elements
// in_val    -- length num_in array of values of input elements
// in_pos    -- length num_in array of values of input elements
// out       -- length grid_size array of output values
// cutoff2   -- square of cutoff distance for later kernels
// in_val_sorted -- same as in_val, but with input elements sorted in 
//                  increasing order of position
// in_pos_sorted -- same as in_val, but with input elements sorted in 
//                  increasing order of position
// bin_pts       -- length (NUM_BINS + 1) array of indices into in_val_sorted
//                  and in_pos_sorted; element N defines the starting index
//                  for bin N, and element (N+1) defines the ending index + 1
//                  for bin N
//
// constants that you will need for the binned kernel
// NUM_BINS  -- number of bins; these split the range [0,grid_size) 
//              into NUM_BINS equally-sized bins; all input elements fall
//              into the specified range, and thus map into some bin
//

__global__ void gpu_normal_kernel(float *in_val, float *in_pos, float *out,
                                  int grid_size, int num_in) {
  //@@ INSERT CODE HERE
}

__global__ void gpu_cutoff_kernel(float *in_val, float *in_pos, float *out,
                                  int grid_size, int num_in,
                                  float cutoff2) {
  //@@ INSERT CODE HERE
}

__global__ void gpu_cutoff_binned_kernel(int *bin_ptrs,
                                         float *in_val_sorted,
                                         float *in_pos_sorted, float *out,
                                         int grid_size, float cutoff2) {
  //@@ INSERT CODE HERE

}

/******************************************************************************
 Main computation functions
*******************************************************************************/

void cpu_normal(float *in_val, float *in_pos, float *out, int grid_size,
                int num_in) {

  for (int inIdx = 0; inIdx < num_in; ++inIdx) {
    const float in_val2 = in_val[inIdx] * in_val[inIdx];
    for (int outIdx = 0; outIdx < grid_size; ++outIdx) {
      const float dist = in_pos[inIdx] - (float)outIdx;
      out[outIdx] += in_val2 / (dist * dist);
    }
  }
}

void gpu_normal(float *in_val, float *in_pos, float *out, int grid_size,
                int num_in) {

  const int numThreadsPerBlock = 512;
  const int numBlocks = (grid_size - 1) / numThreadsPerBlock + 1;
  gpu_normal_kernel<<<numBlocks, numThreadsPerBlock>>>(in_val, in_pos, out,
                                                       grid_size, num_in);
}

void gpu_cutoff(float *in_val, float *in_pos, float *out, int grid_size,
                int num_in, float cutoff2) {

  const int numThreadsPerBlock = 512;
  const int numBlocks = (grid_size - 1) / numThreadsPerBlock + 1;
  gpu_cutoff_kernel<<<numBlocks, numThreadsPerBlock>>>(
      in_val, in_pos, out, grid_size, num_in, cutoff2);
}

void gpu_cutoff_binned(int *bin_ptrs, float *in_val_sorted,
                       float *in_pos_sorted, float *out, int grid_size,
                       float cutoff2) {

  const int numThreadsPerBlock = 512;
  const int numBlocks = (grid_size - 1) / numThreadsPerBlock + 1;
  gpu_cutoff_binned_kernel<<<numBlocks, numThreadsPerBlock>>>(
      bin_ptrs, in_val_sorted, in_pos_sorted, out, grid_size, cutoff2);
}

/******************************************************************************
 Preprocessing kernels
*******************************************************************************/

//
// parallelization scheme
//
// Again, for the preprocessing kernels, you are not writing (nor should 
// you modify) the kernel launch code.  
//
// histogram and sort use a fixed number of thread blocks of fixed size.  
// Use these threads to iterate through the entire data set and compute
// the number of input elements in each bin for histogram, and to sort 
// the input elements in sort.
//
// scan uses one block of NUM_BINS / 2 threads; implement an exclusive
// scan using Brent-Kung, as described in 
// http://lumetta.web.engr.illinois.edu/408-S20/slide-copies/ece408-lecture16-S20.pdf
// and be sure to fill in the last element of bin_pts (element NUM_BINS)
// with the total sum of counts (should equal num_in, which may help in
// debugging)..
//

//
// definition of inputs for preprocessing kernels
// 
// grid_size -- size of coordinate space: [0, grid_size - 1) -- all input
//              elements are within this interval
// num_in    -- number of input elements
// in_val    -- length num_in array of values of input elements
// in_pos    -- length num_in array of values of input elements
// bin_counts -- length NUM_BINS array of counts of input elements in bins
//               (initialized to 0s, computed by histogram kernel, 
//               then provided to sort)
// bin_ptrs   -- length (NUM_BINS + 1) array of indices into sorted
//               input elements (produced by scan, provided to sort)
// in_val_sorted -- produced by sort by sorting from in_val
// in_pos_sorted -- produced by sort by sorting from in_pos

// constants that you will need for preprocessing
// NUM_BINS  -- number of bins; these split the range [0,grid_size) 
//              into NUM_BINS equally-sized bins; all input elements fall
//              into the specified range, and thus map into some bin
// 
// N.B.  If you change the number of bins, some code may break.  In 
// particular, your scan algorithm will need to be much more flexible
// to handle a larger data set--simply launching another thread block
// is not sufficient.

__global__ void histogram(float *in_pos, int *bin_counts, int num_in,
                          int grid_size) {

  //@@ INSERT CODE HERE
}

__global__ void scan(int *bin_counts, int *bin_ptrs) {

  //@@ INSERT CODE HERE
}

__global__ void sort(float *in_val, float *in_pos, float *in_val_sorted,
                     float *in_pos_sorted, int grid_size, int num_in,
                     int *bin_counts, int *bin_ptrs) {

  //@@ INSERT CODE HERE
}

/******************************************************************************
 Preprocessing functions
*******************************************************************************/

static void cpu_preprocess(float *in_val, float *in_pos,
                           float *in_val_sorted, float *in_pos_sorted,
                           int grid_size, int num_in, int *bin_counts,
                           int *bin_ptrs) {

  // Histogram the input positions
  for (int binIdx = 0; binIdx < NUM_BINS; ++binIdx) {
    bin_counts[binIdx] = 0;
  }
  for (int inIdx = 0; inIdx < num_in; ++inIdx) {
    const int binIdx = (int)((in_pos[inIdx] / grid_size) * NUM_BINS);
    ++bin_counts[binIdx];
  }

  // Scan the histogram to get the bin pointers
  bin_ptrs[0] = 0;
  for (int binIdx = 0; binIdx < NUM_BINS; ++binIdx) {
    bin_ptrs[binIdx + 1] = bin_ptrs[binIdx] + bin_counts[binIdx];
  }

  // Sort the inputs into the bins
  for (int inIdx = 0; inIdx < num_in; ++inIdx) {
    const int binIdx = (int)((in_pos[inIdx] / grid_size) * NUM_BINS);
    const int newIdx = bin_ptrs[binIdx + 1] - bin_counts[binIdx];
    --bin_counts[binIdx];
    in_val_sorted[newIdx] = in_val[inIdx];
    in_pos_sorted[newIdx] = in_pos[inIdx];
  }
}

static void gpu_preprocess(float *in_val, float *in_pos,
                           float *in_val_sorted, float *in_pos_sorted,
                           int grid_size, int num_in, int *bin_counts,
                           int *bin_ptrs) {

  const int numThreadsPerBlock = 512;

  // Histogram the input positions
  histogram<<<30, numThreadsPerBlock>>>(in_pos, bin_counts, num_in,
                                        grid_size);

  // Scan the histogram to get the bin pointers
  if (NUM_BINS != 1024) {
    FAIL("NUM_BINS must be 1024. Do not change.");
    return;
  }
  scan<<<1, numThreadsPerBlock>>>(bin_counts, bin_ptrs);

  // Sort the inputs into the bins
  sort<<<30, numThreadsPerBlock>>>(in_val, in_pos, in_val_sorted,
                                   in_pos_sorted, grid_size, num_in,
                                   bin_counts, bin_ptrs);
}


template <Mode mode>
int eval(const int num_in, const int max, const int grid_size) {
  const std::string mode_info = mode_name(mode);
  const std::string conf_info =
      std::string("[len:") + std::to_string(num_in) + "/max:" + std::to_string(max) + "/gridSize:" + std::to_string(grid_size) + "]";

  // Initialize host variables
  // ----------------------------------------------

  // Variables
  std::vector<float> in_val_h;
  std::vector<float> in_pos_h;
  float *in_val_d = nullptr;
  float *in_pos_d = nullptr;
  float *out_d    = nullptr;

  // Constants
  const float cutoff  = 3000.0f; // Cutoff distance for optimized computation
  const float cutoff2 = cutoff * cutoff;

  // Extras needed for input binning
  std::vector<int> bin_counts_h;
  std::vector<int> bin_ptrs_h;
  std::vector<float> in_val_sorted_h;
  std::vector<float> in_pos_sorted_h;
  int *bin_counts_d      = nullptr;
  int *bin_ptrs_d        = nullptr;
  float *in_val_sorted_d = nullptr;
  float *in_pos_sorted_d = nullptr;

  in_val_h = generate_input(num_in, max);
  in_pos_h = generate_input(num_in, grid_size);

  std::vector<float> out_h(grid_size);
  std::fill_n(out_h.begin(), grid_size, 0.0f);

  INFO("Running " << mode_info << conf_info);

  // CPU Preprocessing
  // ------------------------------------------------------

  if (mode == Mode::GPUBinnedCPUPreprocessing) {

    timer_start("Allocating data for preprocessing");
    // Data structures needed to preprocess the bins on the CPU
    bin_counts_h.reserve(NUM_BINS);
    bin_ptrs_h.reserve(NUM_BINS + 1);
    in_val_sorted_h.reserve(num_in);
    in_pos_sorted_h.reserve(num_in);

    cpu_preprocess(in_val_h.data(), in_pos_h.data(), in_val_sorted_h.data(), in_pos_sorted_h.data(), grid_size, num_in, bin_counts_h.data(),
                   bin_ptrs_h.data());
    timer_stop();
  }
  // Allocate device variables
  // ----------------------------------------------

  if (mode != Mode::CPUNormal) {

    timer_start("Allocating data");
    // If preprocessing on the CPU, GPU doesn't need the unsorted arrays
    if (mode != Mode::GPUBinnedCPUPreprocessing) {
      THROW_IF_ERROR(cudaMalloc((void **) &in_val_d, num_in * sizeof(float)));
      THROW_IF_ERROR(cudaMalloc((void **) &in_pos_d, num_in * sizeof(float)));
    }

    // All modes need the output array
    THROW_IF_ERROR(cudaMalloc((void **) &out_d, grid_size * sizeof(float)));

    // Only binning modes need binning information
    if (mode == Mode::GPUBinnedCPUPreprocessing || mode == Mode::GPUBinnedGPUPreprocessing) {
      THROW_IF_ERROR(cudaMalloc((void **) &in_val_sorted_d, num_in * sizeof(float)));
      THROW_IF_ERROR(cudaMalloc((void **) &in_pos_sorted_d, num_in * sizeof(float)));
      THROW_IF_ERROR(cudaMalloc((void **) &bin_ptrs_d, (NUM_BINS + 1) * sizeof(int)));

      if (mode == Mode::GPUBinnedGPUPreprocessing) {
        // Only used in preprocessing but not the actual computation
        THROW_IF_ERROR(cudaMalloc((void **) &bin_counts_d, NUM_BINS * sizeof(int)));
      }
    }

    cudaDeviceSynchronize();
    timer_stop();
  }

  // Copy host variables to device
  // ------------------------------------------

  if (mode != Mode::CPUNormal) {
    timer_start("Copying data");
    // If preprocessing on the CPU, GPU doesn't need the unsorted arrays
    if (mode != Mode::GPUBinnedCPUPreprocessing) {
      THROW_IF_ERROR(cudaMemcpy(in_val_d, in_val_h.data(), num_in * sizeof(float), cudaMemcpyHostToDevice));
      THROW_IF_ERROR(cudaMemcpy(in_pos_d, in_pos_h.data(), num_in * sizeof(float), cudaMemcpyHostToDevice));
    }

    // All modes need the output array
    THROW_IF_ERROR(cudaMemset(out_d, 0, grid_size * sizeof(float)));

    if (mode == Mode::GPUBinnedCPUPreprocessing) {
      THROW_IF_ERROR(cudaMemcpy(in_val_sorted_d, in_val_sorted_h.data(), num_in * sizeof(float), cudaMemcpyHostToDevice));
      THROW_IF_ERROR(cudaMemcpy(in_pos_sorted_d, in_pos_sorted_h.data(), num_in * sizeof(float), cudaMemcpyHostToDevice));
      THROW_IF_ERROR(cudaMemcpy(bin_ptrs_d, bin_ptrs_h.data(), (NUM_BINS + 1) * sizeof(int), cudaMemcpyHostToDevice));
    } else if (mode == Mode::GPUBinnedGPUPreprocessing) {
      // If preprocessing on the GPU, bin counts need to be initialized
      //  and nothing needs to be copied
      THROW_IF_ERROR(cudaMemset(bin_counts_d, 0, NUM_BINS * sizeof(int)));
    }

    THROW_IF_ERROR(cudaDeviceSynchronize());
    timer_stop();
  }

  // GPU Preprocessing
  // ------------------------------------------------------

  if (mode == Mode::GPUBinnedGPUPreprocessing) {

    timer_start("Preprocessing data on the GPU...");

    gpu_preprocess(in_val_d, in_pos_d, in_val_sorted_d, in_pos_sorted_d, grid_size, num_in, bin_counts_d, bin_ptrs_d);
    THROW_IF_ERROR(cudaDeviceSynchronize());
    timer_stop();
  }

  // Launch kernel
  // ----------------------------------------------------------

  timer_start(std::string("Performing ") + mode_info + conf_info + std::string(" computation"));
  switch (mode) {
    case Mode::CPUNormal:
      cpu_normal(in_val_h.data(), in_pos_h.data(), out_h.data(), grid_size, num_in);
      break;
    case Mode::GPUNormal:
      gpu_normal(in_val_d, in_pos_d, out_d, grid_size, num_in);
      break;
    case Mode::GPUCutoff:
      gpu_cutoff(in_val_d, in_pos_d, out_d, grid_size, num_in, cutoff2);
      break;
    case Mode::GPUBinnedCPUPreprocessing:
    case Mode::GPUBinnedGPUPreprocessing:
      gpu_cutoff_binned(bin_ptrs_d, in_val_sorted_d, in_pos_sorted_d, out_d, grid_size, cutoff2);
      break;
    default:
      FAIL("Invalid mode " << (int) mode);
  }
  THROW_IF_ERROR(cudaDeviceSynchronize());
  timer_stop();

  // Copy device variables from host
  // ----------------------------------------

  if (mode != Mode::CPUNormal) {
    THROW_IF_ERROR(cudaMemcpy(out_h.data(), out_d, grid_size * sizeof(float), cudaMemcpyDeviceToHost));
    THROW_IF_ERROR(cudaDeviceSynchronize());
  }

  // Verify correctness
  // -----------------------------------------------------

  const auto actual_output = compute_output(in_val_h, in_pos_h, num_in, grid_size);
  verify(actual_output, out_h);

  // Free memory
  // ------------------------------------------------------------

  if (mode != Mode::CPUNormal) {
    if (mode != Mode::GPUBinnedCPUPreprocessing) {
      cudaFree(in_val_d);
      cudaFree(in_pos_d);
    }
    cudaFree(out_d);
    if (mode == Mode::GPUBinnedCPUPreprocessing || mode == Mode::GPUBinnedGPUPreprocessing) {
      cudaFree(in_val_sorted_d);
      cudaFree(in_pos_sorted_d);
      cudaFree(bin_ptrs_d);
      if (mode == Mode::GPUBinnedGPUPreprocessing) {
        cudaFree(bin_counts_d);
      }
    }
  }

  std::cout << "----------------------------------------\n";
  return 0;
}

TEST_CASE("CPUNormal", "[cpu_normal]") {
  SECTION("[len:60/max:1/gridSize:60]") {
    eval<Mode::CPUNormal>(60, 1, 60);
  }
  SECTION("[len:600/max:1/gridSize:100]") {
    eval<Mode::CPUNormal>(600, 1, 100);
  }
  SECTION("[len:603/max:1/gridSize:201]") {
    eval<Mode::CPUNormal>(603, 1, 201);
  }
  SECTION("[len:409/max:1/gridSize:160]") {
    eval<Mode::CPUNormal>(409, 1, 160);
  }
  SECTION("[len:419/max:1/gridSize:100]") {
    eval<Mode::CPUNormal>(419, 1, 100);
  }
  SECTION("[len:8065/max:1/gridSize:201]") {
    eval<Mode::CPUNormal>(8065, 1, 201);
  }
  SECTION("[len:1440/max:1/gridSize:443]") {
    eval<Mode::CPUNormal>(1440, 1, 443);
  }
  SECTION("[len:400/max:1/gridSize:200]") {
    eval<Mode::CPUNormal>(400, 1, 200);
  }
  SECTION("[len:696/max:1/gridSize:232]") {
    eval<Mode::CPUNormal>(696, 1, 232);
  }
}

TEST_CASE("GPUNormal", "[gpu_normal]") {
  SECTION("[len:60/max:1/gridSize:60]") {
    eval<Mode::GPUNormal>(60, 1, 60);
  }
  SECTION("[len:600/max:1/gridSize:100]") {
    eval<Mode::GPUNormal>(600, 1, 100);
  }
  SECTION("[len:603/max:1/gridSize:201]") {
    eval<Mode::GPUNormal>(603, 1, 201);
  }
  SECTION("[len:409/max:1/gridSize:160]") {
    eval<Mode::GPUNormal>(409, 1, 160);
  }
  SECTION("[len:419/max:1/gridSize:100]") {
    eval<Mode::GPUNormal>(419, 1, 100);
  }
  SECTION("[len:8065/max:1/gridSize:201]") {
    eval<Mode::GPUNormal>(8065, 1, 201);
  }
  SECTION("[len:1440/max:1/gridSize:443]") {
    eval<Mode::GPUNormal>(1440, 1, 443);
  }
  SECTION("[len:400/max:1/gridSize:200]") {
    eval<Mode::GPUNormal>(400, 1, 200);
  }
  SECTION("[len:696/max:1/gridSize:232]") {
    eval<Mode::GPUNormal>(696, 1, 232);
  }
}

TEST_CASE("GPUCutoff", "[gpu_cutoff]") {
  SECTION("[len:60/max:1/gridSize:60]") {
    eval<Mode::GPUCutoff>(60, 1, 60);
  }
  SECTION("[len:600/max:1/gridSize:100]") {
    eval<Mode::GPUCutoff>(600, 1, 100);
  }
  SECTION("[len:603/max:1/gridSize:201]") {
    eval<Mode::GPUCutoff>(603, 1, 201);
  }
  SECTION("[len:409/max:1/gridSize:160]") {
    eval<Mode::GPUCutoff>(409, 1, 160);
  }
  SECTION("[len:419/max:1/gridSize:100]") {
    eval<Mode::GPUCutoff>(419, 1, 100);
  }
  SECTION("[len:8065/max:1/gridSize:201]") {
    eval<Mode::GPUCutoff>(8065, 1, 201);
  }
  SECTION("[len:1440/max:1/gridSize:443]") {
    eval<Mode::GPUCutoff>(1440, 1, 443);
  }
  SECTION("[len:400/max:1/gridSize:200]") {
    eval<Mode::GPUCutoff>(400, 1, 200);
  }
  SECTION("[len:696/max:1/gridSize:232]") {
    eval<Mode::GPUCutoff>(696, 1, 232);
  }
}

TEST_CASE("GPUBinnedCPUPreprocessing", "[gpu_binned_cpu_preprocessing]") {
  SECTION("[len:60/max:1/gridSize:60]") {
    eval<Mode::GPUBinnedCPUPreprocessing>(60, 1, 60);
  }
  SECTION("[len:600/max:1/gridSize:100]") {
    eval<Mode::GPUBinnedCPUPreprocessing>(600, 1, 100);
  }
  SECTION("[len:603/max:1/gridSize:201]") {
    eval<Mode::GPUBinnedCPUPreprocessing>(603, 1, 201);
  }
  SECTION("[len:409/max:1/gridSize:160]") {
    eval<Mode::GPUBinnedCPUPreprocessing>(409, 1, 160);
  }
  SECTION("[len:419/max:1/gridSize:100]") {
    eval<Mode::GPUBinnedCPUPreprocessing>(419, 1, 100);
  }
  SECTION("[len:8065/max:1/gridSize:201]") {
    eval<Mode::GPUBinnedCPUPreprocessing>(8065, 1, 201);
  }
  SECTION("[len:1440/max:1/gridSize:443]") {
    eval<Mode::GPUBinnedCPUPreprocessing>(1440, 1, 443);
  }
  SECTION("[len:400/max:1/gridSize:200]") {
    eval<Mode::GPUBinnedCPUPreprocessing>(400, 1, 200);
  }
  SECTION("[len:696/max:1/gridSize:232]") {
    eval<Mode::GPUBinnedCPUPreprocessing>(696, 1, 232);
  }
}

TEST_CASE("GPUBinnedGPUPreprocessing", "[gpu_binned_gpu_preprocessing]") {
  SECTION("[len:60/max:1/gridSize:60]") {
    eval<Mode::GPUBinnedGPUPreprocessing>(60, 1, 60);
  }
  SECTION("[len:600/max:1/gridSize:100]") {
    eval<Mode::GPUBinnedGPUPreprocessing>(600, 1, 100);
  }
  SECTION("[len:603/max:1/gridSize:201]") {
    eval<Mode::GPUBinnedGPUPreprocessing>(603, 1, 201);
  }
  SECTION("[len:409/max:1/gridSize:160]") {
    eval<Mode::GPUBinnedGPUPreprocessing>(409, 1, 160);
  }
  SECTION("[len:419/max:1/gridSize:100]") {
    eval<Mode::GPUBinnedGPUPreprocessing>(419, 1, 100);
  }
  SECTION("[len:8065/max:1/gridSize:201]") {
    eval<Mode::GPUBinnedGPUPreprocessing>(8065, 1, 201);
  }
  SECTION("[len:1440/max:1/gridSize:443]") {
    eval<Mode::GPUBinnedGPUPreprocessing>(1440, 1, 443);
  }
  SECTION("[len:400/max:1/gridSize:200]") {
    eval<Mode::GPUBinnedGPUPreprocessing>(400, 1, 200);
  }
  SECTION("[len:696/max:1/gridSize:232]") {
    eval<Mode::GPUBinnedGPUPreprocessing>(696, 1, 232);
  }
}
