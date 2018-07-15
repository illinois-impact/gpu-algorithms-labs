
#include "common/fmt.hpp"
#include "common/utils.hpp"


#define PRINT(...) LOG(info, std::string(fmt::format(__VA_ARGS__)))

//@@ The purpose of this code is to become familiar with the submission
//@@ process. Do not worry if you do not understand all the details of
//@@ the code.

int main(int argc, char ** argv) {
    int deviceCount;

    cudaGetDeviceCount(&deviceCount);

    timer_start("Getting GPU Data."); //@@ start a timer

    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp deviceProp;

        cudaGetDeviceProperties(&deviceProp, dev);

        if (dev == 0) {
            if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
                PRINT("No CUDA GPU has been detected");
                return -1;
            } else if (deviceCount == 1) {
                //@@ WbLog is a provided logging API (similar to Log4J).
                //@@ The logging function wbLog takes a level which is either
                //@@ OFF, FATAL, ERROR, WARN, INFO, DEBUG, or trace and a
                //@@ message to be printed.
                PRINT("There is 1 device supporting CUDA");
            } else {
                PRINT("There are {} devices supporting CUDA", deviceCount);
            }
        }

        PRINT("Device {} name {}", dev, deviceProp.name);
        PRINT("\tComputational Capabilities: {}.{}", deviceProp.major, deviceProp.minor);
        PRINT("\tMaximum global memory size: {}", deviceProp.totalGlobalMem);
        PRINT("\tMaximum constant memory size: {}", deviceProp.totalConstMem);
        PRINT("\tMaximum shared memory size per block: {}", deviceProp.sharedMemPerBlock);
        PRINT("\tMaximum block dimensions: {}x{}x{}", deviceProp.maxThreadsDim[0],
                                                    deviceProp.maxThreadsDim[1],
                                                    deviceProp.maxThreadsDim[2]);
        PRINT("\tMaximum grid dimensions: {}x{}x{}", deviceProp.maxGridSize[0],
                                                   deviceProp.maxGridSize[1],
                                                   deviceProp.maxGridSize[2]);
        PRINT("\tWarp size: {}", deviceProp.warpSize);
    }

    timer_stop(); //@@ stop the timer

    return 0;
}

