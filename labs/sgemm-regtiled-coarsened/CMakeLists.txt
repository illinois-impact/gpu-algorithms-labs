
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(sgemm LANGUAGES CXX CUDA)


find_package(CUDA REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(VERBOSE_BUILD ON)
set(CMAKE_CXX_STANDARD 11)
set(DEFAULT_BUILD_TYPE "Release")


set_property(GLOBAL PROPERTY USE_FOLDERS ON)

include(CTest)

add_executable(sgemm
  eval.cu
  helper.hpp
  template.cu
  template.hu
  common/catch.hpp
  common/fmt.hpp
  common/clara.hpp
  common/utils.hpp
)


# CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS Auto)

target_compile_features(sgemm PUBLIC cxx_std_11)

# We need to explicitly state that we need all CUDA files in the particle
# library to be built with -dc as the member functions could be called by
# other libraries and executables
set_target_properties( sgemm PROPERTIES
                       # CUDA_SEPARABLE_COMPILATION ON
                       CUDA_RESOLVE_DEVICE_SYMBOLS ON
                      )

target_link_libraries(sgemm ${CUDA_LIBRARIES})


include_directories(sgemm
  ${PROJECT_SOURCE_DIR}/src
  ${CUDA_INCLUDE_DIRS}
)

if(APPLE)
  # We need to add the default path to the driver (libcuda.dylib) as an rpath,
  # so that the static cuda runtime can find it at runtime.
  set_property(TARGET sgemm PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()



enable_testing()
