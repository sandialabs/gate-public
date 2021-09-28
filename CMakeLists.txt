# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.10)

project ("gate-public" LANGUAGES C CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES "61")
set(CMAKE_CXX_STANDARD 11)
# Requires the /permissive- flag for Eigen to play nice with the MSVC compiler
if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /permissive")
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -maxrregcount=32 -arch=sm_61 -w -Xptxas -dlcm=cg,-v")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} --device-debug --keep")


enable_testing()

# Place the libraries and the executables in the same plae
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Make sure that we use shared libraries for GTEST, otherwise the build will fail'
set( gtest_force_shared_crt ON CACHE BOOL "Always use msvcrt.dll" FORCE)

find_package(CUDA REQUIRED)

# Include sub-projects.
add_subdirectory ("googletest-master")

# Allows us to use gtest_discover_tests
include(GoogleTest)

# Add our local version of eigen
include_directories(SYSTEM "eigen-3.3.8/")
add_subdirectory ("gate")

# Add the testng subdirectory
add_subdirectory(test)