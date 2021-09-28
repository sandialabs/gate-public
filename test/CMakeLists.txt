# Add project includes and cuda includes for g++
include_directories(${PROJECT_SOURCE_DIR}/gate/include/
	$CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_subdirectory(cuda_util_tests)
add_subdirectory(general_util_tests)
add_subdirectory(gate_tests)
