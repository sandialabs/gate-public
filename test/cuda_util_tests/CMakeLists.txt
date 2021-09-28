cmake_minimum_required(VERSION 3.8)
set(CMAKE_CXX_STANDARD 11)

# Test for dynamics_stream_managed.cuh
set(TARGET_NAME dynamics_stream_managed_tests)

add_executable(${TARGET_NAME} ../test_main.cpp dynamics_stream_managed_test.cu)
target_link_libraries(${TARGET_NAME} gtest gmock gtest_main)
set_target_properties(${TARGET_NAME} PROPERTIES FOLDER test)

gtest_discover_tests(${TARGET_NAME})
