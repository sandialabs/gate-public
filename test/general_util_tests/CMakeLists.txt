cmake_minimum_required(VERSION 3.8)
set(CMAKE_CXX_STANDARD 11)

# Test for stream_managed.cuh
set(TARGET_NAME rk4_test)

add_executable(${TARGET_NAME} ../test_main.cpp rk4_test.cu)
target_link_libraries(${TARGET_NAME} gtest gmock gtest_main)
set_target_properties(${TARGET_NAME} PROPERTIES FOLDER test)

gtest_discover_tests(${TARGET_NAME})

set(TARGET_NAME eigen_test)

add_executable(${TARGET_NAME} ../test_main.cpp eigen_test.cu)
target_link_libraries(${TARGET_NAME} gtest gmock gtest_main)
set_target_properties(${TARGET_NAME} PROPERTIES FOLDER test)

gtest_discover_tests(${TARGET_NAME})

set(TARGET_NAME eigen_cpp_test)

add_executable(${TARGET_NAME} ../test_main.cpp eigen_cpp_test.cpp)
target_link_libraries(${TARGET_NAME} gtest gmock gtest_main)
set_target_properties(${TARGET_NAME} PROPERTIES FOLDER test)

gtest_discover_tests(${TARGET_NAME})