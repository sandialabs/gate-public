cmake_minimum_required(VERSION 3.8)
set(CMAKE_CXX_STANDARD 11)

# Test for stream_managed.cuh
set(TARGET_NAME gate_dubin_tests)

add_executable(${TARGET_NAME} ../test_main.cpp gate_dubin_test.cu)
target_link_libraries(${TARGET_NAME} gtest gmock gtest_main)
set_target_properties(${TARGET_NAME} PROPERTIES FOLDER test)

gtest_discover_tests(gate_dubin_tests)

set(TARGET_NAME gate_mass_spring_damper_tests)

add_executable(${TARGET_NAME} ../test_main.cpp gate_mass_spring_damper_test.cu)
target_link_libraries(${TARGET_NAME} gtest gmock gtest_main)
set_target_properties(${TARGET_NAME} PROPERTIES FOLDER test)

gtest_discover_tests(gate_mass_spring_damper_tests)

set(TARGET_NAME gate_cartpole_tests)
add_executable(${TARGET_NAME} ../test_main.cpp gate_cartpole_test.cu)
target_link_libraries(${TARGET_NAME} gtest gmock gtest_main)
set_target_properties(${TARGET_NAME} PROPERTIES FOLDER test)
gtest_discover_tests(gate_cartpole_tests)

set(TARGET_NAME gate_quad_tests)
add_executable(${TARGET_NAME} ../test_main.cpp gate_quad_test.cu)
target_link_libraries(${TARGET_NAME} gtest gmock gtest_main)
set_target_properties(${TARGET_NAME} PROPERTIES FOLDER test)
gtest_discover_tests(gate_quad_tests)
