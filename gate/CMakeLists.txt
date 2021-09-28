cmake_minimum_required (VERSION 3.8)

# Add the correct cuda headers 
include_directories (SYSTEM ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}/../../include/ )
include_directories( ${CMAKE_CURRENT_LIST_DIR}/include/ 
                     ${CMAKE_CURRENT_LIST_DIR}/include/dynamics
					 ${CMAKE_CURRENT_LIST_DIR}/include/controllers/
					 ${CMAKE_CURRENT_LIST_DIR}/include/perturbations/
		             ${CMAKE_CURRENT_LIST_DIR}/include/trajectory_generation/)

add_subdirectory (src)


# RTE executable for spring mass damper (smd)
set(TARGET_NAME gate_smd_main)
add_executable(${TARGET_NAME} "examples/GATE_smd.cu")
target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# RTE executable for dubins
set(TARGET_NAME gate_dubins_main)
add_executable(${TARGET_NAME} "examples/GATE_dubins.cu")
target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# RTE executable for cartpole
set(TARGET_NAME gate_cartpole_main)
add_executable(${TARGET_NAME} "examples/GATE_cartpole.cu")
target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# RTE executable for quadrotor
set(TARGET_NAME gate_quad_main)
add_executable(${TARGET_NAME} "examples/GATE_quad.cu")
target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

