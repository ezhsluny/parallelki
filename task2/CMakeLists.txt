cmake_minimum_required(VERSION 3.10)

project(Task2)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED True)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build (Debug or Release)" FORCE)
endif()

find_package(OpenMP REQUIRED)

add_executable(matrix_prod matrix_product.c)
add_executable(integrate integrate.c)
add_executable(system_sol system_sol.cpp)

target_link_libraries(matrix_prod PRIVATE OpenMP::OpenMP_C)
target_link_libraries(integrate PRIVATE OpenMP::OpenMP_C)
target_link_libraries(system_sol PRIVATE OpenMP::OpenMP_C)