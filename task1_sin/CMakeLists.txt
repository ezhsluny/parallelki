cmake_minimum_required(VERSION 3.10)

project(SinSum)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

option(USE_DOUBLE "Use double instead of float" OFF)

if(USE_DOUBLE)
    add_definitions(-DFLOAT_TYPE=double)
else()
    add_definitions(-DFLOAT_TYPE=float)
endif()

add_executable(SinSum main.cpp)
