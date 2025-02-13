cmake_minimum_required(VERSION 3.10)

project(SinSum)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(SinSum main.cpp)

set(SOURCE_FILE main.cpp)

add_library(SinSum_obj OBJECT ${SOURCE_FILE})

#add option