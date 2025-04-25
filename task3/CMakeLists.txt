cmake_minimum_required(VERSION 3.12)
project(matrix_vector_product CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-O3)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_executable(matrix_prod
    matrix_prod.cpp
)
target_link_libraries(matrix_prod
    PRIVATE 
    Threads::Threads
)

add_executable(app 
    cl_ser_app.cpp
)
target_link_libraries(app 
    PRIVATE 
    Threads::Threads
)