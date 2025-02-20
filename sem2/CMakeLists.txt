cmake_minimum_required(VERSION 3.10)
project(UndefinedBehavior LANGUAGES CXX)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Определяем компилятор, устанавливаем переменную -DCMAKE_CXX_COMPILER="..." (значения переменной clang++, g++)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(COMPILER_NAME "g++")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(COMPILER_NAME "clang++")
else()
    message(FATAL_ERROR "Unsupported compiler: ${CMAKE_CXX_COMPILER_ID}")
endif()

# Собирайте с разными опциями оптимизации и смотрете, что выдают
# программы собранные разными компиляторами

# Опции компиляции (можно передавать через -DCMAKE_CXX_FLAGS="...")
if(NOT CMAKE_CXX_FLAGS)
    set(CMAKE_CXX_FLAGS -O2) # Значения по умолчанию
endif()

# Создаём папку для бинарников
set(OUTPUT_DIR ${CMAKE_BINARY_DIR}/${COMPILER_NAME})
file(MAKE_DIRECTORY ${OUTPUT_DIR})

# Находим все .cpp файлы
file(GLOB SOURCES "*.cpp")

# Создаём отдельный исполняемый файл для каждого .cpp файла
foreach(SRC ${SOURCES})
    get_filename_component(EXE_NAME ${SRC} NAME_WE)
    add_executable(${EXE_NAME} ${SRC})
    set_target_properties(${EXE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR})
    target_compile_features(${EXE_NAME} PRIVATE cxx_std_17)
endforeach()