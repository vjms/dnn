cmake_minimum_required(VERSION 3.0.0)
project(dnn VERSION 0.1.0)

include(CTest)
enable_testing()

set(CMAKE_CXX_STANDARD 20)

cmake_policy (SET CMP0076 NEW)

add_executable(${CMAKE_PROJECT_NAME} main.cpp)

target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE -Wall -Wextra -Wshadow -Wnon-virtual-dtor -pedantic)

add_subdirectory(src)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
