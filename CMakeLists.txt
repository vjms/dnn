cmake_minimum_required(VERSION 3.0.0)
project(dnn VERSION 0.1.0)

include(CTest)
enable_testing()

cmake_policy (SET CMP0076 NEW)

add_executable(${CMAKE_PROJECT_NAME} main.cpp)

add_subdirectory(src)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
