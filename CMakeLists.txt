cmake_minimum_required(VERSION 3.8)
project(bayes)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 17)

set(SOURCE_FILES functions.cpp tests/catch.h tests/catch.cpp tests/testReplaceAllValues.cpp tests/manual.cpp functions.h tests/testImputeHiddenNode.cpp)
add_executable(bayes ${SOURCE_FILES})
target_link_libraries(bayes ${ARMADILLO_LIBRARIES})
