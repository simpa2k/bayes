cmake_minimum_required(VERSION 3.8)
project(bayes)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 17)

set(SOURCE_FILES functions.cpp tests/catch.h tests/catch.cpp tests/brain/testReplaceAllValues.cpp tests/brain/manual.cpp functions.h tests/brain/testImputeHiddenNode.cpp tests/directedGraph/graphTest.cpp directedGraph/Graph.cpp directedGraph/Graph.h)
add_executable(bayes ${SOURCE_FILES})
target_link_libraries(bayes ${ARMADILLO_LIBRARIES})
