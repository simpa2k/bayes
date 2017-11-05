cmake_minimum_required(VERSION 3.8)
project(bayes)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 17)

set(SOURCE_FILES classifier/functions.cpp tests/catch.h tests/catch.cpp tests/classifier/testReplaceAllValues.cpp tests/classifier/manual.cpp classifier/functions.h tests/classifier/testImputeHiddenNode.cpp tests/directedGraph/graphTest.cpp directedGraph/Graph.cpp directedGraph/Graph.h network/BayesianNetwork.cpp network/BayesianNetwork.h classifier/Classifier.cpp classifier/Classifier.h)
add_executable(bayes ${SOURCE_FILES})
target_link_libraries(bayes ${ARMADILLO_LIBRARIES})
