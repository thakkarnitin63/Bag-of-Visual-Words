cmake_minimum_required(VERSION 3.0.0)
project(BOW VERSION 0.1.0 LANGUAGES C CXX)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenCV package
find_package(OpenCV REQUIRED)

add_executable(BOW main.cpp BagofWords.cpp)

add_executable(kmean kmean.cpp)

# Link OpenCV libraries
target_link_libraries(BOW ${OpenCV_LIBS})

target_link_libraries(kmean ${OpenCV_LIBS})
