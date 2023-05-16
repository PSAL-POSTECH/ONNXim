cmake_minimum_required(VERSION 3.16)
project(ramulator_project)

file(GLOB_RECURSE RAMULATOR_SRCS CONFIGURE_DEPENDS src/*.cpp)
add_library(ramulator STATIC ${RAMULATOR_SRCS})
target_include_directories(ramulator 
  PUBLIC include
  PRIVATE include/ramulator
  PRIVATE src
)
target_compile_options(ramulator PRIVATE -Wall -O3)
