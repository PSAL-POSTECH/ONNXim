CMAKE_MINIMUM_REQUIRED(VERSION 3.15)

# project setting
set(LIB_NAME "Simulator")

# set source and headers
file(GLOB_RECURSE SRC_FILES
  "${CMAKE_SOURCE_DIR}/src/*.h"
  "${CMAKE_SOURCE_DIR}/src/*.cc"
)

# build
add_executable(${LIB_NAME} ${SRC_FILES})
add_library(${LIB_NAME}_lib ${SRC_FILES})
