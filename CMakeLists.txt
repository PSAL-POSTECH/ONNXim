cmake_minimum_required(VERSION 3.15.0)
set(project_name "AiFrameworkSim")
project(${project_name})

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include("${CMAKE_SOURCE_DIR}/build/conanbuildinfo.cmake")
conan_basic_setup()

# find_package(Boost 1.70 REQUIRED COMPONENTS program_options)
# message("LIB ${Boost_LIBRARY_DIRS}")
option(USE_RAMULATOR "USE_RAMULATOR" ON)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/extern)

# C++ setttings
set(CMAKE_CXX_STANDARD 20)
set(ONNX_ML 1)
set(JSON_BuildTests OFF CACHE INTERNAL "")
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/build/bin")
set(LIBRARY_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/build/lib")
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
message("BINARY DIR ${CMAKE_BINARY_DIR}")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-fsanitize=address)
    add_link_options(-fsanitize=address)
endif()

# Build source
add_subdirectory("${PROJECT_SOURCE_DIR}/src")

# Add libaray ramulator
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/ramulator_custom")

# Add libaray ramulator
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/ramulator2")
include_directories("${PROJECT_SOURCE_DIR}/extern/ramulator2/src")
include_directories("${PROJECT_SOURCE_DIR}/extern/ramulator2/resources/ndp_wrappers")

# Add libaray booksim
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/booksim")

# Add libary protobuf
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/protobuf/cmake" EXCLUDE_FROM_ALL)
set_target_properties(libprotoc PROPERTIES FOLDER "external/protobuf")
set_target_properties(protoc PROPERTIES FOLDER "external/protobuf")

# Add libaray onnx
add_definitions("-DONNX_NAMESPACE=onnx")
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/onnx" EXCLUDE_FROM_ALL)
set_target_properties(onnx PROPERTIES FOLDER "extern/onnx")
set_target_properties(onnx_proto PROPERTIES FOLDER "extern/onnx")

target_include_directories(Simulator PUBLIC ${ONNX_INCLUDE_DIRS})
target_link_libraries(Simulator ramulator1 booksim2 ramulator)
target_link_libraries(Simulator ${PROTOBUF_LIB} onnx_proto ${CONAN_LIBS} stdc++fs)

target_include_directories(Simulator_lib PUBLIC ${ONNX_INCLUDE_DIRS})
target_link_libraries(Simulator_lib ramulator1 booksim2 ramulator)
target_link_libraries(Simulator_lib ${PROTOBUF_LIB} onnx_proto ${CONAN_LIBS} stdc++fs)

enable_testing()
add_subdirectory("${PROJECT_SOURCE_DIR}/tests")

