SET(BINARY Simulator_test)
SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR})

#
# import googletest as an external project
#
include(ExternalProject)
include(GoogleTest)
externalproject_add(
  GoogleTest
  URL https://github.com/google/googletest/archive/release-1.8.1.zip
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/lib
  CMAKE_ARGS -DCMAKE_CXX_FLAGS:STRING="-D_GLIBCXX_USE_CXX11_ABI=0"
  INSTALL_COMMAND ""
)

externalproject_get_property(GoogleTest source_dir)
include_directories(${source_dir}/googletest/include)
include_directories(${source_dir}/googlemock/include)

externalproject_get_property(GoogleTest binary_dir)
set(GTEST_LIBRARY_PATH ${binary_dir}/googlemock/gtest/${CMAKE_FIND_LIBRARY_PREFIXES}gtest.a)
set(GTEST_LIBRARY GTest::GTest)
add_library(${GTEST_LIBRARY} UNKNOWN IMPORTED)
set_target_properties(${GTEST_LIBRARY} PROPERTIES
  IMPORTED_LOCATION ${GTEST_LIBRARY_PATH})
add_dependencies(${GTEST_LIBRARY} GoogleTest)

set(GMOCK_LIBRARY_PATH ${binary_dir}/googlemock/${CMAKE_FIND_LIBRARY_PREFIXES}gmock.a)
set(GMOCK_LIBRARY GTest::GMock)
add_library(${GMOCK_LIBRARY} UNKNOWN IMPORTED)
set_target_properties(${GMOCK_LIBRARY} PROPERTIES
  IMPORTED_LOCATION ${GMOCK_LIBRARY_PATH})
add_dependencies(${GMOCK_LIBRARY} GoogleTest)

file(GLOB_RECURSE TEST_SOURCES LIST_DIRECTORIES false *.h *.cc)
SET(SOURCES ${TEST_SOURCES})
add_executable(${BINARY} ${TEST_SOURCES})

target_include_directories(Simulator_test PUBLIC ${ONNX_INCLUDE_DIRS})
target_include_directories(Simulator_test PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(Simulator_test Simulator_lib)
# target_link_libraries(Simulator_test ramulator booksim2)
# target_link_libraries(Simulator_test nlohmann_json::nlohmann_json ${PROTOBUF_LIB} onnx_proto ${CONAN_LIBS} stdc++fs spdlog::spdlog)
target_link_libraries(Simulator_test GTest::GTest GTest::GMock)

# gtest_discover_tests(
#   Simulator_test
#   WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
# )
add_test(NAME Simulator_test COMMAND Simulator_test)
