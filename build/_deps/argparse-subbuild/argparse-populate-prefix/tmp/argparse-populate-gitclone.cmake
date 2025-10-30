
if(NOT "/workspace/ONNXim/build/_deps/argparse-subbuild/argparse-populate-prefix/src/argparse-populate-stamp/argparse-populate-gitinfo.txt" IS_NEWER_THAN "/workspace/ONNXim/build/_deps/argparse-subbuild/argparse-populate-prefix/src/argparse-populate-stamp/argparse-populate-gitclone-lastrun.txt")
  message(STATUS "Avoiding repeated git clone, stamp file is up to date: '/workspace/ONNXim/build/_deps/argparse-subbuild/argparse-populate-prefix/src/argparse-populate-stamp/argparse-populate-gitclone-lastrun.txt'")
  return()
endif()

execute_process(
  COMMAND ${CMAKE_COMMAND} -E rm -rf "/workspace/ONNXim/build/ext/argparse"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to remove directory: '/workspace/ONNXim/build/ext/argparse'")
endif()

# try the clone 3 times in case there is an odd git clone issue
set(error_code 1)
set(number_of_tries 0)
while(error_code AND number_of_tries LESS 3)
  execute_process(
    COMMAND "/usr/bin/git"  clone --no-checkout --config "advice.detachedHead=false" "https://github.com/p-ranav/argparse.git" "argparse"
    WORKING_DIRECTORY "/workspace/ONNXim/build/ext"
    RESULT_VARIABLE error_code
    )
  math(EXPR number_of_tries "${number_of_tries} + 1")
endwhile()
if(number_of_tries GREATER 1)
  message(STATUS "Had to git clone more than once:
          ${number_of_tries} times.")
endif()
if(error_code)
  message(FATAL_ERROR "Failed to clone repository: 'https://github.com/p-ranav/argparse.git'")
endif()

execute_process(
  COMMAND "/usr/bin/git"  checkout v2.9 --
  WORKING_DIRECTORY "/workspace/ONNXim/build/ext/argparse"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to checkout tag: 'v2.9'")
endif()

set(init_submodules TRUE)
if(init_submodules)
  execute_process(
    COMMAND "/usr/bin/git"  submodule update --recursive --init 
    WORKING_DIRECTORY "/workspace/ONNXim/build/ext/argparse"
    RESULT_VARIABLE error_code
    )
endif()
if(error_code)
  message(FATAL_ERROR "Failed to update submodules in: '/workspace/ONNXim/build/ext/argparse'")
endif()

# Complete success, update the script-last-run stamp file:
#
execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy
    "/workspace/ONNXim/build/_deps/argparse-subbuild/argparse-populate-prefix/src/argparse-populate-stamp/argparse-populate-gitinfo.txt"
    "/workspace/ONNXim/build/_deps/argparse-subbuild/argparse-populate-prefix/src/argparse-populate-stamp/argparse-populate-gitclone-lastrun.txt"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to copy script-last-run stamp file: '/workspace/ONNXim/build/_deps/argparse-subbuild/argparse-populate-prefix/src/argparse-populate-stamp/argparse-populate-gitclone-lastrun.txt'")
endif()

