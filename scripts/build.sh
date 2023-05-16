#!/bin/bash
CC=`which gcc`
CXX=`which g++`

mkdir -p ./build
cd build
conan install ..
cd ..
cmake --no-warn-unused-cli -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE -DCMAKE_BUILD_TYPE:STRING=Debug -DCMAKE_C_COMPILER:FILEPATH=$CC -DCMAKE_CXX_COMPILER:FILEPATH=$CXX -S`pwd` -B`pwd`/build -G "Unix Makefiles"
/home/hhk971/anaconda3/envs/ai_framework/bin/cmake --build /home/hhk971/ai_framework/my-project/ai-framwork-sim/build --config Debug --target all -j 34 --
#cmake --build `pwd`/build --config Debug --target all -j 64 --
