#!/bin/bash

base_dir=/opt/mark/CppLibs/librdkafka

export CPATH=$base_dir/include:$CPATH
export LIBRARY_PATH=$base_dir/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$base_dir/lib:$LD_LIBRARY_PATH

