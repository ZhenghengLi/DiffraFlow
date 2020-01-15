#!/bin/bash

tmp_dir=$(dirname ${BASH_SOURCE[0]})
prefix_dir=$(cd $tmp_dir/.. > /dev/null ; pwd)

export CLASSPATH=$prefix_dir/jar/*

export PATH=$prefix_dir/bin:$PATH

export LD_LIBRARY_PATH=$prefix_dir/lib:$LD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=$prefix_dir/lib:$DYLD_LIBRARY_PATH

