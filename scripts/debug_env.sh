#!/bin/bash

tmp_dir=$(dirname ${BASH_SOURCE[0]})
my_dir=$(cd $tmp_dir/.. > /dev/null ; pwd)

# change this variable if needed
packageDir=$my_dir/build/package_debug

export CLASSPATH=$packageDir/jar/*

export PATH=$packageDir/bin:$PATH

export LD_LIBRARY_PATH=$packageDir/lib:$LD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=$packageDir/lib:$DYLD_LIBRARY_PATH

