#!/bin/bash

tmp_dir=$(dirname $(realpath $0))
my_dir=$(cd $tmp_dir/.. > /dev/null ; pwd)

# change this variable if needed
packageDir=$my_dir/build/package
scriptsDir=$my_dir/scripts

export PATH=$scriptsDir:$packageDir/bin:$PATH
export PYTHONPATH=$scriptsDir:$PYTHONPATH

export LD_LIBRARY_PATH=$packageDir/lib:$LD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=$packageDir/lib:$DYLD_LIBRARY_PATH

