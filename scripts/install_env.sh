#!/bin/bash

tmp_dir=$(dirname ${BASH_SOURCE[0]})
my_dir=$(cd $tmp_dir/.. > /dev/null ; pwd)

# change this variable if needed
installDir=$my_dir/install

export CLASSPATH=$installDir/jar/*

export PATH=$installDir/bin:$PATH

export LD_LIBRARY_PATH=$installDir/lib:$LD_LIBRARY_PATH

