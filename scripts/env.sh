#!/bin/bash

tmp_dir=$(dirname ${BASH_SOURCE[0]})
my_dir=$(cd $tmp_dir/.. > /dev/null ; pwd)

export CLASSPATH=

# dispatcher
export CLASSPATH=$CLASSPATH:$my_dir/dispatcher/build/libs/*

# combiner
export PATH=combiner/build/install/main/release:$PATH

# testing
export CLASSPATH=$CLASSPATH:$my_dir/testing/build/libs/*

