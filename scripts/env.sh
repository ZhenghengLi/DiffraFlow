#!/bin/bash

tmp_dir=$(dirname ${BASH_SOURCE[0]})
my_dir=$(cd $tmp_dir/.. > /dev/null ; pwd)

export CLASSPATH=

# dispatcher
export CLASSPATH=$CLASSPATH:$my_dir/dispatcher/build/libs/*

# combiner
export PATH=$my_dir/combiner/build/install/main/release:$PATH

# ingester
export PATH=$my_dir/ingester/build/install/main/release:$PATH

# monitor
export PATH=$my_dir/monitor/build/install/main/release:$PATH

# testing
export CLASSPATH=$CLASSPATH:$my_dir/testing/build/libs/*

