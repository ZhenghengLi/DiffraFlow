#!/bin/bash

tmp_dir=$(dirname ${BASH_SOURCE[0]})
my_dir=$(cd $tmp_dir/.. > /dev/null ; pwd)

# dispatcher
export CLASSPATH=
export CLASSPATH=$CLASSPATH:$my_dir/dispatcher/build/libs/*

# ingester
export PATH=ingester/build/install/main/release:$PATH

