#!/bin/bash

tmp_dir=$(dirname ${BASH_SOURCE[0]})
my_dir=$(cd $tmp_dir/.. > /dev/null ; pwd)

export CLASSPATH=

# dispatcher
echo "Project 'dispatcher'"
export CLASSPATH=$CLASSPATH:$my_dir/dispatcher/build/libs/*

# testing
echo "Project 'testing'"
export CLASSPATH=$CLASSPATH:$my_dir/testing/build/libs/*

# cpp-application
for name in combiner ingester monitor
do
    echo "Project '$name'"
    export PATH=$my_dir/$name/build/exe/main/release:$PATH
done

# cpp-library
echo "Project 'common'"
export LD_LIBRARY_PATH=$my_dir/common/build/lib/main/release:$LD_LIBRARY_PATH

