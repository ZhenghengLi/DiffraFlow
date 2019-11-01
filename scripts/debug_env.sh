#!/bin/bash

tmp_dir=$(dirname ${BASH_SOURCE[0]})
my_dir=$(cd $tmp_dir/.. > /dev/null ; pwd)

export CLASSPATH=

# java
for name in dispatcher
do
    echo "Project 'diffraflow/$name'"
    export CLASSPATH=$CLASSPATH:$my_dir/diffraflow/$name/build/libs/*
done

# cpp-application
for name in combiner ingester monitor
do
    echo "Project 'diffraflow/$name'"
    export PATH=$my_dir/diffraflow/$name/build/exe/main/release:$PATH
done

# cpp-library
echo "Project 'common'"
export LD_LIBRARY_PATH=$my_dir/diffraflow/common/build/lib/main/release:$LD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=$my_dir/diffraflow/common/build/lib/main/release:$DYLD_LIBRARY_PATH

# testing
echo "Project 'testing'"
export CLASSPATH=$CLASSPATH:$my_dir/testing/build/libs/*

