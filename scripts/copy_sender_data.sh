#!/bin/bash

if [[ $# -lt 4 ]]; then
    echo "USAGE: copy_sender_data.sh <node_map.txt> <node_name> <from_dir> <to_dir>"
    exit 1
fi

node_map_file=$1
node_name=$2
from_dir=$3
to_dir=$4

function get_module_id {
    while read -r line; do
       # skip empty line and comment
       if [[ -z $line || $line =~ ^# ]]; then
           continue
       fi
       # extract fields
       IFS=, read -r hostname module address port <<< $line
       if [[ $hostname == $2 ]]; then
           echo $module
       fi
    done < $1
}

module_id=$(printf "%02d" $(get_module_id $node_map_file $node_name))

cp -v $from_dir/AGIPD-BIN-R0243-M${module_id}-S*.dat $to_dir

