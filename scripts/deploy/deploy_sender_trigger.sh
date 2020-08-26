#!/bin/bash

tmp_dir=$(dirname ${BASH_SOURCE[0]})
base_dir=$(cd $tmp_dir/../.. > /dev/null ; pwd)
chart_dir=$base_dir/deploy/diffraflow/charts
value_dir=$base_dir/deploy/diffraflow/values

helm -n diffraflow install -f $value_dir/sender-1.yaml sender-1 $chart_dir/sender
helm -n diffraflow install -f $value_dir/sender-2.yaml sender-2 $chart_dir/sender
helm -n diffraflow install -f $value_dir/sender-3.yaml sender-3 $chart_dir/sender
helm -n diffraflow install -f $value_dir/sender-4.yaml sender-4 $chart_dir/sender
helm -n diffraflow install -f $value_dir/trigger-1.yaml trigger-1 $chart_dir/trigger

