#!/bin/bash

tmp_dir=$(dirname ${BASH_SOURCE[0]})
base_dir=$(cd $tmp_dir/../.. > /dev/null ; pwd)
chart_dir=$base_dir/deploy/diffraflow/charts
value_dir=$base_dir/deploy/diffraflow/values

helm -n diffraflow install -f $value_dir/combiner-1.yaml combiner-1 $chart_dir/combiner
helm -n diffraflow install -f $value_dir/dispatcher-1.yaml dispatcher-1 $chart_dir/dispatcher

