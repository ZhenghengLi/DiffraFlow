#!/bin/bash

tmp_dir=$(dirname ${BASH_SOURCE[0]})
base_dir=$(cd $tmp_dir/../.. > /dev/null ; pwd)
chart_dir=$base_dir/deploy/diffraflow/charts
value_dir=$base_dir/deploy/diffraflow/values

helm -n diffraflow install -f $value_dir/combiner-1.yaml combiner-1 $chart_dir/combiner
helm -n diffraflow install -f $value_dir/combiner-2.yaml combiner-2 $chart_dir/combiner
helm -n diffraflow install -f $value_dir/combiner-3.yaml combiner-3 $chart_dir/combiner
helm -n diffraflow install -f $value_dir/combiner-4.yaml combiner-4 $chart_dir/combiner

