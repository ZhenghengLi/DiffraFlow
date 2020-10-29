#!/bin/bash

tmp_dir=$(dirname ${BASH_SOURCE[0]})
base_dir=$(cd $tmp_dir/../.. > /dev/null ; pwd)
chart_dir=$base_dir/deploy/diffraflow/charts
value_dir=$base_dir/deploy/diffraflow/values

helm -n diffraflow install -f $value_dir/aggregator-1.yaml aggregator-1 $chart_dir/aggregator
helm -n diffraflow install -f $value_dir/dashboard-1.yaml dashboard-1 $chart_dir/dashboard

