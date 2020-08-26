#!/bin/bash

tmp_dir=$(dirname ${BASH_SOURCE[0]})
base_dir=$(cd $tmp_dir/../.. > /dev/null ; pwd)
chart_dir=$base_dir/deploy/diffraflow/charts
value_dir=$base_dir/deploy/diffraflow/values

helm -n diffraflow install -f $value_dir/ingester-01.yaml ingester-01 $chart_dir/ingester
helm -n diffraflow install -f $value_dir/ingester-02.yaml ingester-02 $chart_dir/ingester

