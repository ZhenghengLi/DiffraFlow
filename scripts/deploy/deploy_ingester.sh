#!/bin/bash

tmp_dir=$(dirname ${BASH_SOURCE[0]})
base_dir=$(cd $tmp_dir/../.. > /dev/null ; pwd)
chart_dir=$base_dir/deploy/diffraflow/charts
value_dir=$base_dir/deploy/diffraflow/values

helm -n diffraflow install -f $value_dir/ingester-01.yaml ingester-01 $chart_dir/ingester
helm -n diffraflow install -f $value_dir/ingester-02.yaml ingester-02 $chart_dir/ingester
helm -n diffraflow install -f $value_dir/ingester-03.yaml ingester-03 $chart_dir/ingester
helm -n diffraflow install -f $value_dir/ingester-04.yaml ingester-04 $chart_dir/ingester
# helm -n diffraflow install -f $value_dir/ingester-05.yaml ingester-05 $chart_dir/ingester
# helm -n diffraflow install -f $value_dir/ingester-06.yaml ingester-06 $chart_dir/ingester
# helm -n diffraflow install -f $value_dir/ingester-07.yaml ingester-07 $chart_dir/ingester
# helm -n diffraflow install -f $value_dir/ingester-08.yaml ingester-08 $chart_dir/ingester

