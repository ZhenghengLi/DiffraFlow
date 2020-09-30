#!/bin/bash

tmp_dir=$(dirname ${BASH_SOURCE[0]})
base_dir=$(cd $tmp_dir/../.. > /dev/null ; pwd)
yaml_dir=$base_dir/deploy/services

kubectl apply -f $yaml_dir/zookeeper.yaml
kubectl apply -f $yaml_dir/pulsar-bookkeeper.yaml
kubectl apply -f $yaml_dir/pulsar-broker.yaml

