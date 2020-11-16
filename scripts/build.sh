#!/bin/bash

tmp_dir=$(dirname ${BASH_SOURCE[0]})
base_dir=$(cd $tmp_dir/.. > /dev/null ; pwd)

registry_url="10.15.86.19:25443"

tag_name="$registry_url/diffraflow"
docker build $base_dir -t $tag_name && docker push $tag_name

