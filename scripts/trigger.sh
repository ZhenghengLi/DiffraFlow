#!/bin/bash

interval=395
if [[ -n $1 && $1 -ge 0 ]]; then
    interval=$1
fi

counts=89000
if [[ -n $2 && $2 -le 89000 && $2 -gt 0 ]]; then
    counts=$2
fi

echo "interval = $interval"
echo "counts = $counts"

set -x

kubectl -n diffraflow exec pod/trigger-1 -- trigger -s /config/sender_addresses.txt -l /config/log4cxx.properties -e 0 -c $counts -i $interval

