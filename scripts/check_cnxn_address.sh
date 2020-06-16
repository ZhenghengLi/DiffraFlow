#!/bin/bash

if [[ -z $1 ]]; then
    echo "USAGE: check_connection.sh <host1:port1> <host2:port2> ..."
    exit 1
fi

function check_cnxn_address {
    for item in $@; do
        IFS=: read -r host port <<< $item
        if [[ -z $host || -z $port ]]; then
            return 1
        fi
        if ! nc -z -v -w 1 $host $port; then
            return 1
        fi
    done
}

until check_cnxn_address $@; do
    sleep ${2:-1}
    echo "continue checking ..."
done
