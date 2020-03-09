#!/bin/bash

if [ -z $1 ]; then
    echo "USAGE: check_connection.sh address_list.txt"
    exit 1
fi

function check_cnxn_address {
    IFS=: read -r host port <<< $1
    if [[ -z $host || -z $port ]]; then
        return 1
    fi
    if ! nc -z -v -w 2 $host $port; then
        return 1
    fi
}

until check_cnxn_address $1; do
    sleep ${2:-2}
    echo "continue checking ..."
done
