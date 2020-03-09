#!/bin/bash

if [[ -z $1 ]]; then
    echo "USAGE: check_connection.sh address_list.txt"
    exit 1
fi

function check_cnxn_addrlist {
    while read -r line; do
        if [[ -z $line || $line =~ ^# ]]; then
            continue
        fi
        IFS=: read -r host port <<< $line
        if [[ -z $host || -z $port ]]; then
            return 1
        fi
        if ! nc -z -v $host $port; then
            return 1
        fi
    done < "$1"
}

until check_cnxn_addrlist $1; do
    sleep ${2:-1}
    echo "continue checking ..."
done
