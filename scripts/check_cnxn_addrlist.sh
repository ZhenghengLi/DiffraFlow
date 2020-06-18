#!/bin/bash

if [[ -z $1 ]]; then
    echo "USAGE: check_connection.sh address_list.txt"
    exit 1
fi

function check_cnxn_addrlist {
    while read -r line; do
        # skip empty line and comment
        if [[ -z $line || $line =~ ^# ]]; then
            continue
        fi
        # remove http:// or https:// header
        if [[ $line =~ ^\s*https?:// ]]; then
            line=$(sed -r "s|^\s*https?://||" <<< $line)
        fi
        # replace NODE_NAME or NODE_IP
        if [[ $line =~ NODE_NAME && ! -z $NODE_NAME ]]; then
            line=$(sed -r "s|NODE_NAME|$NODE_NAME|" <<< $line)
        elif [[ $line =~ NODE_IP && ! -z $NODE_IP ]]; then
            line=$(sed -r "s|NODE_IP|$NODE_IP|" <<< $line)
        fi
        # extract host and port, then check TCP connection
        IFS=: read -r host port <<< $line
        if [[ -z $host || -z $port ]]; then
            return 1
        fi
        if ! nc -z -v -w 1 $host $port; then
            return 1
        fi
    done < $1
}

until check_cnxn_addrlist $1; do
    sleep 1
    echo "continue checking ..."
done
