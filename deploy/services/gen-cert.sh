openssl req -config cu01.conf -new -nodes -x509 -newkey rsa:4096 -sha256 -keyout cu01.key -out cu01.crt -days 36500

## on each node, copy cu01.crt into /etc/docker/certs.d/10.15.85.28:25443 as ca.crt, in order to instruct every docker daemon to trust it.
## $ sudo mkdir -p /etc/docker/certs.d/10.15.85.28:25443
## $ sudo cp cu01.crt /etc/docker/certs.d/10.15.85.28:25443/ca.crt
## and there is no need to restart docker daemon.

