openssl req -newkey rsa:4096 -nodes -sha256 -keyout cu01.key -x509 -days 10000 -out cu01.crt -config cu01.conf -extensions v3_ext

## on each node, copy cu01.crt into /etc/docker/certs.d/10.15.86.19:25443 as ca.crt, in order to instruct every docker daemon to trust it.
## $ sudo mkdir -p /etc/docker/certs.d/10.15.86.19:25443
## $ sudo cp cu01.crt /etc/docker/certs.d/10.15.86.19:25443/ca.crt
## and there is no need to restart docker daemon.

