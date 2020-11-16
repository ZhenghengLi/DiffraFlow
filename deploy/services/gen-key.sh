openssl req -newkey rsa:4096 -nodes -sha256 -keyout cu01.key -x509 -days 10000 -out cu01.crt -config cu01.conf -extensions v3_ext
