sysctl -w net.core.rmem_max=201326592
sysctl -w net.core.rmem_default=33554432
sysctl -w net.core.wmem_max=201326592
sysctl -w net.core.wmem_default=33554432
sysctl -w net.core.netdev_max_backlog=2000000
ifconfig bond0 mtu 9000 txqueuelen 200000 up
