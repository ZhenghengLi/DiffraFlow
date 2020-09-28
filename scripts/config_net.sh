sysctl -w net.core.rmem_max=67108864
sysctl -w net.core.rmem_default=33554432
sysctl -w net.core.wmem_max=67108864
sysctl -w net.core.wmem_default=33554432
sysctl -w net.core.netdev_max_backlog=100000
ifconfig bond0 mtu 9000 txqueuelen 10000 up
