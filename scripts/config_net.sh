sysctl -w net.core.rmem_max=134217728
sysctl -w net.core.rmem_default=33554432
sysctl -w net.core.wmem_max=134217728
sysctl -w net.core.wmem_default=33554432
sysctl -w net.core.netdev_max_backlog=5000000
ifconfig bond0 mtu 9000 txqueuelen 500000 up
