# DiffraFlow

High volume data acquisition and online analysis system for SHINE project

## dependencies

1. dispatcher:  
    (1) org.apache.kafka:kafka-clients:2.3.0

2. combiner:  
    (2) https://github.com/edenhill/librdkafka (v1.2.0)

## blueprint

![plan](docs/images/plan.png)

## online event-building

![online-event-building](docs/images/online_event_building.png)

## dispatcher

![dispatcher](docs/images/dispatcher_node.png)

## combiner

![combiner](docs/images/combiner_node.png)

## ingester and monitor

![ingester-and-monitor](docs/images/ingester_and_monitor.png)

## subsecond FFB

![FFB](docs/images/ffb.png)
