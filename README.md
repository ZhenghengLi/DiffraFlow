# DiffraFlow

High volume data acquisition and on-the-fly data analysis for area detectors.

Started on 12th September 2019, this project is proposed to develop an optimized distributed software system (after FPGA) for streaming the data from area detectors (source) to distributed file system (sink) at very high overall rate (e.g. >100GiB/s).

The software should be capable of doing event-building online for all events before data sink, as well as online doing calibration, deep event filtering and fast feedback based on full event image data.

Currently, the overall design of data flow is as below, and more details will be in documents and wiki.

![plan](docs/images/plan.png)

The overall design is quite similar to LCLS-II's (see: https://ieeexplore.ieee.org/document/8533033). The difference could be that this design uses two PC layes to do data transfering and Data Reduction Pipline (DRP), which will make it possible to do online event-building under very high data rate condition.

This project is under developing ...
