# DiffraFlow

**Diffra**ction image data **Flow**, aiming at the high volume data acquisition and online data analysis for area detectors.

Started on 12th September 2019 by Zhengheng Li after he joined BE-SHINE, this project is proposed to develop an optimized distributed software system (after FPGA) for streaming the data from area detectors (source) to distributed file system (sink) at very high overall input rate (e.g. >100GiB/s).

The software should be capable of doing online event-building for all events before data sink, as well as doing online calibration, deep event filtering and quasi-realtime analysis for fast feedback based on full event image data.

Currently, the overall design of data flow is as below, and more details will be in docs and wiki.

![plan](docs/images/plan.png)

The overall design is quite similar to LCLS-II's (see: <https://ieeexplore.ieee.org/document/8533033).> The difference could be that this design uses a MapReduce-like data shuffling process to do the online event-building, which will make it easier to do online data filtering based on full event data.

This project is under developing ...
