#!/bin/bash

helm -n diffraflow uninstall trigger-1
helm -n diffraflow uninstall sender-1
helm -n diffraflow uninstall sender-2
helm -n diffraflow uninstall sender-3
helm -n diffraflow uninstall sender-4
helm -n diffraflow uninstall dispatcher-1
helm -n diffraflow uninstall combiner-1
helm -n diffraflow uninstall ingester-01
helm -n diffraflow uninstall ingester-02
helm -n diffraflow uninstall ingester-03
helm -n diffraflow uninstall ingester-04
helm -n diffraflow uninstall ingester-05
helm -n diffraflow uninstall ingester-06
helm -n diffraflow uninstall ingester-07
helm -n diffraflow uninstall ingester-08

