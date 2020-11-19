#!/bin/bash

helm -n diffraflow uninstall trigger-1
helm -n diffraflow uninstall sender-1
helm -n diffraflow uninstall sender-2
helm -n diffraflow uninstall sender-3
helm -n diffraflow uninstall sender-4

