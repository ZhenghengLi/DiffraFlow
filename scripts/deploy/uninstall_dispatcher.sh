#!/bin/bash

helm -n diffraflow uninstall dispatcher-1
helm -n diffraflow uninstall dispatcher-2
helm -n diffraflow uninstall dispatcher-3
helm -n diffraflow uninstall dispatcher-4

