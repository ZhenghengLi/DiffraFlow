#!/usr/bin/env python3

import argparse
import h5py
import numpy as np
import sys, os, zlib

from cxidb_euxfel_utils import get_image_dset, compose_image

parser = argparse.ArgumentParser(description='generate raw data files for one module')
parser.add_argument("data_dir", help = "directory that contains input data files")
parser.add_argument("mod_id", help = "module ID (0 -- 15)", type=int)
parser.add_argument("output_dir", help = "output directory")
parser.add_argument("-s", dest = "seg_count", help = "segments count", default=3, type=int)
parser.add_argument("-a", dest = "align_file", help = "alignment file", required = True)
parser.add_argument("-e", dest = "event_file", help = "event number file", required = True)
parser.add_argument("-c", dest = "calib_file", help = "calibration file", required = True)
parser.add_argument("-m", dest = "max_events", help = "maximum events per binary file", default = 10000, type = int)
args = parser.parse_args()

# check module ID
if args.mod_id < 0 or args.mod_id > 15:
    print('module ID is out of range.')
    exit(1)

# read calibration data
h5file_calib = h5py.File(args.calib_file, 'r')

