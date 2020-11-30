#!/usr/bin/env python3

import argparse
import h5py
import numpy as np
import sys
import os

parser = argparse.ArgumentParser(description='get event number from bunch id')
parser.add_argument("event_num_file", help="event number file")
parser.add_argument("bunch_id", help="bunch id", type=int)
args = parser.parse_args()

event_num_h5file = h5py.File(args.event_num_file, 'r')
event_num_dset = event_num_h5file['event_num']

if args.bunch_id < 0:
    print('bunch_id < 0')
    exit(1)

if args.bunch_id >= event_num_dset.shape[0]:
    print("bunch_id is too large.")
    exit(1)

event_num = event_num_dset[args.bunch_id]

print('event_num = ', event_num)
