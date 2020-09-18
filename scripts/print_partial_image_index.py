#!/usr/bin/env python3

import argparse
import h5py
import numpy as np
import sys
import os

parser = argparse.ArgumentParser(description='print index of all partial images in a file')
parser.add_argument("data_file", help="data file")
args = parser.parse_args()

h5file = h5py.File(args.data_file, 'r')
image_data_dset = h5file['image_data']

total_entries = image_data_dset.shape[0]

for x in range(total_entries):
    bunch_id = image_data_dset[x]['bunch_id']
    alignment = image_data_dset[x]['alignment']
    if np.sum(alignment) != 16:
        print(x, ":", bunch_id, ":", list(alignment))

h5file.close()
