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
pedestal_dset = h5file_calib['pedestal']
pedestal_data = pedestal_dset[args.mod_id]
gain_dset = h5file_calib['gain']
gain_data = gain_dset[args.mod_id]
threshold_dset = h5file_calib['threshold']
threshold_data = threshold_dset[args.mod_id]
h5file_calib.close()

# open alignment file
h5file_align = h5py.File(args.align_file, 'r')
align_idx_dset = h5file_align['alignment_index']

# open event number file
h5file_event = h5py.File(args.event_file, 'r')
event_num_dset = h5file_event['event_num']
event_num_len = event_num_dset.shape[0]

# iterate image data file and convert

det_path = '/INSTRUMENT/SPB_DET_AGIPD1M-1/DET'
h5file_data = None
current_file_idx = -1
cellId_dset = None
mask_dset = None
image_dset = None

sequential_id = -1
event_counts = -1

binary_outfile = None

for x in range(event_num_len):
    if (x % 1000 == 0):
        print("converting ", x)
    event_num = event_num_dset[x]
    index_vec = align_idx_dset[event_num][args.mod_id]
    if index_vec[0] < 0 or index_vec[1] < 0:
        print("unexpected index: ", index_vec)
        break
    if event_counts < 0 or event_counts >= args.max_events:
        if binary_outfile:
            binary_outfile.flush()
            binary_outfile.close()
        sequential_id += 1
        binary_fn = os.path.join(args.output_dir, 'AGIPD-BIN-R0243-M%02d-S%03d.dat' % (args.mod_id, sequential_id))
        binary_outfile = open(binary_fn, 'wb')
        event_counts = 0
    if current_file_idx != index_vec[0]:
        if h5file_data: h5file_data.close()
        h5file_data_fn = os.path.join(args.data_dir, 'CORR-R0243-AGIPD%02d-S%05d.h5')
        h5file_data = h5py.File(h5file_data_fn, 'r')
        cellId_dset = get_image_dset(h5file_data, det_path, 'cellId')
        mask_dset = get_image_dset(h5file_data, det_path, 'mask')
        image_dset = get_image_dset(h5file_data, det_path, 'data')
        current_file_idx = index_vec[0]
    cellId = cellId_dset[index_vec[1]]
    mask_data = mask_dset[index_vec[1]]
    image_data = np.nan_to_num(image_dset[index_vec[1]])
    image_data[mask_data > 0] = 0
    # header
    # image data
    for [idx, [row, col]] in enumerate(np.ndindex(512, 128)):
        energy = image_data[row, col]
        gain = 0
        ADC = 0
        if energy < threshold_data[0, row, col]:
            gain = 0
            ADC = energy * gain_data[0, row, col] + pedestal_data[0, row, col]
        elif energy < threshold_data[1, row, col]:
            gain = 1
            ADC = energy * gain_data[1, row, col] + pedestal_data[1, row, col]
        else:
            gain = 2
            ADC = energy * gain_data[2, row, col] + pedestal_data[2, row, col]
        if ADC < 0: ADC = 0
        if ADC > 16384: ADC = 16384
    # CRC
    # write one frame


h5file_event.close()
h5file_align.close()
if h5file_data: h5file_data.close()
if binary_outfile: binary_outfile.close()
