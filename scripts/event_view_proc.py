#!/usr/bin/env python3

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from cxidb_euxfel_utils import get_image_dset, compose_image

parser = argparse.ArgumentParser(description='view an event')
parser.add_argument("data_dir", help = "directory that contains input data files")
parser.add_argument("align_file", help = "alignment file")
parser.add_argument("start_event", help = "start event", type=int)
parser.add_argument("cell_offset", help = "cell offset", type=int)
parser.add_argument("-s", dest = "seg_count", help = "segments count", default=3, type=int)
parser.add_argument("-a", dest = "min_z", help = "minimum z value", default=-100, type=int)
parser.add_argument("-b", dest = "max_z", help = "maximum z value", default=1200, type=int)
args = parser.parse_args()

event_num = args.start_event + args.cell_offset * 64

align_idx_h5file = h5py.File(args.align_file, 'r')
align_idx_dset = align_idx_h5file['alignment_index']

if event_num < 0:
    print("event_num < 0")
    exit(1)

if event_num >= align_idx_dset.shape[0]:
    print("event_num is too large.")
    exit(1)

index_mat = align_idx_dset[event_num]

align_idx_h5file.close()

det_path = '/INSTRUMENT/SPB_DET_AGIPD1M-1/DET'
file_paths = [
    ['%s/CORR-R0243-AGIPD%02d-S%05d.h5' % (args.data_dir, x, y)
    for y in range(args.seg_count)] for x in range(16) ]

for x in range(16):
    if index_mat[x][0] >= args.seg_count:
        print("file index is out of range for module", x)
        exit(1)

image_h5files = [h5py.File(file_paths[x][index_mat[x][0]], 'r') for x in range(16)]
image_dsets = [get_image_dset(x, det_path, 'data') for x in image_h5files]
trainId_dsets = [get_image_dset(x, det_path, 'trainId') for x in image_h5files]
pulseId_dsets = [get_image_dset(x, det_path, 'pulseId') for x in image_h5files]
cellId_dsets = [get_image_dset(x, det_path, 'cellId') for x in image_h5files]
mask_dsets = [get_image_dset(x, det_path, 'mask') for x in image_h5files]

image_data = np.empty( (16, 512, 128) )
image_data[:] = np.nan
trainId_arr = [-1 for x in range(16)]
pulseId_arr = [-1 for x in range(16)]
cellId_arr = [-1 for x in range(16)]

for x in range(16):
    cur_idx = index_mat[x][1]
    if cur_idx < 0: continue
    trainId_arr[x] = trainId_dsets[x][cur_idx][0]
    pulseId_arr[x] = pulseId_dsets[x][cur_idx][0]
    cellId_arr[x] = cellId_dsets[x][cur_idx]
    mask_data = mask_dsets[x][cur_idx]
    with image_dsets[x].astype('float64'):
        image_data[x] = np.nan_to_num(image_dsets[x][cur_idx])
        image_data[x][mask_data > 0] = 0

for x in range(16):
    image_h5files[x].close()

signal_sum = np.sum(image_data)
signal_mean = signal_sum / (1024. * 1024)

print("signal sum:", signal_sum)
print("signal mean:", signal_mean)

print("trainId:", trainId_arr)
print("pulseId:", pulseId_arr)
print("cellId:", cellId_arr)

image_size = (1300, 1300)

offset_1 = 26
offset_2 = 4

quad_offset = [
    (-offset_1, offset_2),
    (offset_2, offset_1),
    (offset_1, -offset_2),
    (-offset_2, -offset_1)
]

mod_gap = 30

full_image = compose_image(image_data, image_size, quad_offset, mod_gap)

cset1 = plt.imshow(full_image, cmap = "rainbow", vmin = args.min_z, vmax = args.max_z)
plt.colorbar(cset1)

plt.tight_layout()
plt.show()
