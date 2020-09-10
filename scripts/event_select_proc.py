#!/usr/bin/env python3

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from cxidb_euxfel_utils import get_image_dset, compose_image

parser = argparse.ArgumentParser(description='view an event')
parser.add_argument("data_dir", help="directory that contains input data files")
parser.add_argument("align_file", help="alignment file")
parser.add_argument("outfile", help="output HDF5 file")
parser.add_argument("-s", dest="seg_count", help="segments count", default=3, type=int)
parser.add_argument("-c", dest="compress", help="compress level (0 -- 9)", default=5, type=int)
args = parser.parse_args()

align_idx_h5file = h5py.File(args.align_file, 'r')
align_idx_dset = align_idx_h5file['alignment_index']

align_idx_len = align_idx_dset.shape[0]

det_path = '/INSTRUMENT/SPB_DET_AGIPD1M-1/DET'
file_paths = [
    ['%s/CORR-R0243-AGIPD%02d-S%05d.h5' % (args.data_dir, x, y)
     for y in range(args.seg_count)] for x in range(16)]

last_index_mat = align_idx_dset[align_idx_len - 1]

for x in range(16):
    if last_index_mat[x][0] >= args.seg_count:
        print("file index is out of range for module", x)
        exit(1)

current_file_idx = [-1] * 16
image_h5files = [None] * 16
cellId_dsets = [None] * 16

h5file_out = h5py.File(args.outfile, 'w')
event_num_dset = h5file_out.create_dataset("event_num", (0,),
                                           maxshape=(None,), chunks=(1024,), dtype='int32', compression=args.compress)

for x in range(align_idx_len):
    if (x % 1000 == 0):
        print("checking ", x)
    index_mat = align_idx_dset[x]
    invalid_flag = False
    for d in range(16):
        if index_mat[d][0] < 0 or index_mat[d][1] < 0:
            invalid_flag = True
            break
        if current_file_idx[d] != index_mat[d][0]:
            if image_h5files[d]:
                image_h5files[d].close()
            image_h5files[d] = h5py.File(file_paths[d][index_mat[d][0]], 'r')
            cellId_dsets[d] = get_image_dset(image_h5files[d], det_path, 'cellId')
            current_file_idx[d] = index_mat[d][0]
        cellId = cellId_dsets[d][index_mat[d][1]]
        if cellId % 2 == 1 or cellId == 0 or cellId == 62:
            invalid_flag = True
            break
    if not invalid_flag:
        dim1_len = event_num_dset.shape[0]
        event_num_dset.resize((dim1_len + 1,))
        event_num_dset[-1] = x

h5file_out.flush()
h5file_out.close()
