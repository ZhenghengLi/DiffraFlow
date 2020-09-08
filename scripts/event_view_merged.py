#!/usr/bin/env python3

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from cxidb_euxfel_utils import get_image_dset, compose_image

parser = argparse.ArgumentParser(description='view an event')
parser.add_argument("data_file", help="data file")
parser.add_argument("event_offset", help="event offset", type=int)

parser.add_argument("-a", dest="min_z", help="minimum z value", default=-100, type=int)
parser.add_argument("-b", dest="max_z", help="maximum z value", default=1200, type=int)
args = parser.parse_args()


image_data = np.empty((16, 512, 128))
image_data[:] = np.nan

h5file = h5py.File(args.data_file, 'r')
image_data_dset = h5file['image_data']
image_data_obj = image_data_dset[args.event_offset]
image_data = image_data_obj['pixel_data'] * 10.0

signal_sum = np.sum(image_data)
signal_mean = signal_sum / (1024. * 1024)

print("signal sum:", signal_sum)
print("signal mean:", signal_mean)
print("bunch_id:", image_data_obj['bunch_id'])
print("cell_id:", image_data_obj['cell_id'])

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

cset1 = plt.imshow(full_image, cmap="rainbow", vmin=args.min_z, vmax=args.max_z)
plt.colorbar(cset1)

plt.tight_layout()
plt.show()
