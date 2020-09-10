#!/usr/bin/env python3

import argparse
import h5py
import numpy as np
import scipy as sp
import sys
import os
import random

parser = argparse.ArgumentParser(
    description='generate calibraion file which contains pedestal, gain and gain level threshold')
parser.add_argument("outfile", help="output HDF5 file")
parser.add_argument("-s", dest="seed", help="random seed", default=0, type=int)
parser.add_argument("-c", dest="compress", help="compress level (0 -- 9)", default=5, type=int)
args = parser.parse_args()

random.seed(args.seed)

ped_mu = [5000, 7000, 8000]
ped_sigma = [500, 500, 500]
ped_max = [5500, 7500, 8500]
ped_min = [4500, 6500, 7500]

gain_mu = [8, 0.35, 0.05]
gain_sigma = [0.5, 0.03, 0.003]
gain_max = [8.5, 0.4, 0.055]
gain_min = [7.5, 0.3, 0.045]

threshold_mu = [1000, 20000]
threshold_sigma = [100, 600]
threshold_max = [1200, 21000]
threshold_min = [800, 19000]

h5file_out = h5py.File(args.outfile, 'w')

pedestal_dset = h5file_out.create_dataset('pedestal', (16, 3, 512, 128), dtype='float32', compression=args.compress)
pedestal_dset.attrs['unit'] = 'ADC'
pedestal_data = np.zeros((16, 3, 512, 128), dtype='float32')

gain_dset = h5file_out.create_dataset('gain', (16, 3, 512, 128), dtype='float32', compression=args.compress)
gain_dset.attrs['unit'] = 'ADC/keV'
gain_data = np.zeros((16, 3, 512, 128), dtype='float32')

threshold_dset = h5file_out.create_dataset('threshold', (16, 2, 512, 128), dtype='float32', compression=args.compress)
threshold_dset.attrs['unit'] = 'keV'
threshold_data = np.zeros((16, 2, 512, 128), dtype='float32')

print("calculating pedestal and gain ...")

for det, level in np.ndindex(16, 3):
    pedestal_arr = np.empty((512, 128))
    gain_arr = np.empty((512, 128))
    for x, y in np.ndindex(512, 128):
        while True:
            ped = random.gauss(ped_mu[level], ped_sigma[level])
            if ped <= ped_max[level] and ped >= ped_min[level]:
                break
        pedestal_arr[x, y] = ped
        while True:
            gain = random.gauss(gain_mu[level], gain_sigma[level])
            if gain <= gain_max[level] and gain >= gain_min[level]:
                break
        gain_arr[x, y] = gain
    pedestal_data[det, level] = pedestal_arr
    gain_data[det, level] = gain_arr

pedestal_dset[:] = pedestal_data
gain_dset[:] = gain_data

print("calculating threshold ...")

for det, edge in np.ndindex(16, 2):
    threshold_arr = np.empty((512, 128))
    for x, y in np.ndindex(512, 128):
        while True:
            thr = random.gauss(threshold_mu[edge], threshold_sigma[edge])
            if thr <= threshold_max[edge] and thr >= threshold_min[edge]:
                break
        threshold_arr[x, y] = thr
    threshold_data[det, edge] = threshold_arr

threshold_dset[:] = threshold_data


ADC_range_1_data = np.zeros((16, 2, 512, 128), dtype='int16')
ADC_range_2_data = np.zeros((16, 2, 512, 128), dtype='int16')
ADC_range_3_data = np.zeros((16, 2, 512, 128), dtype='int16')

print("calculating range 1 ...")

overflow_count = 0
ADC_max = 2**14

# range 1
for det in range(16):
    ADC_min_arr = np.empty((512, 128))
    ADC_max_arr = np.empty((512, 128))
    for x, y in np.ndindex(512, 128):
        ped = pedestal_data[det, 0, x, y]
        gain = gain_data[det, 0, x, y]
        energy_min = 0
        energy_max = threshold_data[det, 0, x, y]
        ADC_min_arr[x, y] = ped + gain * energy_min
        ADC_max_arr[x, y] = ped + gain * energy_max
        if ADC_max_arr[x, y] > ADC_max:
            overflow_count += 1
    ADC_range_1_data[det, 0] = ADC_min_arr
    ADC_range_1_data[det, 1] = ADC_max_arr

print("calculating range 2 ...")

# range 2
for det in range(16):
    ADC_min_arr = np.empty((512, 128))
    ADC_max_arr = np.empty((512, 128))
    for x, y in np.ndindex(512, 128):
        ped = pedestal_data[det, 1, x, y]
        gain = gain_data[det, 1, x, y]
        energy_min = threshold_data[det, 0, x, y]
        energy_max = threshold_data[det, 1, x, y]
        ADC_min_arr[x, y] = ped + gain * energy_min
        ADC_max_arr[x, y] = ped + gain * energy_max
        if ADC_max_arr[x, y] > ADC_max:
            overflow_count += 1
    ADC_range_2_data[det, 0] = ADC_min_arr
    ADC_range_2_data[det, 1] = ADC_max_arr

print("calculating range 3 ...")

# range 3
for det in range(16):
    ADC_min_arr = np.empty((512, 128))
    ADC_max_arr = np.empty((512, 128))
    for x, y in np.ndindex(512, 128):
        ped = pedestal_data[det, 2, x, y]
        gain = gain_data[det, 2, x, y]
        energy_min = threshold_data[det, 1, x, y]
        energy_max = 48000
        ADC_min_arr[x, y] = ped + gain * energy_min
        ADC_max_arr[x, y] = ped + gain * energy_max
        if ADC_max_arr[x, y] > ADC_max:
            overflow_count += 1
    ADC_range_3_data[det, 0] = ADC_min_arr
    ADC_range_3_data[det, 1] = ADC_max_arr

print('overflow_count:', overflow_count)


ADC_range_1_dset = h5file_out.create_dataset('ADC_range_1', (16, 2, 512, 128), dtype='int16', compression=args.compress)
ADC_range_2_dset = h5file_out.create_dataset('ADC_range_2', (16, 2, 512, 128), dtype='int16', compression=args.compress)
ADC_range_3_dset = h5file_out.create_dataset('ADC_range_3', (16, 2, 512, 128), dtype='int16', compression=args.compress)

ADC_range_1_dset[:] = ADC_range_1_data
ADC_range_2_dset[:] = ADC_range_2_data
ADC_range_3_dset[:] = ADC_range_3_data


h5file_out.flush()
h5file_out.close()
