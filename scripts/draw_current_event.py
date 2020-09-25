#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt
import requests
import msgpack

from cxidb_euxfel_utils import compose_image

parser = argparse.ArgumentParser(description='draw current event fetched from monitor')
parser.add_argument("source_url", help="url from which to fecth event")

args = parser.parse_args()

print("fetching event from url ", args.source_url, "...")
response = requests.get(args.source_url)
print(response.status_code)
print(response.headers)

content = response.content

print(type(content))
print(len(content))

image_vis_object = msgpack.unpackb(content)
image_frame_vec = image_vis_object[b'image_data'][b'image_frame_vec']
print(len(image_frame_vec[1]))
max_energy = 0
for m in range(16):
    for i in range(65536):
        energy = image_frame_vec[m][i]
        if energy > max_energy:
            max_energy = energy

print("max_energy = ", max_energy)

# image_data_feature = msgpack.unpackb(content)
# image_data = image_data_feature[b'image_data']
# image_frame_vec = image_data[b'image_frame_vec']
# # print(type(image_frame_vec[0][b'pixel_data'][0]))
# max_energy = -1000000
# min_energy = 1000000
# for m in range(16):
#     for i in range(65536):
#         energy = image_frame_vec[m][b'pixel_data'][i]
#         if energy > max_energy:
#             max_energy = energy
#         if energy < min_energy:
#             min_energy = energy
#
# print("max_energy = ", max_energy)
# print("min_energy = ", min_energy)
