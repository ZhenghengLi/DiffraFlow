#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt
import requests
import msgpack

from cxidb_euxfel_utils import compose_image

parser = argparse.ArgumentParser(description='draw current event that is fetched from monitor')
parser.add_argument("source_url", help="the URL from which to fetch event data")

args = parser.parse_args()

print("fetching event from url ", args.source_url, "...")
response = requests.get(args.source_url)

if (response.status_code != 200):
    print("failed to fetch current event with status code: ", response.status_code)
    print("headers:")
    for k in response.headers.keys():
        print(k, "=", response.headers[k])
    print("content:")
    print(str(response.content))
    exit(1)

print()
for k in response.headers.keys():
    print(k, "=", response.headers[k])
print()

content = response.content

image_data = np.empty((16, 512, 128))
image_data[:] = np.nan

image_vis_object = msgpack.unpackb(content)
bunch_id = image_vis_object[b'image_data'][b'bunch_id']
min_energy = image_vis_object[b'image_data'][b'min_energy']
max_energy = image_vis_object[b'image_data'][b'max_energy']
alignment_vec = image_vis_object[b'image_data'][b'alignment_vec']
image_frame_vec = image_vis_object[b'image_data'][b'image_frame_vec']

print("bunch_id = ", bunch_id)
print("min_energy = ", min_energy)
print("max_energy = ", max_energy)
print()

print("drawing ...")

for m in range(16):
    if not alignment_vec[m]:
        continue
    for pos in range(65536):
        energy = image_frame_vec[m][pos]
        h = int(pos / 128)
        w = int(pos % 128)
        image_data[m][h][w] = energy

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

cset1 = plt.imshow(full_image, cmap="rainbow", vmin=0, vmax=256)
plt.colorbar(cset1)

plt.tight_layout()
plt.show()
