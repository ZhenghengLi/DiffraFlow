#!/usr/bin/env python3

import os
import numpy as np


def get_image_dset(h5file, det_path, name):
    ch_name = list(h5file[det_path].keys())[0]
    return h5file[os.path.join(det_path, ch_name, 'image', name)]


def _tailor_space(ld_pos, ru_pos, h, w):
    # yc:yd, xc:xd
    ya, yb, xa, xb = ru_pos[0], ld_pos[0], ld_pos[1], ru_pos[1]
    yc, yd, xc, xd = 0, yb - ya, 0, xb - xa
    if yd <= 0 or yd > h or xd <= 0 or xd > w:
        return None
    # ya:yb
    if yb < 0:
        return None
    if yb > h:
        yd -= yb - h
        yb = h
    if ya > h:
        return None
    if ya < 0:
        yc -= ya
        ya = 0
    # xa:xb
    if xa > w:
        return None
    if xa < 0:
        xc -= xa
        xa = 0
    if xb < 0:
        return None
    if xb > w:
        xd -= xb - w
        xb = w
    return [ya, yb, xa, xb, yc, yd, xc, xd]


def compose_image(data, image_size=(1024, 1024),
                  quad_offset=[(0, 0), (0, 0), (0, 0), (0, 0)], mod_gap=4):
    if data.shape != (16, 512, 128):
        print('wrong data shape.')
        return
    h, w = image_size
    if h < 800 or w < 800:
        print('image size is too small.')
        return
    # add ASIC gap with 1 pixel for each module
    mod_data = []
    for x in data:
        insert_pos = np.array([[64 + it * 64] * 2 for it in range(7)]).flatten()
        mod_data.append(np.insert(x, insert_pos, np.nan, axis=0))
    full_image = np.empty(image_size)
    full_image[:] = np.nan
    # assemble module data
    center = np.array([h / 2, w / 2], dtype=int)
    # Q1
    for x in range(4):
        offset = np.array(quad_offset[0])
        ld_pos = center + offset + [-(128 + mod_gap) * (3 - x), 0]
        ru_pos = ld_pos + [-128, 526]
        space = _tailor_space(ld_pos, ru_pos, h, w)
        if space:
            ya, yb, xa, xb, yc, yd, xc, xd = space
            mod_image = np.rot90(mod_data[x], 3)
            full_image[ya:yb, xa:xb] = mod_image[yc:yd, xc:xd]
    # Q2
    for x in range(4):
        offset = np.array(quad_offset[1])
        ld_pos = center + offset + [128 + (128 + mod_gap) * x, 0]
        ru_pos = ld_pos + [-128, 526]
        space = _tailor_space(ld_pos, ru_pos, h, w)
        if space:
            ya, yb, xa, xb, yc, yd, xc, xd = space
            mod_image = np.rot90(mod_data[x + 4], 3)
            full_image[ya:yb, xa:xb] = mod_image[yc:yd, xc:xd]
    # Q3
    for x in range(4):
        offset = np.array(quad_offset[2])
        ru_pos = center + offset + [(128 + mod_gap) * x, 0]
        ld_pos = ru_pos + [128, -526]
        space = _tailor_space(ld_pos, ru_pos, h, w)
        if space:
            ya, yb, xa, xb, yc, yd, xc, xd = space
            mod_image = np.rot90(mod_data[x + 8], 1)
            full_image[ya:yb, xa:xb] = mod_image[yc:yd, xc:xd]
    # Q4
    for x in range(4):
        offset = np.array(quad_offset[3])
        ru_pos = center + offset + [-128 - (128 + mod_gap) * (3 - x), 0]
        ld_pos = ru_pos + [128, -526]
        space = _tailor_space(ld_pos, ru_pos, h, w)
        if space:
            ya, yb, xa, xb, yc, yd, xc, xd = space
            mod_image = np.rot90(mod_data[x + 12], 1)
            full_image[ya:yb, xa:xb] = mod_image[yc:yd, xc:xd]
    return full_image
