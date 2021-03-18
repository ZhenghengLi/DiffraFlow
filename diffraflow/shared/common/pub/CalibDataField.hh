#ifndef __CalibDataField_H__
#define __CalibDataField_H__

#include "ImageDimension.hh"

namespace diffraflow {
    struct CalibDataField {
        float gain[MOD_CNT][GLV_CNT][FRAME_H][FRAME_W];     // keV/ADC
        float pedestal[MOD_CNT][GLV_CNT][FRAME_H][FRAME_W]; // ADC
    };

} // namespace diffraflow

#endif