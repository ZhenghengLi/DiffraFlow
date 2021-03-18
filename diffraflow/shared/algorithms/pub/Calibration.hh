#ifndef __Calibration_H__
#define __Calibration_H__

#include "ImageDataField.hh"
#include "CalibDataField.hh"
#include <cuda_runtime.h>

namespace diffraflow {
    namespace Calibration {

        void do_calib_cpu(ImageDataField* image_data_host, CalibDataField* calib_data_host);
        void do_calib_gpu(
            ImageDataField* image_data_device, CalibDataField* calib_data_device, cudaStream_t stream = 0);

    } // namespace Calibration
} // namespace diffraflow

#endif