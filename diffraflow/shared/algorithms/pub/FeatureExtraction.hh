#ifndef __FeatureExtraction_H__
#define __FeatureExtraction_H__

#include "ImageDataField.hh"
#include "ImageFeature.hh"
#include <cuda_runtime.h>

namespace diffraflow {
    namespace FeatureExtraction {

        // global mean and rms
        void global_mean_rms_cpu(ImageDataField* image_data_host, ImageFeature* image_feature_host,
            float min_energy = -10, float max_energy = 1000);
        void global_mean_rms_gpu(cudaStream_t stream, double* sum_device, int* count_device,
            ImageDataField* image_data_device, ImageFeature* image_feature_device, float min_energy = -10,
            float max_energy = 1000);

        // peak pixels parameter
        struct PeakPixelParams {
            float min_energy;
            float max_energy;
            float inlier_thr;
            float outlier_thr;
            float residual_thr;
            float energy_thr;
        };

        // peak pixels
        void peak_pixels_MSSE_cpu(
            ImageFeature* image_feature_host, ImageDataField* image_data_host, PeakPixelParams params);
        void peak_pixels_MSSE_gpu(cudaStream_t stream, ImageFeature* image_feature_device,
            ImageDataField* image_data_device, PeakPixelParams params);

    } // namespace FeatureExtraction
} // namespace diffraflow

#endif