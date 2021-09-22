#ifndef __FeatureExtraction_H__
#define __FeatureExtraction_H__

#include "ImageDataField.hh"
#include "ImageFeature.hh"
#include <cuda_runtime.h>

namespace diffraflow {
    namespace FeatureExtraction {

        // global mean and rms
        void global_mean_rms_cpu(ImageFeature* image_feature_host, const ImageDataField* image_data_host,
            const float min_energy = -10, const float max_energy = 1000);
        void global_mean_rms_gpu(cudaStream_t stream, double* sum_device, int* count_device,
            ImageFeature* image_feature_device, const ImageDataField* image_data_device, const float min_energy = -10,
            const float max_energy = 1000);

        // peak pixels parameter
        struct PeakMsseParams {
            float min_energy;
            float max_energy;
            float inlier_thr;
            float outlier_thr;
            float residual_thr;
            float energy_thr;
        };

        // peak pixels
        void peak_pixels_MSSE_cpu(
            ImageFeature* image_feature_host, const ImageDataField* image_data_host, const PeakMsseParams& params);
        void peak_pixels_MSSE_gpu(cudaStream_t stream, ImageFeature* image_feature_device,
            const ImageDataField* image_data_device, const PeakMsseParams& params);

    } // namespace FeatureExtraction
} // namespace diffraflow

#endif