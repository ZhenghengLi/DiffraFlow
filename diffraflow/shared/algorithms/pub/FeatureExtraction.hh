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

        // peak pixels
        void peak_pixels_MSSE_cpu(ImageDataField* image_data_host, ImageFeature* image_feature_host,
            float min_energy = -10, float max_energy = 1000, float inlier_thr = 2, float outlier_thr = 8,
            float min_residual = 20);
        void peak_pixels_MSSE_gpu(cudaStream_t stream, ImageDataField* image_data_device,
            ImageFeature* image_feature_device, float min_energy = -10, float max_energy = 1000, float inlier_thr = 2,
            float outlier_thr = 8, float min_residual = 20);

    } // namespace FeatureExtraction
} // namespace diffraflow

#endif