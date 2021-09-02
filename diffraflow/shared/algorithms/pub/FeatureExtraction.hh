#ifndef __FeatureExtraction_H__
#define __FeatureExtraction_H__

#include "ImageDataField.hh"
#include "ImageFeature.hh"
#include <cuda_runtime.h>

namespace diffraflow {
    namespace FeatureExtraction {
        // global mean
        void global_mean_cpu(ImageDataField* image_data_host, ImageFeature* image_feature_host, float min_energy = -100,
            float max_energy = 10000);
        void global_mean_gpu(cudaStream_t stream, ImageDataField* image_data_device, ImageFeature* image_feature_device,
            float min_energy = -100, float max_energy = 10000);
        // global rms
        void global_rms_cpu(ImageDataField* image_data_host, ImageFeature* image_feature_host, float min_energy = -100,
            float max_energy = 10000);
        void global_rms_gpu(cudaStream_t stream, ImageDataField* image_data_device, ImageFeature* image_feature_device,
            float min_energy = -100, float max_energy = 10000);
        // peak pixels
        void peak_pixels_MSSE_cpu(ImageDataField* image_data_host, ImageFeature* image_feature_host,
            float min_energy = -100, float max_energy = 10000, float inlier_thr = 2, float outlier_thr = 8,
            float min_residual = 50);
        void peak_pixels_MSSE_gpu(cudaStream_t stream, ImageDataField* image_data_device,
            ImageFeature* image_feature_device, float min_energy = -100, float max_energy = 10000, float inlier_thr = 2,
            float outlier_thr = 8, float min_residual = 50);
    } // namespace FeatureExtraction
} // namespace diffraflow

#endif