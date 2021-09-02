#include "FeatureExtraction.hh"

// global mean
void diffraflow::FeatureExtraction::global_mean_gpu(cudaStream_t stream, ImageDataField* image_data_device,
    ImageFeature* image_feature_device, float min_energy, float max_energy) {
    // atomicAdd all energy into image_feature_device->global_mean
    // devide image_feature_device->global_mean by count using a one-thread kernel
    return;
}

// global rms
void diffraflow::FeatureExtraction::global_rms_gpu(cudaStream_t stream, ImageDataField* image_data_device,
    ImageFeature* image_feature_device, float min_energy, float max_energy) {
    // invoke this function only after global_mean_gpu
    // atomicAdd all (energy - mean)^2 into image_feature_device->global_rms
    // devide image_feature_device->global_rms by count using a one-thread kernel
    return;
}

// peak pixels
void diffraflow::FeatureExtraction::peak_pixels_MSSE_gpu(cudaStream_t stream, ImageDataField* image_data_device,
    ImageFeature* image_feature_device, float min_energy, float max_energy, float inlier_thr, float outlier_thr,
    float min_residual) {
    //
    return;
}