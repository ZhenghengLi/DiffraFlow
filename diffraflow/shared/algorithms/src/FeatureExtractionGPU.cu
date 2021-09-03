#include "FeatureExtraction.hh"

// global mean and rms
void diffraflow::FeatureExtraction::global_mean_rms_gpu(cudaStream_t stream, ImageDataField* image_data_device,
    ImageFeature* image_feature_device, float min_energy, float max_energy) {
    // atomicAdd all energy into image_feature_device->global_mean
    // devide image_feature_device->global_mean by count using a one-thread kernel
    return;
}

// peak pixels
void diffraflow::FeatureExtraction::peak_pixels_MSSE_gpu(cudaStream_t stream, ImageDataField* image_data_device,
    ImageFeature* image_feature_device, float min_energy, float max_energy, float inlier_thr, float outlier_thr,
    float min_residual) {
    //
    return;
}