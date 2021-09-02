#include "FeatureExtraction.hh"

// global mean
void diffraflow::FeatureExtraction::global_mean_gpu(cudaStream_t stream, ImageDataField* image_data_host,
    ImageFeature* image_feature_host, float min_energy, float max_energy) {
    //
    return;
}

// global rms
void diffraflow::FeatureExtraction::global_rms_gpu(cudaStream_t stream, ImageDataField* image_data_host,
    ImageFeature* image_feature_host, float min_energy, float max_energy) {
    //
    return;
}

// peak pixels
void diffraflow::FeatureExtraction::peak_pixels_MSSE_gpu(cudaStream_t stream, ImageDataField* image_data_host,
    ImageFeature* image_feature_host, float min_energy, float max_energy, float inlier_thr, float outlier_thr,
    float min_residual) {
    //
    return;
}