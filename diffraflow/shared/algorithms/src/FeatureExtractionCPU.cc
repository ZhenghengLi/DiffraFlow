#include "FeatureExtraction.hh"
#include <cmath>

// global mean and rms
void diffraflow::FeatureExtraction::global_mean_rms_cpu(
    ImageDataField* image_data_host, ImageFeature* image_feature_host, float min_energy, float max_energy) {
    double energy_sum = 0;
    int energy_count = 0;
    for (int m = 0; m < MOD_CNT; m++) {
        if (!image_data_host->alignment[m]) continue;
        for (int h = 0; h < FRAME_H; h++) {
            int h_asic = h % 64;
            if (h_asic == 0 || h_asic == 63) continue;
            for (int w = 0; w < FRAME_W; w++) {
                float energy = image_data_host->pixel_data[m][h][w];
                if (energy >= min_energy && energy < max_energy) {
                    energy_sum += energy;
                    energy_count++;
                }
            }
        }
    }
    if (energy_count > 0) {
        image_feature_host->global_mean = energy_sum / energy_count;
    }
    double residual_sum = 0;
    int residual_count = 0;
    for (int m = 0; m < MOD_CNT; m++) {
        if (!image_data_host->alignment[m]) continue;
        for (int h = 0; h < FRAME_H; h++) {
            int h_asic = h % 64;
            if (h_asic == 0 || h_asic == 63) continue;
            for (int w = 0; w < FRAME_W; w++) {
                float energy = image_data_host->pixel_data[m][h][w];
                if (energy >= min_energy && energy < max_energy) {
                    double residual = energy - image_feature_host->global_mean;
                    residual_sum += residual * residual;
                    residual_count++;
                }
            }
        }
    }
    if (residual_count > 0) {
        image_feature_host->global_rms = sqrt(residual_sum / residual_count);
    }
}

// peak pixels
void diffraflow::FeatureExtraction::peak_pixels_MSSE_cpu(ImageDataField* image_data_host,
    ImageFeature* image_feature_host, float min_energy, float max_energy, float inlier_thr, float outlier_thr,
    float min_residual) {
    //
    return;
}