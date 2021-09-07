#include "FeatureExtraction.hh"
#include <cmath>

// global mean and rms =============================================================================================

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
                // apply energy cut
                if (energy < min_energy) {
                    energy = min_energy;
                } else if (energy > max_energy) {
                    energy = max_energy;
                }
                energy_sum += energy;
                energy_count++;
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
                // apply energy cut
                if (energy < min_energy) {
                    energy = min_energy;
                } else if (energy > max_energy) {
                    energy = max_energy;
                }
                double residual = energy - image_feature_host->global_mean;
                residual_sum += residual * residual;
                residual_count++;
            }
        }
    }
    if (residual_count > 0) {
        image_feature_host->global_rms = sqrt(residual_sum / residual_count);
    }
}

// peak pixels =====================================================================================================

void diffraflow::FeatureExtraction::peak_pixels_MSSE_cpu(ImageDataField* image_data_host,
    ImageFeature* image_feature_host, float min_energy, float max_energy, float inlier_thr, float outlier_thr,
    float residual_thr) {
    image_feature_host->peak_pixels = 0;
    for (int mod = 0; mod < MOD_CNT; mod++) {
        if (!image_data_host->alignment[mod]) continue;
        for (int blk = 0; blk < 8; blk++) {
            for (int row = 0; row < 2; row++) {
                for (int col = 0; col < 4; col++) {
                    int h_offset = row * 31;
                    int w_offset = col * 32;
                    // (1) global mean
                    double sum = 0;
                    for (int h = 0; h < 31; h++) {
                        for (int w = 0; w < 32; w++) {
                            double energy = image_data_host->pixel_data[mod][1 + h + h_offset + blk * 64][w + w_offset];
                            sum += energy;
                        }
                    }
                    double mean_global = sum / 992.0; // 31 * 32

                    // (2) global std
                    sum = 0;
                    for (int h = 0; h < 31; h++) {
                        for (int w = 0; w < 32; w++) {
                            double energy = image_data_host->pixel_data[mod][1 + h + h_offset + blk * 64][w + w_offset];
                            double residual = energy - mean_global;
                            sum += residual * residual;
                        }
                    }
                    double std_global = sqrt(sum / 991.0); // 31 * 32 - 1

                    // (3) inlier mean
                    double residual_max = std_global * inlier_thr;
                    sum = 0;
                    int count = 0;
                    for (int h = 0; h < 31; h++) {
                        for (int w = 0; w < 32; w++) {
                            double energy = image_data_host->pixel_data[mod][1 + h + h_offset + blk * 64][w + w_offset];
                            double residual_global = fabs(energy - mean_global);
                            if (residual_global < residual_max) {
                                sum += energy;
                                count++;
                            }
                        }
                    }
                    double mean_inlier(0);
                    if (count > 5) {
                        mean_inlier = sum / count;
                    } else {
                        return;
                    }

                    // (4) inlier std
                    sum = 0;
                    count = 0;
                    for (int h = 0; h < 31; h++) {
                        for (int w = 0; w < 32; w++) {
                            double energy = image_data_host->pixel_data[mod][1 + h + h_offset + blk * 64][w + w_offset];
                            double residual_global = fabs(energy - mean_global);
                            if (residual_global < residual_max) {
                                double residual_inlier = energy - mean_inlier;
                                sum += residual_inlier * residual_inlier;
                                count++;
                            }
                        }
                    }
                    double std_inlier(0);
                    if (count > 5) {
                        std_inlier = sqrt(sum / (count - 1));
                    } else {
                        return;
                    }

                    // (5) count pixels of outliers
                    count = 0;
                    double residual_min = std_inlier * outlier_thr;
                    if (residual_min < residual_thr) {
                        residual_min = residual_thr;
                    }
                    for (int h = 0; h < 31; h++) {
                        for (int w = 0; w < 32; w++) {
                            double energy = image_data_host->pixel_data[mod][1 + h + h_offset + blk * 64][w + w_offset];
                            double residual = energy - mean_inlier;
                            if (residual > residual_min) {
                                count++;
                            }
                        }
                    }
                    image_feature_host->peak_pixels += count;
                }
            }
        }
    }
}