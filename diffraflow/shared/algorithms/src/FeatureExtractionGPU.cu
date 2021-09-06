#include "FeatureExtraction.hh"

// global mean and rms =============================================================================================

__global__ void energy_sum_kernel(diffraflow::ImageDataField* image_data_device, double* sum_device, int* count_device,
    float min_energy, float max_energy) {
    int mod = blockIdx.x;  // module
    int row = threadIdx.x; // ASIC row
    int col = threadIdx.y; // ASIC column

    if (!image_data_device->alignment[mod]) return;

    double sum_local = 0;
    int count_local = 0;

    for (int h_asic = 1; h_asic < 63; h_asic++) {
        for (int w_asic = 0; w_asic < 64; w_asic++) {
            float energy = image_data_device->pixel_data[mod][row * 64 + h_asic][col * 64 + w_asic];
            // apply energy cut
            if (energy < min_energy) {
                energy = min_energy;
            } else if (energy > max_energy) {
                energy = max_energy;
            }
            sum_local += energy;
            count_local++;
        }
    }

    atomicAdd(sum_device, sum_local);
    atomicAdd(count_device, count_local);
}

__global__ void mean_square_sum_kernel(float mean, diffraflow::ImageDataField* image_data_device, double* sum_device,
    int* count_device, float min_energy, float max_energy) {
    int mod = blockIdx.x;  // module
    int row = threadIdx.x; // ASIC row
    int col = threadIdx.y; // ASIC column

    if (!image_data_device->alignment[mod]) return;

    double sum_local = 0;
    int count_local = 0;

    for (int h_asic = 1; h_asic < 63; h_asic++) {
        for (int w_asic = 0; w_asic < 64; w_asic++) {
            float energy = image_data_device->pixel_data[mod][row * 64 + h_asic][col * 64 + w_asic];
            // apply energy cut
            if (energy < min_energy) {
                energy = min_energy;
            } else if (energy > max_energy) {
                energy = max_energy;
            }
            double residual = energy - mean;
            sum_local += residual * residual;
            count_local++;
        }
    }

    atomicAdd(sum_device, sum_local);
    atomicAdd(count_device, count_local);
}

__global__ void divide_init_kernel(float* dst_device, double* sum_device, int* count_device) {
    if (*count_device > 0) {
        *dst_device = *sum_device / *count_device;
    }
    *sum_device = 0;
    *count_device = 0;
}

__global__ void divide_root_init_kernel(float* dst_device, double* sum_device, int* count_device) {
    if (*count_device > 0) {
        *dst_device = sqrt(*sum_device / *count_device);
    }
    *sum_device = 0;
    *count_device = 0;
}

void diffraflow::FeatureExtraction::global_mean_rms_gpu(cudaStream_t stream, double* sum_device, int* count_device,
    ImageDataField* image_data_device, ImageFeature* image_feature_device, float min_energy, float max_energy) {
    // init
    double sum_host = 0;
    cudaMemcpyAsync(sum_device, &sum_host, sizeof(double), cudaMemcpyHostToDevice, stream);
    int count_host = 0;
    cudaMemcpyAsync(count_device, &count_host, sizeof(int), cudaMemcpyHostToDevice, stream);
    // mean
    energy_sum_kernel<<<MOD_CNT, dim3(8, 2), 0, stream>>>(
        image_data_device, sum_device, count_device, min_energy, max_energy);
    divide_init_kernel<<<1, 1, 0, stream>>>(&image_feature_device->global_mean, sum_device, count_device);
    float mean = 0;
    cudaMemcpyAsync(&mean, &image_feature_device->global_mean, sizeof(float), cudaMemcpyDeviceToHost, stream);
    // rms
    mean_square_sum_kernel<<<MOD_CNT, dim3(8, 2), 0, stream>>>(
        mean, image_data_device, sum_device, count_device, min_energy, max_energy);
    divide_root_init_kernel<<<1, 1, 0, stream>>>(&image_feature_device->global_rms, sum_device, count_device);
    // wait finish
    cudaStreamSynchronize(stream);
}

// peak pixels ====================================================================================================

__global__ void peak_pixels_kernel(diffraflow::ImageDataField* image_data_device,
    diffraflow::ImageFeature* image_feature_device, float min_energy, float max_energy, float inlier_thr,
    float outlier_thr, float min_residual) {
    int mod = blockIdx.x;  // module
    int blk = blockIdx.y;  // ASIC block
    int row = threadIdx.x; // grid row
    int col = threadIdx.y; // grid column

    if (!image_data_device->alignment[mod]) return;

    __shared__ float energy_cache[62][128];

    // (0) copy energy from global memory to shared memory
    // 31 * 32 grid for each thread
    for (int h = 0; h < 31; h++) {
        int hc = h + row * 31;
        int hg = 1 + hc + blk * 64;
        for (int w = 0; w < 32; w++) {
            int wc = w + col * 32;
            int wg = wc;
            float energy = image_data_device->pixel_data[mod][hg][wg];
            // apply energy cut
            if (energy < min_energy) {
                energy = min_energy;
            } else if (energy > max_energy) {
                energy = max_energy;
            }
            energy_cache[hc][wc] = energy;
        }
    }

    // (1) global mean
    double sum = 0;
    for (int h = 0; h < 31; h++) {
        for (int w = 0; w < 32; w++) {
            double energy = energy_cache[h + row * 31][w + col * 32];
            sum += energy;
        }
    }
    double mean_global = sum / 992.0; // 31 * 32

    // (2) global rms
    sum = 0;
    for (int h = 0; h < 31; h++) {
        for (int w = 0; w < 32; w++) {
            double energy = energy_cache[h + row * 31][w + col * 32];
            double residual = energy - mean_global;
            sum += residual * residual;
        }
    }
    double rms_global = sqrt(sum / 992.0); // 31 * 32

    // (3) inlier mean
    double residual_max = rms_global * inlier_thr;
    sum = 0;
    int count = 0;
    for (int h = 0; h < 31; h++) {
        for (int w = 0; w < 32; w++) {
            double energy = energy_cache[h + row * 31][w + col * 32];
            double residual_global = abs(energy - mean_global);
            if (residual_global < residual_max) {
                sum += energy;
                count++;
            }
        }
    }
    double mean_inlier(0);
    if (count > 0) {
        mean_inlier = sum / count;
    } else {
        return;
    }

    // (4) inlier rms
    sum = 0;
    count = 0;
    for (int h = 0; h < 31; h++) {
        for (int w = 0; w < 32; w++) {
            double energy = energy_cache[h + row * 31][w + col * 32];
            double residual_global = abs(energy - mean_global);
            if (residual_global < residual_max) {
                double residual_inlier = energy - mean_inlier;
                sum += residual_inlier * residual_inlier;
                count++;
            }
        }
    }
    double rms_inlier(0);
    if (count > 0) {
        rms_inlier = sqrt(sum / count);
    } else {
        return;
    }

    // (5) count pixels of outliers
    count = 0;
    double residual_min = rms_inlier * outlier_thr;
    for (int h = 0; h < 31; h++) {
        for (int w = 0; w < 32; w++) {
            double energy = energy_cache[h + row * 31][w + col * 32];
            double residual = energy - mean_inlier;
            if (residual > residual_min) {
                count++;
            }
        }
    }
    atomicAdd(&image_feature_device->peak_pixels, count);
}

void diffraflow::FeatureExtraction::peak_pixels_MSSE_gpu(cudaStream_t stream, ImageDataField* image_data_device,
    ImageFeature* image_feature_device, float min_energy, float max_energy, float inlier_thr, float outlier_thr,
    float min_residual) {
    int peak_pixels_host = 0;
    cudaMemcpyAsync(&image_feature_device->peak_pixels, &peak_pixels_host, sizeof(int), cudaMemcpyHostToDevice, stream);
    peak_pixels_kernel<<<dim3(16, 8), dim3(2, 4), 0, stream>>>(
        image_data_device, image_feature_device, min_energy, max_energy, inlier_thr, outlier_thr, min_residual);
    cudaStreamSynchronize(stream);
}
