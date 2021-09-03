#include "FeatureExtraction.hh"

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
            if (energy >= min_energy && energy < max_energy) {
                sum_local += energy;
                count_local++;
            }
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
            if (energy >= min_energy && energy < max_energy) {
                double residual = energy - mean;
                sum_local += residual * residual;
                count_local++;
            }
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

// global mean and rms
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
    divide_init_kernel<<<1, 1, 0, stream>>>(&image_feature_device->global_rms, sum_device, count_device);
    // wait finish
    cudaStreamSynchronize(stream);
}

// peak pixels
void diffraflow::FeatureExtraction::peak_pixels_MSSE_gpu(cudaStream_t stream, ImageDataField* image_data_device,
    ImageFeature* image_feature_device, float min_energy, float max_energy, float inlier_thr, float outlier_thr,
    float min_residual) {
    //
    return;
}