#include "Calibration.hh"
#include <cuda_runtime.h>

__global__ void calib_kernel(diffraflow::ImageDataField* image_data, diffraflow::CalibDataField* calib_data) {
    int m = blockIdx.x;

    if (!image_data->alignment[m]) return;

    int h = blockIdx.y;
    int w = threadIdx.x;

    size_t l = image_data->gain_level[m][h][w];

    if (l < GLV_CNT) {
        image_data->pixel_data[m][h][w] =
            (image_data->pixel_data[m][h][w] - calib_data->pedestal[m][l][h][w]) * calib_data->gain[m][l][h][w];
    }
}

void diffraflow::Calibration::do_calib_gpu(
    cudaStream_t stream, ImageDataField* image_data_device, CalibDataField* calib_data_device) {

    // check arguments
    if (image_data_device == nullptr) return;
    if (calib_data_device == nullptr) return;

    // check calib level
    char calib_level = 0;
    cudaMemcpyAsync(&calib_level, &image_data_device->calib_level, 1, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    if (calib_level >= 1) return;

    // do calib
    calib_kernel<<<dim3(MOD_CNT, FRAME_H), FRAME_W, 0, stream>>>(image_data_device, calib_data_device);

    // set calib level
    calib_level = 1;
    cudaMemcpyAsync(&image_data_device->calib_level, &calib_level, 1, cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
}